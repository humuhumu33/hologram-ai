//! The prism contract: a canonical, admission-checked manifest for every
//! compiled `.holo` archive (V&V row `prism-canonical-manifest`).
//!
//! [`prism-archive`](https://github.com/humuhumu33/hologram-ai-prism)
//! defines `hologram-ai/archive/1`: one canonical JSON value describing an
//! archive's tensors, κ weight bindings, component graph, and
//! content-addressed provenance. Its identity is `blake3:<hex>` of the
//! exact canonical bytes, so browser and native compiles of the same model
//! can be compared by κ alone. Admission (`strict decode → normalize →
//! admit`) runs validators whose bodies are extracted from a
//! kernel-verified Lean module (empty observed axiom sets; see that
//! repository's SPEC.md and VERIFICATION.md).
//!
//! The manifest is baked into the archive as the open extension section
//! [`PRISM_MANIFEST_EXT`], exactly like the tokenizer: the `.holo` stays
//! self-describing, and [`crate::runner::HoloRunner`] refuses an archive
//! whose baked manifest does not admit. Archives without the section are
//! accepted unchanged (pre-contract archives remain loadable); the
//! dictionary row states exactly that.

use anyhow::{anyhow, Context, Result};
use hologram_ai_common::{AiGraph, AiParam};
use prism_archive::dto::{
    Component, Graph as PrismGraph, Manifest, Node, Provenance, Tensor, Weight,
};
use std::collections::BTreeSet;

/// Archive extension key for the canonical prism manifest bytes.
pub const PRISM_MANIFEST_EXT: &str = "prism.manifest";

/// Content-addressed provenance for a manifest. All labels are
/// `<axis>:<64 hex>` κ-labels on the registered `blake3`/`sha256` axes.
pub struct PrismProvenance {
    /// κ of the source model's canonical `config.json`.
    pub config_kappa: String,
    /// κ of the source tensor container (the safetensors manifest for the
    /// parametric path; the model file itself for single-file sources).
    pub tensor_manifest_kappa: String,
    /// Sorted unique κ closure of every source artifact consumed.
    pub source_kappas: Vec<String>,
    /// Sorted unique substrate witness identifiers.
    pub substrate_witnesses: Vec<String>,
}

/// Stable manifest tensor id for a graph `TensorId`: fixed-width hex so
/// ASCII order equals numeric order and the table is sorted by
/// construction.
fn tensor_id(id: u32) -> String {
    format!("t{id:08x}")
}

fn dtype_name(info: &hologram_ai_common::TensorInfo) -> String {
    format!("{:?}", info.logical_dtype).to_ascii_lowercase()
}

fn concrete_shape(info: &hologram_ai_common::TensorInfo) -> Vec<u64> {
    // Shape is a SmallVec of DimExpr; a still-symbolic dim records as 0
    // (the manifest carries structure, not a shape oracle).
    info.shape
        .iter()
        .map(|d| d.as_concrete().unwrap_or(0))
        .collect()
}

/// Build the canonical manifest for one compiled component graph.
///
/// Tensors are every id the graph mentions (params, node inputs/outputs,
/// graph inputs/outputs). A weight binding is emitted per param: `External`
/// params reuse their substrate κ verbatim; `Inline` params are hashed
/// here; `Mmap` params read and hash their byte range.
pub fn graph_manifest(
    graph: &AiGraph,
    component_id: &str,
    role: &str,
    provenance: PrismProvenance,
) -> Result<Manifest> {
    let mut ids: BTreeSet<u32> = BTreeSet::new();
    ids.extend(graph.inputs.iter().copied());
    ids.extend(graph.outputs.iter().copied());
    for node in &graph.nodes {
        ids.extend(node.inputs.iter().copied());
        ids.extend(node.outputs.iter().copied());
    }
    ids.extend(graph.params.keys().copied());

    let tensors: Vec<Tensor> = ids
        .iter()
        .map(|&id| {
            let (dtype, shape) = graph
                .tensor_info
                .get(&id)
                .map_or(("unknown".to_owned(), Vec::new()), |info| {
                    (dtype_name(info), concrete_shape(info))
                });
            Tensor {
                dtype,
                id: tensor_id(id),
                shape,
            }
        })
        .collect();

    let mut weights = Vec::new();
    for (&id, param) in &graph.params {
        let kappa = match param {
            AiParam::External { kappa, .. } => kappa.clone(),
            AiParam::Inline { data, .. } => crate::materialize::kappa_of(data),
            AiParam::Mmap {
                path, offset, len, ..
            } => {
                let bytes = read_range(path, *offset, *len)
                    .with_context(|| format!("hashing mmap weight {path:?}"))?;
                crate::materialize::kappa_of(&bytes)
            }
        };
        weights.push(Weight {
            kappa,
            tensor: tensor_id(id),
        });
    }
    weights.sort_by(|a, b| a.tensor.cmp(&b.tensor));

    let nodes: Vec<Node> = graph
        .nodes
        .iter()
        .map(|node| Node {
            inputs: node.inputs.iter().map(|&i| tensor_id(i)).collect(),
            op: format!("{:?}", node.op)
                .split(['(', ' ', '{'])
                .next()
                .unwrap_or("op")
                .to_ascii_lowercase(),
            outputs: node.outputs.iter().map(|&o| tensor_id(o)).collect(),
        })
        .collect();

    let manifest = Manifest {
        components: vec![Component {
            graph: PrismGraph {
                inputs: graph.inputs.iter().map(|&i| tensor_id(i)).collect(),
                nodes,
                outputs: graph.outputs.iter().map(|&o| tensor_id(o)).collect(),
            },
            id: component_id.to_owned(),
            role: role.to_owned(),
            weight_group: component_id.to_owned(),
            weight_source: Vec::new(),
        }],
        connections: Vec::new(),
        provenance: Provenance {
            config_kappa: provenance.config_kappa,
            emitter_semantics_id: emitter_semantics_id(),
            source_kappas: provenance.source_kappas,
            substrate_witnesses: provenance.substrate_witnesses,
            tensor_manifest_kappa: provenance.tensor_manifest_kappa,
        },
        tensors,
        weights,
    };
    // Self-check before anything is baked: the emitted manifest must admit.
    admit_bytes(&prism_archive::canonical::canonical_bytes(&manifest))?;
    Ok(manifest)
}

/// Canonical bytes + κ of a manifest (`blake3:<hex>` of the exact bytes).
#[must_use]
pub fn manifest_bytes_and_kappa(manifest: &Manifest) -> (Vec<u8>, String) {
    let bytes = prism_archive::canonical::canonical_bytes(manifest);
    let kappa = prism_archive::manifest_kappa(manifest);
    (bytes, kappa)
}

/// Admit manifest bytes: strict decode → normalize → validator conjunction.
/// Every failure is a registered `HP` diagnostic from the contract layer.
pub fn admit_bytes(bytes: &[u8]) -> Result<Manifest> {
    let manifest = prism_archive::strict_decode(bytes)
        .map_err(|e| anyhow!("prism manifest strict decode refused: {e}"))?;
    let normalized = prism_archive::normalize(&manifest)
        .map_err(|e| anyhow!("prism manifest normalization refused: {e}"))?;
    prism_archive::admit(&normalized)
        .map_err(|e| anyhow!("prism admission refused the archive: {e}"))?;
    Ok(manifest)
}

/// The emitter-semantics digest recorded in provenance: this crate is the
/// emitter, identified by its exact pinned prism-archive dependency rev
/// plus this module's role. A content digest over the compiled-in
/// identity, stable per build of this crate.
fn emitter_semantics_id() -> String {
    let identity = concat!(
        "hologram-ai/prism-emitter/1\n",
        "prism-archive rev 18d602ade7891dc79a51323caf33551138006e01\n",
    );
    crate::materialize::kappa_of(identity.as_bytes())
        .strip_prefix("blake3:")
        .expect("kappa_of mints blake3 labels")
        .to_owned()
}

fn read_range(path: &std::path::Path, offset: u64, len: u64) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0_u8; usize::try_from(len)?];
    file.read_exact(&mut buf)?;
    Ok(buf)
}
