//! Native half of the cross-substrate model twin (M2).
//!
//! Mirrors `DecodeChatSession` (hologram-ai-wasm) step for step — the SAME
//! tokenizer construction (`from_tokenizer_json_bytes`, no config sibling),
//! the same manifest → κ-store → int8-tier → growable → decode-session →
//! seeder → `generate_stream_decode` flow, the same GenConfig — over a model
//! directory on disk. Prints the generated text and its FNV-64 digest; the
//! Node harness (`twin_harness.mjs` beside the wasm crate) runs the wasm
//! build over the same inputs and must print the same digest.
//!
//! Run: HOLO_TWIN_DIR=<dir with config.json/tokenizer.json/model.safetensors>
//!      cargo test --release --test native_wasm_twin -- --ignored --nocapture

use std::num::NonZeroU64;

use hologram_ai::commands::generate::{generate_stream_decode, GenConfig};
use hologram_ai::decode::DecodeSession;
use hologram_ai::engine::{decode_bucket_for_turn, geometric_window};
use hologram_ai::materialize::DirKappaStore;
use hologram_ai::staged::GrowableStagedSession;
use hologram_ai::SessionProvider;
use hologram_ai_common::lower::{quant_key, QuantMap, QuantTier};
use hologram_ai_common::DType;
use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};

const PROMPT: &str = "The sky is blue because";
const MAX_TOKENS: usize = 32;
const LAYERS_PER_STAGE: u64 = 8;

fn fnv64(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[test]
#[ignore]
fn native_twin_digest() {
    let dir = std::path::PathBuf::from(
        std::env::var("HOLO_TWIN_DIR").expect("set HOLO_TWIN_DIR to the model directory"),
    );
    let config_json = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let config: serde_json::Value = serde_json::from_str(&config_json).expect("config parse");
    let tokenizer_bytes = std::fs::read(dir.join("tokenizer.json")).expect("tokenizer.json");
    // EXACTLY the wasm construction: bytes only, no tokenizer_config sibling.
    let tokenizer =
        NativeTokenizer::from_tokenizer_json_bytes(&tokenizer_bytes).expect("tokenizer");

    // Safetensors → per-tensor manifest + κ-store (same insertion the wasm
    // harness performs through its callbacks).
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("safetensors");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_len;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..data_start]).expect("safetensors header");

    let store_dir = std::env::temp_dir().join(format!("holo-twin-{}", std::process::id()));
    std::fs::create_dir_all(&store_dir).expect("store dir");
    let store = DirKappaStore::new(&store_dir);

    let mut keys = Vec::new();
    let mut kappas = Vec::new();
    let mut shapes: Vec<Vec<u64>> = Vec::new();
    let mut dtypes = Vec::new();
    let mut ranges = Vec::new();
    // BTreeMap iteration = lexicographic key order — the same order a JS
    // harness gets from sorting Object.keys(header).
    for (name, entry) in &header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = match entry["dtype"].as_str().unwrap() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            other => panic!("unsupported dtype {other}"),
        };
        let shape: Vec<u64> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let offs = entry["data_offsets"].as_array().unwrap();
        let (s, e) = (
            data_start + offs[0].as_u64().unwrap() as usize,
            data_start + offs[1].as_u64().unwrap() as usize,
        );
        let kappa = store.insert(&bytes[s..e]).expect("insert");
        keys.push(name.clone());
        kappas.push(kappa);
        shapes.push(shape);
        dtypes.push(dtype);
        ranges.push((s, e));
    }

    // int8 tier — identical derivation to the wasm harness (same fns).
    let lps = NonZeroU64::new(LAYERS_PER_STAGE).unwrap();
    let mut quant = QuantMap::new();
    let index: std::collections::HashMap<&str, usize> = kappas
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    for wide in
        hologram_ai::staged::quantizable_weights(&config_json, &keys, &kappas, &shapes, &dtypes, None, lps)
            .expect("quantizable")
    {
        let i = index[wide.as_str()];
        let (s, e) = ranges[i];
        let artifact = hologram_ai::quantized::derive_quantized_artifact_tier(
            &bytes[s..e],
            dtypes[i],
            QuantTier::Int8,
            shapes[i][0],
            shapes[i][1],
        )
        .expect("derive");
        let ak = store.insert(&artifact).expect("insert artifact");
        quant.insert(quant_key(&wide, None), (ak, shapes[i][0], shapes[i][1], QuantTier::Int8));
    }
    for chunk in
        hologram_ai::staged::head_quant_chunks(&config_json, &keys, &kappas, &shapes, &dtypes, None, lps)
            .expect("head chunks")
    {
        let i = index[chunk.kappa.as_str()];
        let (s, _e) = ranges[i];
        let (cs, ce) = (s + chunk.offset as usize, s + (chunk.offset + chunk.len) as usize);
        let artifact = hologram_ai::quantized::derive_quantized_artifact_tier(
            &bytes[cs..ce],
            dtypes[i],
            QuantTier::Int8,
            chunk.out_features,
            chunk.in_features,
        )
        .expect("derive chunk");
        let ak = store.insert(&artifact).expect("insert chunk artifact");
        quant.insert(
            quant_key(&chunk.kappa, Some((chunk.offset, chunk.len))),
            (ak, chunk.out_features, chunk.in_features, QuantTier::Int8),
        );
    }

    let mut growable = GrowableStagedSession::new(
        config_json.clone(),
        keys,
        kappas,
        shapes,
        dtypes,
        None,
        lps,
        Box::new(store),
    )
    .expect("growable");
    growable.set_quant_map(quant);

    let context_len = SessionProvider::max_window(&growable) as u64;
    let rope = hologram_ai_safetensors::parametric::rope_spec_from_config(&config).expect("rope");

    // Mirror DecodeChatSession::generate (no draft, plain decode).
    let prompt_len = tokenizer
        .encode(PROMPT)
        .len()
        .max(1)
        .min(context_len as usize);
    let want = decode_bucket_for_turn(prompt_len, MAX_TOKENS, context_len as usize);
    let runner = growable.decode_runner_for(want).expect("runner");
    let mut session = DecodeSession::new(runner, rope, context_len).expect("session");
    let bucket = session.geometry().bucket;
    let chunk = (geometric_window(1, context_len as usize) as u64).min(bucket as u64);
    if chunk >= 2 {
        let seeder = growable.chunk_runner_for(bucket, chunk).expect("seeder");
        session.set_seeder(seeder).expect("set seeder");
    }

    let cfg = GenConfig {
        max_tokens: Some(MAX_TOKENS),
        temperature: 0.0,
        top_k: None,
        stop: Vec::new(),
        eos: None,
        seed: 0x9E37_79B9_7F4A_7C15,
    };
    let mut sink = Vec::new();
    let text = generate_stream_decode(&mut session, &tokenizer, PROMPT, &cfg, &mut sink)
        .expect("generate");

    println!("NATIVE TWIN text: {text:?}");
    println!("NATIVE TWIN digest: {}", fnv64(text.as_bytes()));
    let _ = std::fs::remove_dir_all(&store_dir);
}
