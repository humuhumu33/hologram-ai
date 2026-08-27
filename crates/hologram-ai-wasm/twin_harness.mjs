// wasm half of the cross-substrate model twin (M2).
// Mirrors tests/native_wasm_twin.rs over the same model directory; the
// printed digest must equal the native run's.
//
// Run: node twin_harness.mjs <model-dir>

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const wasm = require('./pkg/hologram_ai_wasm.js');

const dir = process.argv[2];
if (!dir) { console.error('usage: node twin_harness.mjs <model-dir>'); process.exit(2); }

const PROMPT = 'The sky is blue because';
const MAX_TOKENS = 32;
const LAYERS_PER_STAGE = 8;

const configJson = readFileSync(join(dir, 'config.json'), 'utf8');
const tokenizerBytes = readFileSync(join(dir, 'tokenizer.json'));
const st = readFileSync(join(dir, 'model.safetensors'));

// safetensors header
const headerLen = Number(st.readBigUInt64LE(0));
const dataStart = 8 + headerLen;
const header = JSON.parse(st.subarray(8, dataStart).toString('utf8'));

// Sorted key order — matches the native twin's BTreeMap iteration.
const names = Object.keys(header).filter((n) => n !== '__metadata__').sort();

const keys = [], kappas = [], shapes = [], shapesNum = [], dtypes = [];
const store = new Map();           // kappa -> Uint8Array
const ranges = new Map();          // kappa -> [start, end] into st

for (const name of names) {
  const e = header[name];
  const s = dataStart + e.data_offsets[0];
  const end = dataStart + e.data_offsets[1];
  const bytes = st.subarray(s, end);
  const kappa = wasm.compute_kappa(bytes);
  store.set(kappa, new Uint8Array(bytes));
  ranges.set(kappa, [s, end]);
  keys.push(name);
  kappas.push(kappa);
  shapes.push(JSON.stringify(e.shape.map(Number)));
  shapesNum.push(e.shape.map(Number));
  dtypes.push(e.dtype);
}

// int8 tier — same derivation functions as native.
const wide = wasm.quantizable_weights(configJson, keys, kappas, shapes, dtypes, undefined, LAYERS_PER_STAGE);
const quantEntries = [];
const kappaIndex = new Map(kappas.map((k, i) => [k, i]));
for (const wk of wide) {
  const i = kappaIndex.get(wk);
  const [s, e] = ranges.get(wk);
  const artifact = wasm.derive_quantized_artifact(st.subarray(s, e), dtypes[i], shapesNum[i][0], shapesNum[i][1], 'int8');
  const ak = wasm.compute_kappa(artifact);
  store.set(ak, artifact);
  quantEntries.push({ wide: wk, artifact: ak, out: shapesNum[i][0], in: shapesNum[i][1], tier: 'int8' });
}
const chunks = JSON.parse(wasm.head_quant_chunks(configJson, keys, kappas, shapes, dtypes, undefined, LAYERS_PER_STAGE));
for (const c of chunks) {
  const [s] = ranges.get(c.kappa);
  const i = kappaIndex.get(c.kappa);
  const slice = st.subarray(s + c.offset, s + c.offset + c.len);
  const artifact = wasm.derive_quantized_artifact(slice, dtypes[i], c.out, c.in, 'int8');
  const ak = wasm.compute_kappa(artifact);
  store.set(ak, artifact);
  quantEntries.push({ wide: c.kappa, artifact: ak, out: c.out, in: c.in, offset: c.offset, len: c.len, tier: 'int8' });
}

const resolveKappa = (kappa) => {
  const bytes = store.get(kappa);
  if (!bytes) throw new Error(`κ not in store: ${kappa}`);
  return bytes;
};

const session = new wasm.DecodeChatSession(
  configJson,
  keys,
  kappas,
  shapes,
  dtypes,
  undefined,             // context_length -> model's own
  LAYERS_PER_STAGE,
  resolveKappa,
  undefined,             // invalidate
  undefined,             // resolve range
  JSON.stringify(quantEntries),
  undefined, undefined, undefined,   // derived load/store/evaporate
  undefined,             // weight budget
  undefined,             // size_kappa
  tokenizerBytes,
  undefined,             // on_progress
);

const text = session.generate(PROMPT, { max_tokens: MAX_TOKENS, temperature: 0 }, undefined);

function fnv64(buf) {
  let h = 0xcbf29ce484222325n;
  for (const b of buf) {
    h = ((h ^ BigInt(b)) * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return h.toString(16).padStart(16, '0');
}

console.log('WASM TWIN text:', JSON.stringify(text));
console.log('WASM TWIN digest:', fnv64(Buffer.from(text, 'utf8')));
