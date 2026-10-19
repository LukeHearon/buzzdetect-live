/**
 * End-to-end parity: the app's framing and batching, feeding buzzdetect's own
 * ONNX graphs, against the activations buzzdetect's engine produced for the
 * same audio (public/sample/testbuzz.reference.json, from tools/models.py).
 *
 * Every model is checked. It uses onnxruntime-web's wasm backend -- the same
 * build the browser loads -- so the numbers here are the numbers the app
 * produces. The shipped sample is 32-bit float 16 kHz mono, the exact array
 * buzzdetect was given, so nothing but the runtime and the framing is measured.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { InferenceSession } from 'onnxruntime-web';

import { batchSamples, fillBatch, frameCount, type ModelConfig } from '../src/model/framing';
import { MODEL_NAMES } from '../src/model/models';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const PUBLIC = join(ROOT, 'public');

interface ModelReference {
  classes: string[];
  activations: number[][];
  frameCounts: Record<string, number>;
}

const reference: { samples: number; models: Record<string, ModelReference> } = JSON.parse(
  readFileSync(join(PUBLIC, 'sample', 'testbuzz.reference.json'), 'utf8'),
);

/** The samples of a 32-bit float WAV, which is all the shipped sample is. */
function readFloatWav(path: string): Float32Array {
  const buf = readFileSync(path);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let format = 0;
  for (let at = 12; at + 8 <= buf.byteLength; ) {
    const id = buf.toString('ascii', at, at + 4);
    const size = view.getUint32(at + 4, true);
    if (id === 'fmt ') format = view.getUint16(at + 8, true);
    if (id === 'data') {
      if (format !== 3) throw new Error(`${path}: WAV format ${format}, not 32-bit float`);
      return new Float32Array(buf.buffer.slice(buf.byteOffset + at + 8, buf.byteOffset + at + 8 + size));
    }
    at += 8 + size + (size & 1);
  }
  throw new Error(`${path}: no data chunk`);
}

const wav = readFloatWav(join(PUBLIC, 'sample', 'testbuzz.wav'));

let ort: typeof import('onnxruntime-web');

beforeAll(async () => {
  ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = join(ROOT, 'node_modules', 'onnxruntime-web', 'dist') + '/';
  ort.env.logLevel = 'error';
});

/** Runs every frame of `wav` in batches of `capacity`, as the worker does. */
async function runAll(
  session: InferenceSession,
  cfg: ModelConfig,
  samples: Float32Array,
  capacity: number,
): Promise<Float32Array> {
  const n = frameCount(cfg, samples.length);
  const nClasses = cfg.classes.length;
  const out = new Float32Array(n * nClasses);
  const scratch = new Float32Array(batchSamples(cfg, capacity));
  for (let first = 0; first < n; first += capacity) {
    const count = Math.min(capacity, n - first);
    fillBatch(cfg, samples, first, scratch);
    const result = await session.run({
      waveform: new ort.Tensor('float32', scratch, [scratch.length]),
    });
    const data = result.predictions.data as Float32Array;
    out.set(data.subarray(0, count * nClasses), first * nClasses);
  }
  return out;
}

function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

describe.each(MODEL_NAMES)('%s', (name) => {
  const dir = join(PUBLIC, 'model', name);
  const cfg: ModelConfig = JSON.parse(readFileSync(join(dir, 'config_model.json'), 'utf8'));
  const ref = reference.models[name];
  let session: InferenceSession;

  beforeAll(async () => {
    session = await ort.InferenceSession.create(readFileSync(join(dir, 'model.onnx')), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
  }, 60_000);

  it('counts frames exactly as buzzdetect does, float32 quirks included', () => {
    for (const [n, frames] of Object.entries(ref.frameCounts)) {
      expect(frameCount(cfg, Number(n)), `${n} samples`).toBe(frames);
    }
  });

  it('reproduces buzzdetect activations from raw 16 kHz samples', async () => {
    expect(cfg.classes).toEqual(ref.classes);
    const want = ref.activations.flat();
    const got = await runAll(session, cfg, wav, 32);
    expect(got.length).toBe(want.length);

    const max = maxDiff(got, want);
    console.log(`${name} over ${ref.activations.length} frames: max|diff| ${max.toExponential(3)}`);
    // Well inside the 0.01 quantum of buzzdetect's 2-decimal CSV output; in
    // practice float32 noise, since these are the same graphs.
    expect(max).toBeLessThan(5e-3);
  }, 60_000);

  it('agrees with itself wherever the batch boundaries fall', async () => {
    // The analyser splits a file into batches to bound memory, and a frame's
    // value can depend on audio past its own end. Batches of two put a seam
    // every other frame; without the spare look-ahead frame heavy_v1's last
    // frame in each batch moves by up to 0.5.
    const whole = await runAll(session, cfg, wav, 32);
    const split = await runAll(session, cfg, wav, 2);
    expect(maxDiff(split, whole)).toBeLessThan(1e-4);
  }, 60_000);
});
