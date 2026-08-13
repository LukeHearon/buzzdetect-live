/**
 * End-to-end parity: the browser's own log-mel front end, fed to the browser's
 * own ONNX model, against the activations buzzdetect produced for the same
 * audio through TensorFlow.
 *
 * `melspec.test.ts` checks the features and `tools/04_verify_onnx.py` checks the
 * model; this checks the two composed, which is what a user actually runs. It
 * uses onnxruntime-web's wasm backend -- the same build the browser loads -- so
 * the numbers here are the numbers the app produces.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { InferenceSession } from 'onnxruntime-web';

import { MelPatchExtractor, PATCH_VALUES, PATCH_FRAMES, MEL_BANDS } from '../src/dsp/melspec';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const FIX = join(ROOT, 'test', 'fixtures');
const MODEL = join(ROOT, 'public', 'model', 'buzzdetect_v3.onnx');

function f32(name: string): Float32Array {
  const buf = readFileSync(join(FIX, name));
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

const meta = JSON.parse(readFileSync(join(FIX, 'meta.json'), 'utf8'));

let ort: typeof import('onnxruntime-web');
let session: InferenceSession;

beforeAll(async () => {
  ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = join(ROOT, 'node_modules', 'onnxruntime-web', 'dist') + '/';
  ort.env.logLevel = 'error';
  session = await ort.InferenceSession.create(readFileSync(MODEL), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}, 60_000);

describe('buzzdetect_v3.onnx end to end', () => {
  it('reproduces buzzdetect activations from raw 16 kHz samples', async () => {
    const wav = f32('waveform.f32');
    const want = f32('activations.f32');
    const n = meta.numPatches;
    const classes: string[] = meta.classes;

    const extractor = new MelPatchExtractor(meta.framehopProp);
    const patches = new Float32Array(n * PATCH_VALUES);
    extractor.fillPatches(wav, 0, n, patches);

    const out = await session.run({
      patches: new ort.Tensor('float32', patches, [n, PATCH_FRAMES, MEL_BANDS]),
    });
    const got = out.activations.data as Float32Array;
    expect(got.length).toBe(want.length);

    let max = 0;
    let sum = 0;
    for (let i = 0; i < want.length; i++) {
      const d = Math.abs(got[i] - want[i]);
      if (d > max) max = d;
      sum += d;
    }
    const buzz = classes.indexOf('ins_buzz');
    let maxBuzz = 0;
    for (let p = 0; p < n; p++) {
      maxBuzz = Math.max(maxBuzz, Math.abs(got[p * classes.length + buzz] - want[p * classes.length + buzz]));
    }
    console.log(
      `end to end over ${n} frames: max|diff| ${max.toExponential(3)}, ` +
        `mean|diff| ${(sum / want.length).toExponential(3)}, ins_buzz max ${maxBuzz.toExponential(3)}`,
    );

    // Well inside the 0.01 quantum of buzzdetect's 2-decimal CSV output.
    expect(max).toBeLessThan(5e-3);
  }, 60_000);

  it('agrees with itself whatever batch size the patches arrive in', async () => {
    // The analyser splits a file into batches to bound peak memory; results must
    // not depend on where those splits land.
    const wav = f32('waveform.f32');
    const n = meta.numPatches;
    const nClasses = meta.classes.length;
    const extractor = new MelPatchExtractor(meta.framehopProp);

    const all = new Float32Array(n * PATCH_VALUES);
    extractor.fillPatches(wav, 0, n, all);
    const whole = (
      await session.run({ patches: new ort.Tensor('float32', all, [n, PATCH_FRAMES, MEL_BANDS]) })
    ).activations.data as Float32Array;

    const batched = new Float32Array(n * nClasses);
    const BATCH = 4;
    for (let start = 0; start < n; start += BATCH) {
      const count = Math.min(BATCH, n - start);
      const buf = new Float32Array(count * PATCH_VALUES);
      extractor.fillPatches(wav, start, count, buf);
      const r = (
        await session.run({
          patches: new ort.Tensor('float32', buf, [count, PATCH_FRAMES, MEL_BANDS]),
        })
      ).activations.data as Float32Array;
      batched.set(r, start * nClasses);
    }

    for (let i = 0; i < whole.length; i++) {
      expect(batched[i]).toBeCloseTo(whole[i], 6);
    }
  }, 60_000);
});
