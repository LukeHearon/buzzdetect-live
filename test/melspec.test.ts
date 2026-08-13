/**
 * Checks the browser log-mel front end against TensorFlow's.
 *
 * Fixtures come from `tools/06_fixtures.py`, which runs the real buzzdetect
 * pipeline (soundfile -> librosa resample -> yamnet features) over 10 s of a
 * field recording. Nothing here is compared to a second reimplementation; the
 * expected values are TensorFlow's own.
 *
 * The tolerances are deliberately tight. A log-mel bin is log(mel + 0.001), so
 * an absolute error of 1e-4 is a fraction of a percent of a mel bin's energy,
 * and the model is far less sensitive to that than to a structural mistake --
 * an off-by-one in framing, a symmetric instead of periodic window, a mel band
 * edge computed in the wrong domain. Those all fail by orders of magnitude, so
 * a tight bound is what makes this test worth running.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  MelPatchExtractor,
  buildMelBank,
  paddedLength,
  frameCount,
  MEL_BANDS,
  SPECTROGRAM_BINS,
  PATCH_FRAMES,
  PATCH_VALUES,
  FFT_SIZE,
  WINDOW_SAMPLES,
  HOP_SAMPLES,
} from '../src/dsp/melspec';
import { RealFft, hannPeriodic } from '../src/dsp/fft';

const FIX = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');

function f32(name: string): Float32Array {
  const buf = readFileSync(join(FIX, name));
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

const meta = JSON.parse(readFileSync(join(FIX, 'meta.json'), 'utf8'));

interface Diff {
  max: number;
  mean: number;
  argmax: number;
}

function diff(got: ArrayLike<number>, want: ArrayLike<number>): Diff {
  let max = 0;
  let sum = 0;
  let argmax = 0;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > max) {
      max = d;
      argmax = i;
    }
    sum += d;
  }
  return { max, mean: sum / want.length, argmax };
}

describe('derived YAMNet constants', () => {
  it('match params.py', () => {
    expect(WINDOW_SAMPLES).toBe(400);
    expect(HOP_SAMPLES).toBe(160);
    expect(FFT_SIZE).toBe(512);
    expect(SPECTROGRAM_BINS).toBe(257);
    expect(PATCH_FRAMES).toBe(96);
  });
});

describe('RealFft', () => {
  it('agrees with a direct DFT on a real signal', () => {
    const n = 512;
    const fft = new RealFft(n);
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = Math.sin((2 * Math.PI * 7 * i) / n) + 0.3 * Math.cos((2 * Math.PI * 61 * i) / n);
    }
    const got = new Float32Array(n / 2 + 1);
    fft.magnitudes(x, got);

    const want = new Float64Array(n / 2 + 1);
    for (let k = 0; k <= n / 2; k++) {
      let re = 0;
      let im = 0;
      for (let i = 0; i < n; i++) {
        const a = (-2 * Math.PI * k * i) / n;
        re += x[i] * Math.cos(a);
        im += x[i] * Math.sin(a);
      }
      want[k] = Math.hypot(re, im);
    }
    expect(diff(got, want).max).toBeLessThan(1e-3);
  });

  it('resolves a pure tone into a single bin', () => {
    const n = 512;
    const fft = new RealFft(n);
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = Math.cos((2 * Math.PI * 32 * i) / n);
    const out = new Float32Array(n / 2 + 1);
    fft.magnitudes(x, out);

    let peak = 0;
    for (let k = 1; k <= n / 2; k++) if (out[k] > out[peak]) peak = k;
    expect(peak).toBe(32);
  });
});

describe('Hann window', () => {
  it('is periodic, not symmetric', () => {
    const w = hannPeriodic(400);
    expect(w[0]).toBeCloseTo(0, 12);
    // A periodic window is not zero at its last sample; a symmetric one is.
    expect(w[399]).toBeGreaterThan(0);
    expect(w[200]).toBeCloseTo(1, 12);
  });
});

describe('mel filterbank', () => {
  it('reconstructs tf.signal.linear_to_mel_weight_matrix', () => {
    const want = f32('mel_matrix.f32'); // [257, 64], row-major
    expect(want.length).toBe(SPECTROGRAM_BINS * MEL_BANDS);

    const bank = buildMelBank();
    const dense = new Float32Array(SPECTROGRAM_BINS * MEL_BANDS);
    for (let j = 0; j < MEL_BANDS; j++) {
      for (let k = 0; k < bank.lengths[j]; k++) {
        dense[(bank.starts[j] + k) * MEL_BANDS + j] = bank.weights[bank.offsets[j] + k];
      }
    }

    // Not exact: TensorFlow evaluates the mel scale and the band slopes in
    // float32, while buildMelBank does that arithmetic in float64 and rounds
    // once at the end. The difference is TF's accumulated rounding, not a
    // disagreement about the filterbank, and it is ~2 orders of magnitude below
    // what survives into a log-mel bin (see the patch test below).
    const d = diff(dense, want);
    expect(d.max).toBeLessThan(1e-5);
  });

  it('zeroes the DC bin, as TensorFlow does', () => {
    const bank = buildMelBank();
    for (let j = 0; j < MEL_BANDS; j++) expect(bank.starts[j]).toBeGreaterThan(0);
  });

  it('is sparse enough to be worth the banded layout', () => {
    const bank = buildMelBank();
    const nonzeros = bank.weights.length;
    expect(nonzeros).toBeLessThan(0.1 * SPECTROGRAM_BINS * MEL_BANDS);
  });
});

describe('waveform padding', () => {
  it('reproduces features.pad_waveform for the fixture', () => {
    // 10 s at the 0.96 s hop: the fixture's patch count is TensorFlow's.
    const padded = paddedLength(meta.numSamples, 0.96 * meta.framehopProp);
    const frames = frameCount(padded);
    const patches = Math.floor((frames - PATCH_FRAMES) / (PATCH_FRAMES * meta.framehopProp)) + 1;
    expect(patches).toBe(meta.numPatches);
  });

  // Padded length / frame count / patch count printed by pad_waveform and
  // waveform_to_log_mel_spectrogram_patches at framehop_prop = 1. The short
  // clips are the interesting ones: a file under one patch long still has to
  // produce exactly one patch, and getting the padding target wrong there is
  // invisible on long files, where the shortfall term is zero.
  const TF_PADDING: Array<[number, number, number, number]> = [
    // samples in, padded samples, STFT frames, patches
    [0, 15600, 96, 1],
    [1, 15600, 96, 1],
    [1000, 15600, 96, 1],
    [15600, 15600, 96, 1],
    [160000, 169200, 1056, 11],
    [4799652, 4807920, 30048, 313],
  ];

  it.each(TF_PADDING)(
    '%i samples -> %i padded, %i frames, %i patches',
    (samples, padded, frames, patches) => {
      expect(paddedLength(samples, 0.96)).toBe(padded);
      expect(frameCount(padded)).toBe(frames);
      expect(new MelPatchExtractor(1).patchCount(samples)).toBe(patches);
    },
  );
});

describe('log-mel patches', () => {
  const wav = f32('waveform.f32');
  const want = f32('patches.f32');

  it('match TensorFlow over the whole fixture', () => {
    const ex = new MelPatchExtractor(meta.framehopProp);
    expect(ex.patchCount(wav.length)).toBe(meta.numPatches);

    const got = new Float32Array(meta.numPatches * PATCH_VALUES);
    ex.fillPatches(wav, 0, meta.numPatches, got);

    const d = diff(got, want);
    const patch = Math.floor(d.argmax / PATCH_VALUES);
    console.log(
      `log-mel: max|diff| ${d.max.toExponential(3)} (patch ${patch}), ` +
        `mean|diff| ${d.mean.toExponential(3)} over ${want.length} values`,
    );
    expect(d.max).toBeLessThan(1e-3);
    expect(d.mean).toBeLessThan(1e-5);
  });

  it('gives the same values whatever range they are requested in', () => {
    // The streaming path asks for slices; a slice must not depend on what came
    // before it, or long-file results would drift from short-file ones.
    const ex = new MelPatchExtractor(meta.framehopProp);
    const all = new Float32Array(meta.numPatches * PATCH_VALUES);
    ex.fillPatches(wav, 0, meta.numPatches, all);

    const one = new Float32Array(PATCH_VALUES);
    for (const p of [0, 1, 5, meta.numPatches - 1]) {
      ex.fillPatches(wav, p, 1, one);
      const d = diff(one, all.subarray(p * PATCH_VALUES, (p + 1) * PATCH_VALUES));
      expect(d.max).toBe(0);
    }
  });

  it('reads past the end of the waveform as silence', () => {
    // The final patch of the fixture extends beyond the 10 s of audio, so this
    // is exercised by the parity check above; assert the mechanism directly too.
    const ex = new MelPatchExtractor(1);
    const short = new Float32Array(1000);
    const out = new Float32Array(PATCH_VALUES);
    ex.fillPatches(short, 0, 1, out);
    expect(Number.isFinite(out[PATCH_VALUES - 1])).toBe(true);
    // log(0 + 0.001) for a fully silent frame
    expect(out[PATCH_VALUES - 1]).toBeCloseTo(Math.log(0.001), 5);
  });
});
