/**
 * The display filterbank, checked for the failure it is prone to.
 *
 * Below ~1 kHz a mel band is narrower than the STFT's bin spacing, so a naive
 * triangle picks up one bin, part of one bin, or nothing, purely according to
 * where its centre falls between bins. That draws as horizontal stripes: an
 * artefact of the filterbank that reads as structure in the audio.
 *
 * Flat-spectrum input is the test that catches it. Every band should come back
 * at roughly the same level, so a large step between neighbouring bands is the
 * bank talking rather than the signal. Before the widening in
 * buildDisplayMelBank this reached 51 bytes (~24 dB) around 560 Hz.
 */

import { describe, it, expect } from 'vitest';

import {
  buildDisplayMelBank,
  computeSpectrogram,
  bandForHz,
  DISPLAY_FFT_SIZE,
  DISPLAY_MEL_BANDS,
} from '../src/dsp/spectrogram';

const SAMPLE_RATE = 48000;

/** Deterministic white noise -- flat in expectation, no dependency on Math.random. */
function noise(n: number): Float32Array {
  const wav = new Float32Array(n);
  let s = 12345;
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    wav[i] = (s / 0x3fffffff - 1) * 0.2;
  }
  return wav;
}

function bandMeans(): Float64Array {
  const bank = buildDisplayMelBank(DISPLAY_FFT_SIZE, SAMPLE_RATE);
  const spec = computeSpectrogram(noise(SAMPLE_RATE * 2), SAMPLE_RATE, DISPLAY_FFT_SIZE, 480, bank);
  const mean = new Float64Array(bank.bands);
  for (let c = 0; c < spec.columns; c++) {
    for (let j = 0; j < bank.bands; j++) mean[j] += spec.bins[c * bank.bands + j];
  }
  for (let j = 0; j < bank.bands; j++) mean[j] /= spec.columns;
  return mean;
}

describe('display mel bank', () => {
  it('responds evenly to a flat spectrum', () => {
    const mean = bandMeans();

    // A byte is (DB_CEIL - DB_FLOOR) / 255 = 0.47 dB, so 12 bytes is ~5.6 dB:
    // well above the ~3 dB spread noise itself produces in the wide bands at
    // the top, and far below the striping this is here to catch.
    let worst = 0;
    let at = 0;
    for (let j = 1; j < mean.length; j++) {
      const step = Math.abs(mean[j] - mean[j - 1]);
      if (step > worst) {
        worst = step;
        at = j;
      }
    }
    expect(worst, `largest step at band ${at}`).toBeLessThan(12);
  });

  it('gives every band signal, including below the bin spacing', () => {
    const mean = bandMeans();
    // 48 kHz / 1024 puts the first FFT bin at 47 Hz; the bands below it must
    // still read as the noise floor rather than as zero.
    for (let j = 0; j < mean.length; j++) expect(mean[j]).toBeGreaterThan(100);
  });

  it('maps a frequency back to the band that holds it', () => {
    const bank = buildDisplayMelBank(DISPLAY_FFT_SIZE, SAMPLE_RATE);
    expect(bank.bands).toBe(DISPLAY_MEL_BANDS);
    expect(bandForHz(0, bank)).toBeCloseTo(-1, 6);
    expect(bandForHz(SAMPLE_RATE / 2, bank)).toBeCloseTo(bank.bands, 6);
  });
});
