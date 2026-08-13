/**
 * The display spectrogram: an STFT projected onto mel bands and quantised to
 * one byte per band.
 *
 * Mel rather than linear because that is what the model sees. A linear axis
 * spends most of its height on the 10-20 kHz region where a buzz has nothing,
 * and squeezes the 100-1000 Hz band -- where the whole signal lives -- into a
 * few pixels. Mel spacing gives that band the room it deserves, and makes the
 * picture correspond to the features the activations below it were computed
 * from.
 *
 * The band edges span the full 0..Nyquist range regardless of what is being
 * displayed, so changing the visible frequency window is a row-lookup change
 * and never a recompute.
 *
 * Storage is one byte per band per column, and there are far fewer bands than
 * FFT bins (192 against 513), so the buffer is a quarter the size a linear
 * spectrogram would need. The whole file is analysed once into it, which is
 * what keeps panning and zooming free of FFT work.
 */

import { RealFft, hannPeriodic } from './fft';
import { hertzToMel, melToHertz } from './melspec';

/** dBFS mapped to byte 0. Anything quieter clamps to 0. */
export const DB_FLOOR = -120;
/** dBFS mapped to byte 255. */
export const DB_CEIL = 0;

/** FFT size for the display. Fixed: 1024 is the useful compromise and one less knob. */
export const DISPLAY_FFT_SIZE = 1024;
/**
 * Mel bands stored. Above a canvas-height-ish count the extra bands are never
 * resolved on screen and only cost memory.
 */
export const DISPLAY_MEL_BANDS = 192;

export interface SpectrogramData {
  /** Column-major band energies: `bins[col * binCount + band]`, band 0 lowest. */
  bins: Uint8Array;
  columns: number;
  /** Number of mel bands per column. */
  binCount: number;
  fftSize: number;
  hopSamples: number;
  sampleRate: number;
  /** Seconds per column. */
  secondsPerColumn: number;
  /**
   * Time of column 0.
   *
   * Always 0 for a file. The live scope keeps a bounded history and drops the
   * oldest half when it fills, so its column 0 walks forward in time; carrying
   * that here lets the renderer work in absolute session time either way.
   */
  startSeconds: number;
}

/** Banded mel filterbank: each band is a contiguous run of FFT bins. */
export interface MelBank {
  starts: Int32Array;
  lengths: Int32Array;
  offsets: Int32Array;
  weights: Float32Array;
  bands: number;
  /** Nyquist the bands were laid out against, for the row mapping. */
  nyquist: number;
}

/**
 * Triangular mel filterbank over the full 0..Nyquist range.
 *
 * Same construction as YAMNet's (see dsp/melspec.ts) but sized for display and
 * spanning everything the file can represent, so the visible window can be
 * changed without rebuilding.
 */
export function buildDisplayMelBank(
  fftSize: number,
  sampleRate: number,
  bands = DISPLAY_MEL_BANDS,
): MelBank {
  const binCount = fftSize / 2 + 1;
  const nyquist = sampleRate / 2;

  const binMel = new Float64Array(binCount);
  for (let i = 0; i < binCount; i++) binMel[i] = hertzToMel((i * nyquist) / (binCount - 1));

  const hiMel = hertzToMel(nyquist);
  const edges = new Float64Array(bands + 2);
  for (let i = 0; i < bands + 2; i++) edges[i] = (hiMel * i) / (bands + 1);

  const starts = new Int32Array(bands);
  const lengths = new Int32Array(bands);
  const offsets = new Int32Array(bands);
  const rows: number[][] = [];

  // Below ~1 kHz a mel band is narrower than the FFT's bin spacing, so the
  // naive triangle lands on one bin, on the tail of one bin, or between two and
  // catches nothing at all -- which draws as horizontal stripes, band by band,
  // rather than as anything in the audio. Widening every triangle to reach at
  // least one bin either side of its centre, then normalising any band whose
  // weights sum to less than one, turns those bands into a linear interpolation
  // between the two bins they sit between. The picture varies smoothly with
  // frequency again, at the cost of admitting that the display cannot resolve
  // detail the STFT never had. Bands wider than a bin are left alone.
  const binHz = nyquist / (binCount - 1);

  for (let j = 0; j < bands; j++) {
    const centerHz = melToHertz(edges[j + 1]);
    const lower = Math.min(edges[j], hertzToMel(Math.max(0, centerHz - binHz)));
    const center = edges[j + 1];
    const upper = Math.max(edges[j + 2], hertzToMel(centerHz + binHz));
    const row: number[] = [];
    let start = -1;
    let sum = 0;
    for (let i = 0; i < binCount; i++) {
      const w = Math.max(
        0,
        Math.min((binMel[i] - lower) / (center - lower), (upper - binMel[i]) / (upper - center)),
      );
      if (w > 0) {
        if (start < 0) start = i;
        row.push(w);
        sum += w;
      } else if (start >= 0) {
        break; // triangular bands are contiguous
      }
    }
    if (row.length === 0) {
      // Unreachable: a triangle spanning the centre plus a bin either side
      // always contains a bin. Kept so a future edge-case cannot draw a black
      // stripe instead of failing loudly.
      start = Math.max(0, Math.round(centerHz / binHz));
      row.push(1);
    } else if (sum < 1) {
      for (let k = 0; k < row.length; k++) row[k] /= sum;
    }
    starts[j] = start;
    lengths[j] = row.length;
    rows.push(row);
  }

  let total = 0;
  for (let j = 0; j < bands; j++) {
    offsets[j] = total;
    total += lengths[j];
  }
  const weights = new Float32Array(total);
  for (let j = 0; j < bands; j++) weights.set(rows[j], offsets[j]);

  return { starts, lengths, offsets, weights, bands, nyquist };
}

/**
 * Fractional band index for a frequency, for mapping canvas rows to bands.
 *
 * Band centres sit at `hiMel * (j + 1) / (bands + 1)`, so this inverts that.
 */
export function bandForHz(hz: number, bank: MelBank): number {
  const hiMel = hertzToMel(bank.nyquist);
  return (hertzToMel(hz) / hiMel) * (bank.bands + 1) - 1;
}

/**
 * Picks an STFT hop that keeps the spectrogram buffer under `budgetBytes`.
 *
 * Short files get ~10 ms columns; long ones get coarser time resolution rather
 * than an allocation that would swap or fail.
 */
export function chooseHop(
  numSamples: number,
  sampleRate: number,
  bands = DISPLAY_MEL_BANDS,
  budgetBytes = 48 * 1024 * 1024,
): number {
  let hop = Math.max(1, Math.round(sampleRate * 0.01));
  while ((numSamples / hop) * bands > budgetBytes) hop *= 2;
  return hop;
}

/**
 * Computes the mel spectrogram of `wav`.
 *
 * `onProgress` is called with a 0..1 fraction every few thousand columns, often
 * enough to animate and rarely enough not to cost anything.
 */
export function computeSpectrogram(
  wav: Float32Array,
  sampleRate: number,
  fftSize: number,
  hopSamples: number,
  bank: MelBank,
  onProgress?: (fraction: number) => void,
): SpectrogramData {
  const columns = wav.length < fftSize ? 0 : Math.floor((wav.length - fftSize) / hopSamples) + 1;
  const bands = bank.bands;

  const bins = new Uint8Array(columns * bands);
  const fft = new RealFft(fftSize);
  const window = hannPeriodic(fftSize);
  const frame = new Float32Array(fftSize);
  const mag = new Float32Array(fftSize / 2 + 1);

  // Hann's coherent gain is 0.5, so a full-scale sine peaks at fftSize/4 in the
  // magnitude spectrum; scaling by 4/fftSize puts 0 dBFS at 0 dB.
  const norm = 4 / fftSize;
  const scale = 255 / (DB_CEIL - DB_FLOOR);
  const REPORT_EVERY = 4096;

  const { starts, lengths, offsets, weights } = bank;

  for (let c = 0; c < columns; c++) {
    const start = c * hopSamples;
    for (let n = 0; n < fftSize; n++) frame[n] = wav[start + n] * window[n];
    fft.magnitudes(frame, mag);

    const out = c * bands;
    for (let j = 0; j < bands; j++) {
      const s = starts[j];
      const len = lengths[j];
      const off = offsets[j];
      let acc = 0;
      for (let k = 0; k < len; k++) acc += mag[s + k] * weights[off + k];

      const m = acc * norm;
      // 20*log10(m) as 10*log10(m^2), skipping the square root; the floor
      // doubles as the guard against log(0).
      const db = 10 * Math.log10(m * m + 1e-20);
      const v = (db - DB_FLOOR) * scale;
      bins[out + j] = v <= 0 ? 0 : v >= 255 ? 255 : v | 0;
    }

    if (onProgress && c % REPORT_EVERY === 0) onProgress(c / columns);
  }
  onProgress?.(1);

  return {
    bins,
    columns,
    binCount: bands,
    fftSize,
    hopSamples,
    sampleRate,
    secondsPerColumn: hopSamples / sampleRate,
    startSeconds: 0,
  };
}
