/**
 * Live microphone capture, feeding the same spectrogram and activation panels
 * the file path uses.
 *
 * The spectrogram grows to the right and the viewport follows it, so old audio
 * pans off the left edge. History is bounded: the buffer holds ~5 minutes and,
 * when it fills, the oldest half is dropped in one move and `startSeconds`
 * advances. That costs a single 4 MB memmove every two and a half minutes,
 * against the alternative of shifting the whole buffer every 10 ms.
 *
 * The half is sized to a whole number of model frames, so the spectrogram and
 * the activation store can drop the same span of time and stay aligned.
 *
 * Display columns are computed here on the main thread rather than in the
 * worker: at 16 kHz with a 10 ms hop that is a hundred 512-point FFTs per
 * second, a fraction of a millisecond of work, and doing it locally keeps the
 * drawn column in step with the audio instead of a message round-trip behind
 * it. Inference does go to the worker, being milliseconds of work that would
 * otherwise land on the frame budget.
 */

import { RealFft, hannPeriodic } from '../dsp/fft';
import {
  DB_FLOOR,
  DB_CEIL,
  buildDisplayMelBank,
  type MelBank,
  type SpectrogramData,
} from '../dsp/spectrogram';
import { PATCH_FRAMES, HOP_SAMPLES } from '../dsp/melspec';

// Smaller than the file path's 1024 because a live column must be produced
// every 10 ms from audio that has only just arrived; 512 keeps the window
// inside one hop's worth of latency.
const FFT_SIZE = 512;
/** 10 ms, matching the model's frame rate so one patch is exactly 96 columns. */
const HOP = HOP_SAMPLES;
const SAMPLE_RATE = 16000;
/** Columns spanned by one model frame. */
const COLUMNS_PER_PATCH = PATCH_FRAMES;
/** Frames of history dropped per compaction; 156 * 0.96 s = 149.76 s. */
const DROP_PATCHES = 156;
const DROP_COLUMNS = DROP_PATCHES * COLUMNS_PER_PATCH;
/** Capacity: two drops' worth, so ~5 minutes are on screen at the widest zoom. */
const CAPACITY_COLUMNS = DROP_COLUMNS * 2;

export interface LiveSessionOptions {
  /** Called with each batch of captured samples, for inference. */
  onSamples: (samples: Float32Array) => void;
  /**
   * Called when old history is dropped, with the number of model frames
   * discarded, so the activation store can drop the same span.
   */
  onDrop: (patches: number) => void;
}

export class LiveSession {
  /** Shared with SpectrogramView; grows in place. */
  readonly data: SpectrogramData;

  private fft = new RealFft(FFT_SIZE);
  private window = hannPeriodic(FFT_SIZE);
  private frame = new Float32Array(FFT_SIZE);
  private mag = new Float32Array(FFT_SIZE / 2 + 1);
  private bank: MelBank;
  private pending = new Float32Array(FFT_SIZE);
  private pendingFilled = 0;

  /**
   * The captured audio itself, kept so the session can be played back after
   * capture stops. Bounded to the same span as the spectrogram and trimmed in
   * step with it: ~5 minutes of 16 kHz mono is 19 MB, and holding an unbounded
   * recording would be the one thing in the app that grows without limit.
   */
  private recording: Float32Array;
  private recorded = 0;

  private audio: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;

  /** Samples captured since the session started. */
  private samplesSeen = 0;
  /** Peak of the most recent block, 0..1, for the input meter. */
  level = 0;

  constructor(private options: LiveSessionOptions) {
    this.bank = buildDisplayMelBank(FFT_SIZE, SAMPLE_RATE);
    const binCount = this.bank.bands;
    this.recording = new Float32Array(CAPACITY_COLUMNS * HOP + FFT_SIZE);
    this.data = {
      bins: new Uint8Array(CAPACITY_COLUMNS * binCount),
      columns: 0,
      binCount,
      fftSize: FFT_SIZE,
      hopSamples: HOP,
      sampleRate: SAMPLE_RATE,
      secondsPerColumn: HOP / SAMPLE_RATE,
      startSeconds: 0,
    };
  }

  /** Seconds of audio captured, which is the live timeline's length. */
  get elapsed(): number {
    return this.samplesSeen / SAMPLE_RATE;
  }

  get running(): boolean {
    return this.audio !== null;
  }

  async start(): Promise<void> {
    if (this.audio) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // The model was trained on unprocessed field recordings; browser voice
        // processing would gate and squash exactly the sounds being looked for.
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    // Asking the context for 16 kHz makes the browser resample the mic for us,
    // so capture arrives at the model's rate with no extra work.
    this.audio = new AudioContext({ sampleRate: SAMPLE_RATE });
    await this.audio.audioWorklet.addModule(
      new URL('../audio/capture-worklet.js', import.meta.url),
    );

    const source = this.audio.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.audio, 'capture', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
    });
    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => this.onBatch(e.data);
    source.connect(this.node);
  }

  async stop(): Promise<void> {
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.audio?.close();
    this.node = null;
    this.stream = null;
    this.audio = null;
    this.level = 0;
  }

  private onBatch(samples: Float32Array): void {
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
      const a = Math.abs(samples[i]);
      if (a > peak) peak = a;
    }
    // Decay slowly so the meter reads rather than flickers.
    this.level = Math.max(peak, this.level * 0.85);

    this.samplesSeen += samples.length;
    this.record(samples);
    this.appendColumns(samples);
    this.options.onSamples(samples);
  }

  /**
   * Appends samples, emitting one spectrogram column per completed hop.
   *
   * `pending` keeps the tail of the previous batch so windows straddling a
   * batch boundary stay whole; otherwise every boundary would show as a seam.
   */
  private appendColumns(samples: Float32Array): void {
    let offset = 0;
    while (offset < samples.length) {
      const take = Math.min(FFT_SIZE - this.pendingFilled, samples.length - offset);
      this.pending.set(samples.subarray(offset, offset + take), this.pendingFilled);
      this.pendingFilled += take;
      offset += take;

      if (this.pendingFilled < FFT_SIZE) return;

      this.emitColumn();
      this.pending.copyWithin(0, HOP, FFT_SIZE);
      this.pendingFilled = FFT_SIZE - HOP;
    }
  }

  private record(samples: Float32Array): void {
    const room = this.recording.length - this.recorded;
    const take = Math.min(samples.length, room);
    this.recording.set(samples.subarray(0, take), this.recorded);
    this.recorded += take;
  }

  private emitColumn(): void {
    const data = this.data;
    if (data.columns >= CAPACITY_COLUMNS) this.compact();

    for (let i = 0; i < FFT_SIZE; i++) this.frame[i] = this.pending[i] * this.window[i];
    this.fft.magnitudes(this.frame, this.mag);

    const norm = 4 / FFT_SIZE;
    const scale = 255 / (DB_CEIL - DB_FLOOR);
    const { starts, lengths, offsets, weights } = this.bank;
    const out = data.columns * data.binCount;

    for (let j = 0; j < data.binCount; j++) {
      let acc = 0;
      const s = starts[j];
      const off = offsets[j];
      for (let k = 0; k < lengths[j]; k++) acc += this.mag[s + k] * weights[off + k];
      const m = acc * norm;
      const db = 10 * Math.log10(m * m + 1e-20);
      const v = (db - DB_FLOOR) * scale;
      data.bins[out + j] = v <= 0 ? 0 : v >= 255 ? 255 : v | 0;
    }
    data.columns++;
  }

  /** Drops the oldest half of the history -- spectrogram and audio together. */
  private compact(): void {
    const data = this.data;
    data.bins.copyWithin(0, DROP_COLUMNS * data.binCount, data.columns * data.binCount);
    data.columns -= DROP_COLUMNS;
    data.startSeconds += DROP_COLUMNS * data.secondsPerColumn;

    const dropSamples = DROP_COLUMNS * HOP;
    this.recording.copyWithin(0, dropSamples, this.recorded);
    this.recorded -= dropSamples;

    this.options.onDrop(DROP_PATCHES);
  }

  /**
   * The retained audio, and the session time it starts at.
   *
   * After a compaction the buffer no longer begins at t = 0, so the caller has
   * to know where it does before lining it up with the spectrogram.
   */
  takeRecording(): { samples: Float32Array; startSeconds: number; sampleRate: number } {
    return {
      samples: this.recording.subarray(0, this.recorded),
      startSeconds: this.data.startSeconds,
      sampleRate: SAMPLE_RATE,
    };
  }

  /**
   * Rebases the timeline so the retained history starts at zero.
   *
   * Called when capture stops: from then on the session behaves like a loaded
   * file, and a file starts at 0. Without this the playhead of the recorded WAV
   * would be offset from the spectrogram by however much history was dropped.
   */
  rebase(): number {
    const shifted = this.data.startSeconds;
    this.data.startSeconds = 0;
    return shifted;
  }
}

/** Frames of activation history the live store should hold. */
export const LIVE_STORE_PATCHES = DROP_PATCHES * 2;
