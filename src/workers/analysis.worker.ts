/**
 * Everything expensive happens here: the display STFT, the mel front end, and
 * model inference. The main thread only ever receives finished buffers, so
 * scrolling and playback stay smooth while a file is being analysed.
 *
 * Order of work is chosen for how it feels rather than for raw throughput. The
 * display spectrogram is computed and handed over first -- it takes a fraction
 * of a second and gives the user something to look at and scrub through -- and
 * only then does inference start, streaming activations back a batch at a time
 * so detections fill in progressively instead of arriving all at once.
 *
 * That means two passes over the audio (one STFT for display, one for mel)
 * rather than one fused pass. The second pass is a few hundred milliseconds
 * against inference's several seconds, and keeping them separate is what lets
 * the spectrogram appear immediately and the FFT size be changed later without
 * touching the model path.
 */

import {
  MelPatchExtractor,
  PATCH_VALUES,
  PATCH_FRAMES,
  HOP_SAMPLES,
  WINDOW_SAMPLES,
} from '../dsp/melspec';
import {
  computeSpectrogram,
  chooseHop,
  buildDisplayMelBank,
  DISPLAY_FFT_SIZE,
  type SpectrogramData,
} from '../dsp/spectrogram';
import { loadModel, runBatch, BATCH_PATCHES, CLASSES, type LoadedModel } from '../model/session';

export interface AnalyzeRequest {
  type: 'analyze';
  /** Mono samples at 16 kHz for the model. Transferred, not copied. */
  wav: Float32Array;
  /** Mono samples at the recording's own rate, for the spectrogram. */
  display: Float32Array;
  displayRate: number;
  framehopProp: number;
}

export type WorkerRequest =
  | AnalyzeRequest
  | { type: 'liveStart' }
  | { type: 'liveSamples'; samples: Float32Array }
  | { type: 'liveStop' };

export type WorkerResponse =
  | { type: 'ready'; threads: number; modelBytes: number }
  | { type: 'spectrogram'; data: SpectrogramData }
  | { type: 'progress'; stage: 'spectrogram' | 'inference'; fraction: number }
  | {
      type: 'results';
      firstPatch: number;
      activations: Float32Array;
      patchHopSeconds: number;
      totalPatches: number;
    }
  | { type: 'done'; totalPatches: number; elapsedMs: number }
  | { type: 'live'; patchIndex: number; activations: Float32Array }
  | { type: 'error'; message: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

let model: LoadedModel | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
}

async function ensureModel(): Promise<LoadedModel> {
  if (!model) {
    model = await loadModel();
    post({ type: 'ready', threads: model.threads, modelBytes: model.bytes });
  }
  return model;
}

/**
 * Builds the display spectrogram and hands it over.
 *
 * The display audio is only needed here, so the reference is dropped as soon as
 * the bands are computed -- at 48 kHz that array is the largest thing in the
 * session and there is no reason to hold it while inference runs.
 */
function buildSpectrogram(display: Float32Array, rate: number): void {
  const hop = chooseHop(display.length, rate);
  const bank = buildDisplayMelBank(DISPLAY_FFT_SIZE, rate);
  const data = computeSpectrogram(display, rate, DISPLAY_FFT_SIZE, hop, bank, (fraction) =>
    post({ type: 'progress', stage: 'spectrogram', fraction }),
  );
  // The byte buffer is the big one; hand ownership over rather than copying it.
  post({ type: 'spectrogram', data }, [data.bins.buffer]);
}

async function analyze(req: AnalyzeRequest): Promise<void> {
  const wav = req.wav;

  buildSpectrogram(req.display, req.displayRate);

  const loaded = await ensureModel();
  const extractor = new MelPatchExtractor(req.framehopProp);
  const total = extractor.patchCount(wav.length);
  const started = performance.now();

  // One scratch buffer for the whole run; batches are copied into it in place.
  const patches = new Float32Array(BATCH_PATCHES * PATCH_VALUES);

  for (let first = 0; first < total; first += BATCH_PATCHES) {
    const count = Math.min(BATCH_PATCHES, total - first);
    extractor.fillPatches(wav, first, count, patches);
    const activations = await runBatch(loaded.session, patches, count);

    // ORT reuses its output buffer between runs, so this has to be a copy.
    post({
      type: 'results',
      firstPatch: first,
      activations: activations.slice(0, count * CLASSES.length),
      patchHopSeconds: extractor.patchHopSeconds,
      totalPatches: total,
    });
    post({ type: 'progress', stage: 'inference', fraction: (first + count) / total });
  }

  post({ type: 'done', totalPatches: total, elapsedMs: performance.now() - started });
}

/**
 * Live microphone path.
 *
 * Frames are cut on the SAME grid the file path uses -- contiguous 0.96 s
 * frames starting at the moment capture began -- so a live reading and a
 * recorded one of the same sound are the same number, and the activation panel
 * can plot both without knowing which it has. That fixes the update rate at
 * about one frame per second; running the model more often on overlapping
 * windows would look livelier but would no longer be buzzdetect's framing.
 */
class LiveAnalyzer {
  private readonly extractor = new MelPatchExtractor(1);
  /** Samples one patch spans: 96 frames of 10 ms, plus the analysis window. */
  private readonly patchSamples = (PATCH_FRAMES - 1) * HOP_SAMPLES + WINDOW_SAMPLES;
  private readonly hopSamples = PATCH_FRAMES * HOP_SAMPLES;
  /** Buffered audio, and the absolute sample index of its first element. */
  private buffer = new Float32Array(0);
  private bufferStart = 0;
  private nextPatch = 0;
  private readonly patch = new Float32Array(PATCH_VALUES);
  private busy = false;

  async push(samples: Float32Array, loaded: LoadedModel): Promise<void> {
    const merged = new Float32Array(this.buffer.length + samples.length);
    merged.set(this.buffer);
    merged.set(samples, this.buffer.length);
    this.buffer = merged;

    // One patch per call at most. If inference falls behind, the buffer holds
    // the backlog and the next call drains it; dropping would leave gaps in a
    // series whose x-axis is frame index.
    if (this.busy) return;

    while (true) {
      const need = this.nextPatch * this.hopSamples + this.patchSamples;
      if (this.bufferStart + this.buffer.length < need) break;

      const from = this.nextPatch * this.hopSamples - this.bufferStart;
      const window = this.buffer.subarray(from, from + this.patchSamples);

      this.busy = true;
      try {
        this.extractor.fillPatches(window, 0, 1, this.patch);
        const activations = await runBatch(loaded.session, this.patch, 1);
        post({ type: 'live', patchIndex: this.nextPatch, activations: activations.slice() });
      } finally {
        this.busy = false;
      }

      this.nextPatch++;

      // Discard everything before the next patch's start.
      const drop = this.nextPatch * this.hopSamples - this.bufferStart;
      if (drop > 0) {
        this.buffer = this.buffer.slice(drop);
        this.bufferStart += drop;
      }
    }
  }
}

let live: LiveAnalyzer | null = null;

ctx.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  try {
    const msg = e.data;
    switch (msg.type) {
      case 'analyze':
        await analyze(msg);
        break;
      case 'liveStart':
        await ensureModel();
        live = new LiveAnalyzer();
        break;
      case 'liveSamples':
        if (live && model) await live.push(msg.samples, model);
        break;
      case 'liveStop':
        live = null;
        break;
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
