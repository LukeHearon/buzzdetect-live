/**
 * Everything expensive happens here: the display STFT and model inference. The
 * main thread only ever receives finished buffers, so scrolling and playback
 * stay smooth while a file is being analysed.
 *
 * Order of work is chosen for how it feels rather than for raw throughput. The
 * display spectrogram is computed and handed over first -- it takes a fraction
 * of a second and gives the user something to look at and scrub through -- and
 * only then does inference start, streaming activations back a batch at a time
 * so detections fill in progressively instead of arriving all at once.
 *
 * That means two passes over the audio (one STFT for display, and the model's
 * own front end inside its graph) rather than one fused pass. Keeping them
 * separate is what lets the spectrogram appear immediately, and lets a model
 * switch re-run inference without recomputing the display.
 *
 * Switching models re-runs the audio already here. The worker keeps the 16 kHz
 * samples of whatever is loaded -- a file, or a finished recording handed over
 * with 'adopt' -- because the main thread transferred its copy away. A run is
 * abandoned at its next batch boundary when a newer one starts, so a switch
 * mid-analysis never interleaves two models' results.
 */

import {
  computeSpectrogram,
  chooseHop,
  buildDisplayMelBank,
  DISPLAY_FFT_SIZE,
  type SpectrogramData,
} from '../dsp/spectrogram';
import { batchSamples, frameCount } from '../model/framing';
import { DEFAULT_MODEL, type ModelName } from '../model/models';
import { loadModel, runFrames, scratchFor, BATCH_FRAMES, type LoadedModel } from '../model/session';

export interface AnalyzeRequest {
  type: 'analyze';
  /** Mono samples at 16 kHz for the model. Transferred, not copied. */
  wav: Float32Array;
  /** Mono samples at the recording's own rate, for the spectrogram. */
  display: Float32Array;
  displayRate: number;
}

export type WorkerRequest =
  | AnalyzeRequest
  | { type: 'model'; name: ModelName }
  /** A finished recording's 16 kHz samples, kept so a model switch can re-run them. */
  | { type: 'adopt'; wav: Float32Array }
  | { type: 'liveStart' }
  | { type: 'liveSamples'; samples: Float32Array }
  | { type: 'liveStop' };

export type WorkerResponse =
  | { type: 'ready'; model: ModelName; classes: string[]; threads: number; modelBytes: number }
  | { type: 'spectrogram'; data: SpectrogramData }
  | { type: 'progress'; stage: 'spectrogram' | 'inference'; fraction: number }
  | {
      type: 'results';
      firstPatch: number;
      activations: Float32Array;
      classCount: number;
      patchHopSeconds: number;
      totalPatches: number;
    }
  | { type: 'done'; model: ModelName; totalPatches: number; elapsedMs: number }
  | { type: 'live'; patchIndex: number; activations: Float32Array }
  | { type: 'error'; message: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

function post(msg: WorkerResponse, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
}

/**
 * Sessions by name, kept once built: switching back to a model is then
 * instant, and the cost is the wasm heap holding each one picked so far.
 */
const loaded = new Map<ModelName, Promise<LoadedModel>>();
let currentName: ModelName = DEFAULT_MODEL;

function getModel(name: ModelName): Promise<LoadedModel> {
  let model = loaded.get(name);
  if (!model) {
    model = loadModel(name);
    // A failed load is not cached, so picking the model again retries it.
    model.catch(() => loaded.delete(name));
    loaded.set(name, model);
  }
  return model;
}

/** The 16 kHz audio of whatever is loaded, for re-running on a model switch. */
let audio: Float32Array | null = null;
/** Bumped by every run, and by anything that makes a run in flight stale. */
let generation = 0;

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

/** Runs the current model over `wav`, streaming results, until done or superseded. */
async function infer(wav: Float32Array): Promise<void> {
  const run = ++generation;
  const model = await getModel(currentName);
  if (run !== generation) return;

  const { config } = model;
  const total = frameCount(config, wav.length);
  const classCount = config.classes.length;
  const patchHopSeconds = config.samples_hop / config.samplerate;
  const started = performance.now();
  // One scratch buffer for the whole run; every batch is padded to its length.
  const scratch = scratchFor(model, BATCH_FRAMES);

  for (let first = 0; first < total; first += BATCH_FRAMES) {
    const count = Math.min(BATCH_FRAMES, total - first);
    const activations = await runFrames(model, wav, first, count, scratch);
    if (run !== generation) return;

    post({
      type: 'results',
      firstPatch: first,
      activations,
      classCount,
      patchHopSeconds,
      totalPatches: total,
    });
    post({ type: 'progress', stage: 'inference', fraction: (first + count) / total });
  }

  post({ type: 'done', model: model.name, totalPatches: total, elapsedMs: performance.now() - started });
}

async function analyze(req: AnalyzeRequest): Promise<void> {
  audio = req.wav;
  // Supersede any run still going on the previous file before the (slow,
  // synchronous) spectrogram, so none of its batches land on the new one.
  generation++;
  buildSpectrogram(req.display, req.displayRate);
  await infer(audio);
}

async function selectModel(name: ModelName): Promise<void> {
  currentName = name;
  generation++;
  const model = await getModel(name);
  // Picked again while this one was downloading: the newer pick reports itself.
  if (currentName !== name) return;
  post({
    type: 'ready',
    model: name,
    classes: model.config.classes,
    threads: model.threads,
    modelBytes: model.bytes,
  });
  if (audio && !live) await infer(audio);
}

/**
 * Live microphone path.
 *
 * Frames are cut on the SAME grid the file path uses -- contiguous 0.96 s
 * frames starting at the moment capture began -- and each is run with the same
 * spare frame of look-ahead a file batch gets, so a live reading and a
 * recorded one of the same sound are the same number, and the activation panel
 * can plot both without knowing which it has. That fixes the update rate at
 * about one frame per second, with a frame's value arriving once the frame
 * after it has been heard. Running the model more often on overlapping windows
 * would look livelier but would no longer be buzzdetect's framing.
 */
class LiveAnalyzer {
  /** Samples one frame's input spans: the frame, its overhang, and the spare frame. */
  private readonly inputSamples: number;
  private readonly hopSamples: number;
  /** Buffered audio, and the absolute sample index of its first element. */
  private buffer = new Float32Array(0);
  private bufferStart = 0;
  private nextPatch = 0;
  private readonly scratch: Float32Array;
  private busy = false;

  constructor(private readonly model: LoadedModel) {
    this.inputSamples = batchSamples(model.config, 1);
    this.hopSamples = model.config.samples_hop;
    this.scratch = scratchFor(model, 1);
  }

  async push(samples: Float32Array): Promise<void> {
    const merged = new Float32Array(this.buffer.length + samples.length);
    merged.set(this.buffer);
    merged.set(samples, this.buffer.length);
    this.buffer = merged;

    // One frame per call at most. If inference falls behind, the buffer holds
    // the backlog and the next call drains it; dropping would leave gaps in a
    // series whose x-axis is frame index.
    if (this.busy) return;

    while (true) {
      const need = this.nextPatch * this.hopSamples + this.inputSamples;
      if (this.bufferStart + this.buffer.length < need) break;

      const from = this.nextPatch * this.hopSamples - this.bufferStart;
      const window = this.buffer.subarray(from, from + this.inputSamples);

      this.busy = true;
      try {
        const activations = await runFrames(this.model, window, 0, 1, this.scratch);
        post({ type: 'live', patchIndex: this.nextPatch, activations });
      } finally {
        this.busy = false;
      }

      this.nextPatch++;

      // Discard everything before the next frame's start.
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
      case 'model':
        // Sent once at startup too, so the download and the wasm compile
        // overlap with the user finding a file to open.
        await selectModel(msg.name);
        break;
      case 'analyze':
        await analyze(msg);
        break;
      case 'adopt':
        audio = msg.wav;
        break;
      case 'liveStart': {
        audio = null;
        generation++;
        const model = await getModel(currentName);
        live = new LiveAnalyzer(model);
        break;
      }
      case 'liveSamples':
        if (live) await live.push(msg.samples);
        break;
      case 'liveStop':
        live = null;
        break;
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
