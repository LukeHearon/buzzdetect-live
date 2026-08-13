/**
 * Loading and running buzzdetect_v3.onnx.
 *
 * The graph is YAMNet's convolution stack with model_general_v3's dense head
 * fused onto it, taking log-mel patches and returning one activation per class.
 * See tools/03_build_onnx.py for how it is built and what was traded away.
 */

import * as ort from 'onnxruntime-web/wasm';
import { PATCH_FRAMES, MEL_BANDS, PATCH_VALUES } from '../dsp/melspec';

/**
 * The model and the runtime wasm live in public/ and are fetched at runtime
 * rather than imported, so the bundler never inlines 6.5 MB of weights into a
 * JS chunk and the browser can cache them separately from the code.
 *
 * Resolved against the deployment base rather than the document, because this
 * module is loaded inside a worker where `document` does not exist.
 */
const BASE = new URL(import.meta.env.BASE_URL, self.location.href);

export const MODEL_URL = new URL('model/buzzdetect_v3.onnx', BASE);

/**
 * Class order of the model's output. Fixed by config_model.json in
 * models/model_general_v3 -- the head's columns are in this order and nothing
 * in the file records it, so it has to be mirrored here.
 */
export const CLASSES = [
  'mech_train',
  'ins_trill',
  'frog',
  'ambient_noise',
  'mech_plane',
  'ambient_rain',
  'mech_hum',
  'mech_auto',
  'ins_buzz',
  'mech_siren',
  'ambient_background',
  'bird_goose',
  'human',
] as const;

export const BUZZ_INDEX = CLASSES.indexOf('ins_buzz');

/**
 * Patches per inference call.
 *
 * Bigger batches amortise call overhead but hold more activations live: at 32,
 * the input tensor is 32 * 96 * 64 * 4 = 786 kB, which stays comfortably inside
 * a laptop's cache budget while still giving the runtime enough work to keep
 * its threads busy. It also bounds how long the worker is unresponsive between
 * progress messages.
 */
export const BATCH_PATCHES = 32;

let configured = false;

function configure(): void {
  if (configured) return;
  configured = true;

  // wasmPaths is deliberately NOT set. onnxruntime-web resolves its own wasm
  // relative to its module URL, which the bundler rewrites to a hashed asset on
  // our own origin -- no CDN, works offline, cached by content. Pointing
  // wasmPaths at a hand-copied directory instead would ship a second, unhashed
  // 13 MB copy of the same binary.

  // Threads need SharedArrayBuffer, which needs the page to be cross-origin
  // isolated (COOP/COEP). Where that is not available -- plain static hosting,
  // for instance -- ORT must be told to stay single-threaded, or session
  // creation fails outright instead of degrading.
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  const cores = navigator.hardwareConcurrency || 2;
  // Leave a core for the UI and the audio thread; past four the convolutions
  // stop scaling and start contending.
  ort.env.wasm.numThreads = isolated ? Math.max(1, Math.min(4, cores - 1)) : 1;
  ort.env.logLevel = 'error';
}

export interface LoadedModel {
  session: ort.InferenceSession;
  /** True when the runtime was able to use more than one thread. */
  threaded: boolean;
  threads: number;
  /** Bytes of model downloaded. */
  bytes: number;
}

export async function loadModel(url: string | URL = MODEL_URL): Promise<LoadedModel> {
  configure();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not fetch model: ${response.status}`);
  const bytes = await response.arrayBuffer();

  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    // The graph has a dynamic batch dimension but is otherwise fully static, so
    // the runtime can plan its allocations once and reuse them.
    enableMemPattern: true,
  });

  const threads = ort.env.wasm.numThreads ?? 1;
  return { session, threaded: threads > 1, threads, bytes: bytes.byteLength };
}

/**
 * Runs one batch of patches.
 *
 * `patches` must hold `count * PATCH_VALUES` floats. Returns activations laid
 * out as `[count, CLASSES.length]`, row-major.
 */
export async function runBatch(
  session: ort.InferenceSession,
  patches: Float32Array,
  count: number,
): Promise<Float32Array> {
  const expected = count * PATCH_VALUES;
  const input = patches.length === expected ? patches : patches.subarray(0, expected);
  const tensor = new ort.Tensor('float32', input, [count, PATCH_FRAMES, MEL_BANDS]);
  const out = await session.run({ patches: tensor });
  return out.activations.data as Float32Array;
}
