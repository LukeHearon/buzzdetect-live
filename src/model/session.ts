/**
 * Loading and running buzzdetect_v3.onnx.
 *
 * The graph is YAMNet's convolution stack with the yamnet_large_general dense
 * head fused onto it, taking log-mel patches and returning one activation per
 * class. See tools/03_build_onnx.py for how it is built and what was traded
 * away.
 *
 * yamnet_large_general is experimental: it hasn't been validated as
 * thoroughly as buzzdetect's earlier models. See DEFAULT_THRESHOLD's note in
 * ui/series.ts and the caveat surfaced in the help dialog.
 */

import * as ort from 'onnxruntime-web/wasm';
// Imported for its URL, not its contents: see the wasmPaths note in configure().
// The path is relative into node_modules because the package's exports map does
// not expose dist/, and going through it keeps the file version-locked to the
// installed onnxruntime-web rather than a copy that can drift.
import runtimeMjsUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import runtimeWasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
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
 * models/yamnet_large_general -- the head's columns are in this order and
 * nothing in the file records it, so it has to be mirrored here. Only a
 * subset is surfaced in the UI; see DISPLAY_ORDER in ui/series.ts.
 */
export const CLASSES = [
  'ambient_background',
  'ambient_music',
  'ambient_noise',
  'ambient_rain',
  'ambient_ringing',
  'animal',
  'human',
  'ins_buzz',
  'ins_trill',
  'mech_ac',
  'mech_auto',
  'mech_beep',
  'mech_hum',
  'mech_hum_chainsaw',
  'mech_plane',
  'mech_tool',
  'static',
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

  // Both runtime files are named explicitly, because the runtime cannot locate
  // either one from inside a bundle.
  //
  // To start pthreads it spawns workers from ort-wasm-simd-threaded.mjs, and
  // when it cannot use its inlined copy it derives that name from the script
  // directory -- a file the build never emits. Under `vite dev` the request
  // happens to hit the real one in node_modules, so this only fails once built:
  // the spawn 404s, and a worker that never starts leaves
  // InferenceSession.create pending forever. Nothing throws, so it presents as
  // a load that hangs rather than an error.
  //
  // Setting wasmPaths then routes *every* runtime file through it, the .wasm
  // included, so that one must be named too or it 404s in turn. Importing both
  // for their URLs keeps them ordinary build artifacts: hashed, on our own
  // origin, no CDN, and version-locked to the installed onnxruntime-web. The
  // bundler already emitted this same .wasm for its own internal reference and
  // dedupes by source path, so there is still only one 13 MB copy.
  //
  // Absolutised: the bundler emits root-relative paths, and the runtime hands
  // whatever it is given straight to `new URL()` with no base.
  ort.env.wasm.wasmPaths = {
    mjs: new URL(runtimeMjsUrl, self.location.href).href,
    wasm: new URL(runtimeWasmUrl, self.location.href).href,
  };

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

/**
 * What ORT reports when its runtime fails to load is "no available backend
 * found. ERR: [wasm] RuntimeError: Aborted(CompileError: ... failed to match
 * magic number)" -- which says only that something that wasn't WebAssembly was
 * handed to the compiler. The useful fact is *which* request returned it, and
 * only the loader knows that, so record its fetches for the duration of the
 * call and name the culprit if it fails.
 *
 * A misconfigured static host is the expected cause: the runtime's own assets
 * 404 to an HTML error page, and the compiler is fed that HTML.
 */
interface FetchRecord {
  url: string;
  status: number;
  type: string;
}

async function withFetchLog<T>(fn: () => Promise<T>): Promise<[T | undefined, FetchRecord[], unknown]> {
  const log: FetchRecord[] = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (...args: Parameters<typeof fetch>) => {
    const response = await original(...args);
    log.push({
      url: typeof args[0] === 'string' ? args[0] : String((args[0] as Request).url ?? args[0]),
      status: response.status,
      type: response.headers.get('content-type') ?? 'unknown',
    });
    return response;
  };
  try {
    return [await fn(), log, undefined];
  } catch (err) {
    return [undefined, log, err];
  } finally {
    globalThis.fetch = original;
  }
}

function explainRuntimeFailure(err: unknown, log: FetchRecord[]): Error {
  const detail = err instanceof Error ? err.message : String(err);
  const bad = log.find((r) => !(r.status >= 200 && r.status < 300));
  const html = log.find((r) => r.type.includes('html'));

  if (bad) {
    return new Error(
      `The WebAssembly runtime could not be loaded: ${bad.url} returned HTTP ${bad.status} ` +
        `(${bad.type}). Its file is missing from the deployment.`,
    );
  }
  if (html) {
    return new Error(
      `The WebAssembly runtime could not be loaded: ${html.url} was served as ${html.type}, ` +
        `not application/wasm. The host is likely returning an error page in its place.`,
    );
  }
  if (typeof WebAssembly === 'undefined') {
    return new Error('This browser has no WebAssembly support, which the model requires.');
  }
  return new Error(`The model could not be started: ${detail}`);
}

export async function loadModel(url: string | URL = MODEL_URL): Promise<LoadedModel> {
  configure();
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not download the model: ${url} returned HTTP ${response.status}.`);
  }
  const bytes = await response.arrayBuffer();

  const [session, log, err] = await withFetchLog(() =>
    ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      // The graph has a dynamic batch dimension but is otherwise fully static, so
      // the runtime can plan its allocations once and reuse them.
      enableMemPattern: true,
    }),
  );
  if (!session) throw explainRuntimeFailure(err, log);

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
