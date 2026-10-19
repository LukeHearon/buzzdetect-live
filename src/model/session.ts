/**
 * Loading and running buzzdetect's models.
 *
 * Each graph is the whole of a buzzdetect model -- log-mel front end, trunk and
 * classifier head, fused by buzzdetect-training's export -- taking raw 16 kHz
 * samples and returning one row of activations per 0.96 s frame. These are
 * buzzdetect's released files, unchanged; see tools/models.py.
 *
 * The full-precision graphs are shipped rather than buzzdetect's fp16
 * siblings, which halve the download but move predictions by ~2e-2. heavy_v1 is
 * the cost of that choice at 21 MB; it is only fetched when picked.
 */

import * as ort from 'onnxruntime-web/wasm';
// Imported for its URL, not its contents: see the wasmPaths note in configure().
// The path is relative into node_modules because the package's exports map does
// not expose dist/, and going through it keeps the file version-locked to the
// installed onnxruntime-web rather than a copy that can drift.
import runtimeMjsUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';
import runtimeWasmUrl from '../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import { batchSamples, fillBatch, type ModelConfig } from './framing';
import { fetchConfig, modelUrl, type ModelName } from './models';

/**
 * Frames per inference call.
 *
 * Bigger batches amortise call overhead and the spare look-ahead frame each
 * one carries (see batchSamples), but bound how long the worker is
 * unresponsive between progress messages. At 32 the input is half a million
 * samples, 2 MB; heavy_v1 takes a few hundred milliseconds over it on one
 * thread.
 */
export const BATCH_FRAMES = 32;

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
  name: ModelName;
  session: ort.InferenceSession;
  config: ModelConfig;
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

export async function loadModel(name: ModelName): Promise<LoadedModel> {
  configure();
  const url = modelUrl(name, 'model.onnx');
  const [config, response] = await Promise.all([fetchConfig(name), fetch(url)]);
  if (!response.ok) {
    throw new Error(`Could not download ${name}: ${url} returned HTTP ${response.status}.`);
  }
  const bytes = await response.arrayBuffer();

  const [session, log, err] = await withFetchLog(() =>
    ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      // Every call is the same BATCH_FRAMES-sized input (the last one padded),
      // so the runtime can plan its allocations once and reuse them.
      enableMemPattern: true,
    }),
  );
  if (!session) throw explainRuntimeFailure(err, log);

  const threads = ort.env.wasm.numThreads ?? 1;
  return { name, session, config, threaded: threads > 1, threads, bytes: bytes.byteLength };
}

/**
 * Runs frames [first, first + count) of `wav`, `count` at most `capacity`.
 *
 * `scratch` is the input buffer, batchSamples(config, capacity) long, reused
 * across calls; the input is always that full length, zero-padded, so every
 * call has the same shape. Returns activations laid out as
 * `[count, classes.length]`, row-major, as a copy -- ORT reuses its output
 * buffer between runs.
 */
export async function runFrames(
  model: LoadedModel,
  wav: Float32Array,
  first: number,
  count: number,
  scratch: Float32Array,
): Promise<Float32Array> {
  fillBatch(model.config, wav, first, scratch);
  const tensor = new ort.Tensor('float32', scratch, [scratch.length]);
  const out = await model.session.run({ waveform: tensor });
  const data = out.predictions.data as Float32Array;
  return data.slice(0, count * model.config.classes.length);
}

/** A scratch input buffer for runFrames, holding up to `capacity` frames. */
export function scratchFor(model: LoadedModel, capacity: number): Float32Array {
  return new Float32Array(batchSamples(model.config, capacity));
}
