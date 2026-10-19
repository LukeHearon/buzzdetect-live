/**
 * The models this app offers, and where their files are.
 *
 * Each is one of buzzdetect's released models, copied verbatim into
 * public/model/<name>/ by tools/models.py: the graph, and the config_model.json
 * buzzdetect itself reads for the class list and the framing. Nothing about a
 * model is mirrored here beyond its name, so a re-export only has to be
 * copied in.
 *
 * Kept free of the runtime so the main thread can read configs (for the picker
 * and the class list) without pulling onnxruntime into its bundle.
 */

import type { ModelConfig } from './framing';

/** Picker order: fastest first. */
export const MODEL_NAMES = ['fast_v1', 'balanced_v1', 'heavy_v1'] as const;

export type ModelName = (typeof MODEL_NAMES)[number];

/**
 * balanced_v1 at startup: a fifth of heavy_v1's 21 MB download, and
 * fewer false positives than fast_v1.
 */
export const DEFAULT_MODEL: ModelName = 'balanced_v1';

/**
 * Resolved against the deployment base rather than the document, because this
 * is also loaded inside a worker where `document` does not exist. Resolved per
 * call rather than at import, so the node tests can import MODEL_NAMES.
 */
export function modelUrl(name: ModelName, file: 'model.onnx' | 'config_model.json'): URL {
  const base = new URL(import.meta.env.BASE_URL, self.location.href);
  return new URL(`model/${name}/${file}`, base);
}

export async function fetchConfig(name: ModelName): Promise<ModelConfig> {
  const url = modelUrl(name, 'config_model.json');
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load ${name}'s config: ${url} returned HTTP ${response.status}.`);
  }
  return (await response.json()) as ModelConfig;
}
