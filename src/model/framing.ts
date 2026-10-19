/**
 * How a buzzdetect model cuts a waveform into frames, and how a batch of those
 * frames is cut out of a file.
 *
 * The models take raw 16 kHz samples and return one row of activations per
 * 0.96 s frame -- the log-mel front end is inside the graph -- so everything the
 * browser has to get right is here: how many frames a file has, and which
 * samples each inference call is given. Both are read off the model's own
 * config_model.json rather than hard-coded, because they differ between
 * models: heavy_v1 needs nearly two frames of audio to produce its first one.
 *
 * Pure arithmetic with no runtime import, so the node tests can pin it.
 */

/** The parts of a model's config_model.json this app reads. */
export interface ModelConfig {
  description: string;
  /** Output columns, in the graph's order. Not recorded in the .onnx itself. */
  classes: string[];
  samplerate: number;
  framelength_s: number;
  /** Samples between the start of one frame and the next. */
  samples_hop: number;
  /** The fewest samples that produce a frame; larger than the hop. */
  samples_min: number;
}

/**
 * Frames buzzdetect reports for `n` samples: OnnxModel.n_frames, exactly.
 *
 * The hop division is a float32 multiply by the hop's float32 reciprocal,
 * because that is what the exported graph does, and at some exact multiples of
 * the hop it lands just above an integer and the ceil adds a frame. Real
 * arithmetic is one frame short there (61680 samples is one such length), so
 * every step goes through Math.fround, which rounds exactly as numpy's float32
 * casts do.
 */
export function frameCount(cfg: ModelConfig, n: number): number {
  if (n <= 0) return 0;
  const after = Math.fround(Math.max(0, n - cfg.samples_min));
  const reciprocal = Math.fround(1 / cfg.samples_hop);
  return 1 + Math.ceil(Math.fround(after * reciprocal));
}

/**
 * Samples one inference call is given for `count` frames: the frames
 * themselves, plus one spare frame of look-ahead.
 *
 * The spare frame is not optional. A frame's activation can depend on audio
 * past `samples_min` -- heavy_v1's last frame moves by up to 0.5 when its
 * neighbour is padding instead of audio -- so a batch cut exactly to its
 * frames would compute its final frame against zeros that are not in the file,
 * and the result would depend on where batches happened to split. With the
 * spare frame, every batch agrees with running the whole file at once.
 *
 * buzzdetect sizes its sessions the same way (OnnxModel.session_length) and
 * fills them with the audio after each chunk (WorkerStreamer.queue_chunk), so
 * the two agree at its chunk seams too. Releases before that fix scored each
 * chunk's last frame against zeros. At the end of a file both pad with zeros.
 */
export function batchSamples(cfg: ModelConfig, count: number): number {
  return cfg.samples_min + count * cfg.samples_hop;
}

/**
 * Copies the input for frames [first, first + count) into `out`, zero-padding
 * past the end of the audio. `out` must hold batchSamples(cfg, count) samples;
 * the whole of it is written, so a scratch buffer can be reused across calls.
 */
export function fillBatch(
  cfg: ModelConfig,
  wav: Float32Array,
  first: number,
  out: Float32Array,
): void {
  const from = first * cfg.samples_hop;
  const span = wav.subarray(Math.min(from, wav.length), Math.min(from + out.length, wav.length));
  out.set(span);
  out.fill(0, span.length);
}
