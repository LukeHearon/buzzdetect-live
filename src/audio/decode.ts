/**
 * Getting a file into the two forms the app needs.
 *
 * Nothing is uploaded: the File is read into memory in the tab and decoded by
 * the browser's own decoder.
 *
 * The file is decoded TWICE, at two rates, because the two consumers want
 * different things:
 *
 *   - The model gets mono 16 kHz, decoded straight to that rate. This is the
 *     path the parity tests cover, so it is left exactly as it was rather than
 *     being derived from the display copy by a second resampling step.
 *   - The display gets the highest rate worth keeping, so the spectrogram shows
 *     the recording's own bandwidth instead of the model's 8 kHz view of it.
 *
 * Decoding twice costs a second or so on a five-minute file and avoids
 * cascading two resamplers on the path whose numbers have to match buzzdetect.
 * The display copy is handed to the worker and dropped as soon as the
 * spectrogram is built, so both are not held for long.
 *
 * Playback is separate again: an <audio> element streams the original file, so
 * a long recording costs no extra memory and you hear its full bandwidth.
 */

export const MODEL_SAMPLE_RATE = 16000;

/** Duration past which a file is worth warning about. */
export const LONG_FILE_SECONDS = 20 * 60;

/**
 * Display rate by duration.
 *
 * 48 kHz gives a 24 kHz ceiling, past anything a passive recorder puts down.
 * Mono float at 48 kHz is ~11.5 MB per minute, so longer files step down rather
 * than allocating a few hundred megabytes for detail no one is looking at.
 */
function displayRateFor(durationSeconds: number): number {
  if (durationSeconds <= 10 * 60) return 48000;
  if (durationSeconds <= 30 * 60) return 24000;
  return MODEL_SAMPLE_RATE;
}

export interface DecodedAudio {
  /** Mono 16 kHz -- the model's input. */
  wav: Float32Array;
  /** Mono at `displayRate` -- the spectrogram's input. */
  display: Float32Array;
  displayRate: number;
  duration: number;
  channels: number;
  /** Object URL for the <audio> element. Revoke when the file is replaced. */
  url: string;
  /** Peak absolute sample, so a silent decode can be reported rather than guessed at. */
  peak: number;
}

async function decodeAt(bytes: ArrayBuffer, sampleRate: number): Promise<AudioBuffer> {
  // A 1-frame OfflineAudioContext is just a decoding host: decodeAudioData
  // resamples to the context's rate and ignores its length.
  const ctx = new OfflineAudioContext(1, 1, sampleRate);
  // slice() because decodeAudioData detaches the buffer it is given.
  return ctx.decodeAudioData(bytes.slice(0));
}

export async function decodeForAnalysis(file: File): Promise<DecodedAudio> {
  const bytes = await file.arrayBuffer();

  let modelBuffer: AudioBuffer;
  try {
    modelBuffer = await decodeAt(bytes, MODEL_SAMPLE_RATE);
  } catch {
    throw new Error(
      `Could not decode ${file.name}. Browsers handle wav, mp3, m4a/aac, ogg and flac; ` +
        `formats like wma need buzzdetect proper.`,
    );
  }

  const duration = modelBuffer.duration;
  const displayRate = displayRateFor(duration);
  const displayBuffer =
    displayRate === MODEL_SAMPLE_RATE ? modelBuffer : await decodeAt(bytes, displayRate);

  const wav = downmix(modelBuffer);
  const display = displayBuffer === modelBuffer ? wav : downmix(displayBuffer);

  let peak = 0;
  for (let i = 0; i < wav.length; i++) {
    const a = wav[i] < 0 ? -wav[i] : wav[i];
    if (a > peak) peak = a;
  }

  return {
    wav,
    display,
    displayRate,
    duration,
    channels: modelBuffer.numberOfChannels,
    url: URL.createObjectURL(new Blob([bytes], { type: file.type || 'audio/*' })),
    peak,
  };
}

/**
 * Averages channels to mono, matching what buzzdetect does when it reads a
 * multi-channel file (`samples.mean(axis=1)`).
 */
function downmix(buffer: AudioBuffer): Float32Array {
  const n = buffer.length;
  if (buffer.numberOfChannels === 1) {
    // getChannelData returns a live view into the AudioBuffer; copy so the
    // decoded buffer can be collected.
    return new Float32Array(buffer.getChannelData(0));
  }

  const out = new Float32Array(n);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < n; i++) out[i] += data[i];
  }
  const scale = 1 / buffer.numberOfChannels;
  for (let i = 0; i < n; i++) out[i] *= scale;
  return out;
}

/**
 * Wraps mono float samples as a 16-bit PCM WAV blob.
 *
 * Used to hand recorded microphone audio to the <audio> element once capture
 * stops, so a live session can be replayed like a file. 16-bit rather than
 * float keeps the blob half the size for playback that is going through a
 * speaker anyway.
 */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);

  const ascii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  ascii(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM header length
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    offset += 2;
  }

  return new Blob([bytes], { type: 'audio/wav' });
}
