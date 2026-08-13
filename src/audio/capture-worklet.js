/**
 * Microphone tap.
 *
 * An AudioWorklet runs on the audio thread, where being late means a glitch, so
 * it does the least possible: copy the incoming 128-frame block into a buffer
 * and post it on once a useful amount has accumulated. All analysis happens
 * elsewhere.
 *
 * Blocks are batched to 2048 frames (128 ms at 16 kHz) rather than posted every
 * 128 frames, which cuts the message rate from ~125/s to ~8/s. The model's
 * window is 0.96 s, so this costs nothing that matters in responsiveness.
 */

const BATCH = 2048;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(BATCH);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;

    let read = 0;
    while (read < channel.length) {
      const room = BATCH - this.filled;
      const take = Math.min(room, channel.length - read);
      this.buffer.set(channel.subarray(read, read + take), this.filled);
      this.filled += take;
      read += take;

      if (this.filled === BATCH) {
        // Transfer a copy; the worklet keeps its own buffer for the next batch.
        const out = this.buffer.slice();
        this.port.postMessage(out, [out.buffer]);
        this.filled = 0;
      }
    }
    return true;
  }
}

registerProcessor('capture', CaptureProcessor);
