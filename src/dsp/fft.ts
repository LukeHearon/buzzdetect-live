/**
 * Radix-2 FFT with a real-input specialisation.
 *
 * Both the model's mel front end and the display spectrogram run an STFT over
 * every sample of the file, so this is the hottest code in the app. Two things
 * keep it cheap:
 *
 *   - Tables (twiddle factors, bit-reversal permutation) are built once per FFT
 *     size and shared, and every buffer is allocated once and reused. A
 *     transform allocates nothing.
 *   - Real input of length N is transformed with a complex FFT of length N/2
 *     plus an untangle pass, which is the standard two-for-one trick and close
 *     to half the work of zero-stuffing the imaginary part.
 */

/** Complex radix-2 FFT of a fixed power-of-two size, operating in place. */
export class Fft {
  readonly size: number;
  private readonly cos: Float32Array;
  private readonly sin: Float32Array;
  private readonly rev: Uint32Array;

  constructor(size: number) {
    if (size < 2 || (size & (size - 1)) !== 0) {
      throw new Error(`FFT size must be a power of two, got ${size}`);
    }
    this.size = size;

    // Twiddles for the forward transform: e^(-2*pi*i*k/size), k < size/2.
    this.cos = new Float32Array(size / 2);
    this.sin = new Float32Array(size / 2);
    for (let k = 0; k < size / 2; k++) {
      this.cos[k] = Math.cos((-2 * Math.PI * k) / size);
      this.sin[k] = Math.sin((-2 * Math.PI * k) / size);
    }

    // Bit-reversal permutation, so the butterflies below can run in place.
    const bits = Math.log2(size);
    this.rev = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >>> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  /** In-place forward transform of the interleaved-free (re, im) pair. */
  transform(re: Float32Array, im: Float32Array): void {
    const { size, cos, sin, rev } = this;

    for (let i = 0; i < size; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }

    for (let len = 2; len <= size; len <<= 1) {
      const half = len >> 1;
      const step = size / len;
      for (let i = 0; i < size; i += len) {
        for (let k = 0, tw = 0; k < half; k++, tw += step) {
          const wr = cos[tw];
          const wi = sin[tw];
          const a = i + k;
          const b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
        }
      }
    }
  }
}

/**
 * Magnitude spectrum of a real signal, computed via a half-length complex FFT.
 *
 * `magnitudes()` fills bins 0..size/2 inclusive (size/2 + 1 values), matching
 * the output width of `tf.signal.rfft`.
 */
export class RealFft {
  readonly size: number;
  readonly bins: number;
  private readonly fft: Fft;
  private readonly re: Float32Array;
  private readonly im: Float32Array;
  private readonly halfCos: Float32Array;
  private readonly halfSin: Float32Array;

  constructor(size: number) {
    this.size = size;
    this.bins = size / 2 + 1;
    this.fft = new Fft(size / 2);
    this.re = new Float32Array(size / 2);
    this.im = new Float32Array(size / 2);

    // Untangle twiddles: e^(-2*pi*i*k/size) for k < size/2.
    this.halfCos = new Float32Array(size / 2);
    this.halfSin = new Float32Array(size / 2);
    for (let k = 0; k < size / 2; k++) {
      this.halfCos[k] = Math.cos((-2 * Math.PI * k) / size);
      this.halfSin[k] = Math.sin((-2 * Math.PI * k) / size);
    }
  }

  /**
   * Writes |X[k]| for k = 0..size/2 into `out`.
   *
   * `input` must hold exactly `size` real samples (zero-pad before calling if
   * the analysis window is shorter than the FFT).
   */
  magnitudes(input: Float32Array, out: Float32Array): void {
    const m = this.size / 2;
    const { re, im, halfCos, halfSin } = this;

    // Pack even samples as the real part, odd as the imaginary part.
    for (let k = 0; k < m; k++) {
      re[k] = input[2 * k];
      im[k] = input[2 * k + 1];
    }
    this.fft.transform(re, im);

    // k = 0 and k = m are both purely real and share Z[0].
    out[0] = Math.abs(re[0] + im[0]);
    out[m] = Math.abs(re[0] - im[0]);

    for (let k = 1; k < m; k++) {
      const j = m - k;
      // Even/odd split: Fe = (Z[k] + conj(Z[m-k]))/2, Fo = (Z[k] - conj(Z[m-k]))/2i
      const fer = 0.5 * (re[k] + re[j]);
      const fei = 0.5 * (im[k] - im[j]);
      const for_ = 0.5 * (im[k] + im[j]);
      const foi = -0.5 * (re[k] - re[j]);

      const wr = halfCos[k];
      const wi = halfSin[k];
      const xr = fer + (for_ * wr - foi * wi);
      const xi = fei + (for_ * wi + foi * wr);
      out[k] = Math.hypot(xr, xi);
    }
  }
}

/**
 * Periodic Hann window, `0.5 - 0.5*cos(2*pi*n/N)`.
 *
 * Periodic (divisor N) rather than symmetric (divisor N-1) because that is what
 * `tf.signal.stft` applies by default, and the mel front end has to agree with
 * TensorFlow sample for sample.
 */
export function hannPeriodic(length: number): Float32Array {
  const w = new Float32Array(length);
  for (let n = 0; n < length; n++) w[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / length);
  return w;
}
