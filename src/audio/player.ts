/**
 * Playback: an <audio> element routed through a Web Audio gain node.
 *
 * The element does the work -- it streams and decodes the original file itself,
 * so a long recording costs no extra memory, seeking is the browser's problem,
 * and you hear the file's full bandwidth rather than the model's downsampled
 * view of it.
 *
 * The gain node exists for one reason: `HTMLMediaElement.volume` is clamped to
 * 1, and field recordings are often quiet enough that unity is not loud enough
 * to hear a buzz. Routing through Web Audio allows boost above unity.
 *
 * The graph is built lazily on first play, because an AudioContext created
 * before a user gesture starts suspended.
 */

export class Player {
  readonly el: HTMLAudioElement;
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private gainValue = 1;
  private getSelection: (() => { from: number; to: number } | null) | null = null;

  constructor() {
    this.el = new Audio();
    this.el.preload = 'auto';
    // Volume is handled by the gain node; leave the element wide open so the
    // two do not multiply.
    this.el.volume = 1;

    // Stop at the end of a selection. `timeupdate` fires often enough to catch
    // it, and the rAF loop is already busy drawing.
    this.el.addEventListener('timeupdate', () => {
      const sel = this.getSelection?.();
      if (sel && this.el.currentTime >= sel.to) {
        this.el.pause();
        this.el.currentTime = sel.from;
      }
    });
  }

  private ensureGraph(): void {
    if (this.ctx) return;
    this.ctx = new AudioContext();
    const source = this.ctx.createMediaElementSource(this.el);
    this.gain = this.ctx.createGain();
    this.gain.gain.value = this.gainValue;
    source.connect(this.gain).connect(this.ctx.destination);
  }

  load(url: string): void {
    this.el.src = url;
  }

  get currentTime(): number {
    return this.el.currentTime;
  }

  set currentTime(t: number) {
    this.el.currentTime = Math.max(0, t);
  }

  get duration(): number {
    return Number.isFinite(this.el.duration) ? this.el.duration : 0;
  }

  get playing(): boolean {
    return !this.el.paused && !this.el.ended;
  }

  /** Output gain. 1 is unity; above that is real amplification. */
  get volume(): number {
    return this.gainValue;
  }

  set volume(v: number) {
    this.gainValue = Math.max(0, v);
    if (this.gain && this.ctx) {
      // Ramp rather than step, so dragging the slider does not click.
      this.gain.gain.setTargetAtTime(this.gainValue, this.ctx.currentTime, 0.01);
    }
  }

  /** Supplies the current selection, which bounds playback. */
  setSelection(get: (() => { from: number; to: number } | null) | null): void {
    this.getSelection = get;
  }

  /**
   * Starts playing.
   *
   * With a selection, playback always begins at its start -- the selection is
   * the thing being listened to, so resuming from wherever the playhead was
   * left is never what was meant.
   */
  async play(): Promise<void> {
    this.ensureGraph();
    if (this.ctx?.state === 'suspended') await this.ctx.resume();

    const sel = this.getSelection?.();
    if (sel) {
      const t = this.el.currentTime;
      if (t < sel.from || t >= sel.to - 1e-3) this.el.currentTime = sel.from;
    }
    await this.el.play().catch(() => undefined);
  }

  async toggle(): Promise<void> {
    if (this.playing) this.el.pause();
    else await this.play();
  }

  pause(): void {
    this.el.pause();
  }

  /** Rewinds to the start of the selection, or of the file. */
  toStart(): void {
    this.el.currentTime = this.getSelection?.()?.from ?? 0;
  }
}

/**
 * Maps a 0..1 slider position to gain, with unity at the centre.
 *
 * The lower half spans silence to unity and the upper half unity to 3x, so the
 * two ranges get equal travel -- attenuation and boost are equally reachable,
 * which a plain linear 0..3 scale would not give (unity would sit a third of
 * the way along).
 */
export function sliderToGain(position: number): number {
  return position <= 0.5 ? position * 2 : 1 + (position - 0.5) * 4;
}

export function gainToSlider(gain: number): number {
  return gain <= 1 ? gain / 2 : 0.5 + (gain - 1) / 4;
}
