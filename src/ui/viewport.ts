/**
 * The time <-> pixel mapping shared by the spectrogram and the activation plot.
 *
 * Keeping it in one object is what makes the two panels stay locked together
 * while zooming and panning: they read the same transform rather than each
 * deriving one from scroll state.
 */

export class Viewport {
  /** Seconds at the left edge. */
  start = 0;
  /** Horizontal scale. */
  pixelsPerSecond = 100;
  /** Total length of the media, in seconds. */
  duration = 0;
  /** Width of the drawing area, in CSS pixels. */
  width = 0;

  get end(): number {
    return this.start + this.width / this.pixelsPerSecond;
  }

  get visibleSeconds(): number {
    return this.width / this.pixelsPerSecond;
  }

  timeToX(t: number): number {
    return (t - this.start) * this.pixelsPerSecond;
  }

  xToTime(x: number): number {
    return this.start + x / this.pixelsPerSecond;
  }

  /** Clamps the left edge so the view cannot be scrolled off the media. */
  clamp(): void {
    const maxStart = Math.max(0, this.duration - this.visibleSeconds);
    if (this.start > maxStart) this.start = maxStart;
    if (this.start < 0) this.start = 0;
  }

  panBy(seconds: number): void {
    this.start += seconds;
    this.clamp();
  }

  /**
   * Sets the scale about a fixed pixel, so the time under that pixel stays put.
   *
   * A pinch works in absolute scale rather than in factors: it has a finger
   * separation to hand, not an increment, and turning that into a factor per
   * move only to multiply it back in loses precision for nothing.
   */
  setScaleAt(x: number, pixelsPerSecond: number, minPps = 0.5, maxPps = 4000): void {
    const anchor = this.xToTime(x);
    const next = Math.min(maxPps, Math.max(minPps, pixelsPerSecond));
    if (next === this.pixelsPerSecond) return;
    this.pixelsPerSecond = next;
    this.start = anchor - x / this.pixelsPerSecond;
    this.clamp();
  }

  /** Zooms about a fixed pixel, so the time under the cursor stays put. */
  zoomAt(x: number, factor: number, minPps = 0.5, maxPps = 4000): void {
    this.setScaleAt(x, this.pixelsPerSecond * factor, minPps, maxPps);
  }

  /** Fits the whole media in view. */
  fit(): void {
    if (this.duration > 0 && this.width > 0) {
      this.pixelsPerSecond = this.width / this.duration;
      this.start = 0;
    }
  }

  /** Scrolls the minimum amount needed to bring `t` into view. */
  reveal(t: number, margin = 0.15): void {
    const span = this.visibleSeconds;
    const pad = span * margin;
    if (t < this.start + pad) this.start = t - pad;
    else if (t > this.end - pad) this.start = t - span + pad;
    this.clamp();
  }
}

/** A time range the user has dragged out. */
export interface Selection {
  from: number;
  to: number;
}

export function normalizeSelection(sel: Selection): Selection {
  return sel.from <= sel.to ? sel : { from: sel.to, to: sel.from };
}

/** Formats seconds as m:ss.mmm, or h:mm:ss.mmm past an hour. */
export function formatTime(t: number): string {
  if (!Number.isFinite(t) || t < 0) t = 0;
  const hours = Math.floor(t / 3600);
  const minutes = Math.floor((t % 3600) / 60);
  const seconds = t % 60;
  const ss = seconds.toFixed(3).padStart(6, '0');
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, '0')}:${ss}`;
  return `${minutes}:${ss}`;
}

/**
 * Seconds to a tenth, for the transport readout.
 *
 * A tenth is the useful precision beside a model whose frames are 0.96 s long;
 * milliseconds there are just noise that changes every repaint.
 */
export function formatTimeShort(t: number): string {
  if (!Number.isFinite(t) || t < 0) t = 0;
  const minutes = Math.floor(t / 60);
  const seconds = (t % 60).toFixed(1).padStart(4, '0');
  return `${minutes}:${seconds}`;
}
