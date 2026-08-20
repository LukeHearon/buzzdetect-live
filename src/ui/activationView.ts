/**
 * The buzzdetect panel, drawn in SeeNote's idiom: a polyline over the model's
 * frames, one dot per frame, filled where the frame clears the threshold and
 * hollow where it does not, with the frame boundaries hashed in behind and a
 * dashed threshold line in the series colour.
 *
 * Any subset of the model's thirteen outputs can be plotted at once, each with
 * its own colour and its own threshold (see ui/series.ts). They share one value
 * axis, because they are all in the same units and the interesting comparison is
 * which class is loudest at a moment -- separate axes would make that a
 * coincidence of scaling. The dashed threshold lines are per series and drawn
 * behind every polyline, so no line is broken up by another's cutoff.
 *
 * Values are the model's raw outputs, exactly as buzzdetect writes them to its
 * `activation_*` columns. They are not probabilities and are not squashed to
 * 0..1 -- the head has no softmax or sigmoid -- so the axis stays in the model's
 * own units and the threshold is quoted in those units too.
 *
 * The dots and the boundary hashes drop out below the widths where they stop
 * being legible (a dot narrower than a few pixels is noise, not a marker), so
 * zooming out degrades to a plain line instead of a smear.
 */

import { Viewport } from './viewport';
import { CLASSES } from '../model/session';
import type { Series } from './series';

/** Vertical breathing room so dots at the extremes are not clipped. */
const PAD_TOP = 12;
const PAD_BOTTOM = 12;
/** Below this frame width in pixels, per-frame dots stop being legible. */
const MIN_DOT_PX = 5;
/** Below this, frame boundary hashes crowd into a solid wash. */
const MIN_BOUNDARY_PX = 6;
const DOT_RADIUS = 3.5;
/**
 * Above this many series, per-frame dots stop helping: the markers of different
 * classes overlap more often than not, and the lines alone are easier to follow.
 */
const MAX_DOT_SERIES = 4;
/** Panel background; hollow dots are filled with it so the line does not show through. */
export const PANEL_BG = '#0b1220';
/**
 * Detected-frame background wash -- the same brighter dark blue as the app's
 * button chrome, so it reads as "lit up" against PANEL_BG without competing
 * with any series' own colour.
 */
const DETECTED_BG = '#16213a';

/**
 * Activations as they arrive, indexed [patch * CLASSES.length + class].
 *
 * `offset` is the absolute index of patch 0 in this buffer. It stays 0 for a
 * file; the live scope advances it when it drops old history, so callers can
 * keep talking in absolute session time.
 */
export class ActivationStore {
  values: Float32Array = new Float32Array(0);
  /** Patches held, counting from `offset`. */
  filled = 0;
  /** Capacity in patches. */
  total = 0;
  offset = 0;
  patchHopSeconds = 0.96;

  reset(total: number, patchHopSeconds: number): void {
    this.values = new Float32Array(total * CLASSES.length);
    this.filled = 0;
    this.total = total;
    this.offset = 0;
    this.patchHopSeconds = patchHopSeconds;
  }

  /** Stores a run of activations beginning at absolute patch index `firstPatch`. */
  put(firstPatch: number, activations: Float32Array): void {
    const local = firstPatch - this.offset;
    if (local < 0) return;
    const count = activations.length / CLASSES.length;
    if (local + count > this.total) return;
    this.values.set(activations, local * CLASSES.length);
    this.filled = Math.max(this.filled, local + count);
  }

  /** Drops the oldest `patches`, sliding the rest down. */
  dropOldest(patches: number): void {
    const n = Math.min(patches, this.filled);
    this.values.copyWithin(0, n * CLASSES.length, this.filled * CLASSES.length);
    this.filled -= n;
    this.offset += n;
  }

  /** Value at a LOCAL patch index. */
  get(patch: number, classIndex: number): number {
    return this.values[patch * CLASSES.length + classIndex];
  }

  /** Start time of a local patch -- the timestamp in buzzdetect's `start` column. */
  timeOf(patch: number): number {
    return (patch + this.offset) * this.patchHopSeconds;
  }

  /** Local patch index covering a time, which may be out of range. */
  patchAt(time: number): number {
    return Math.floor(time / this.patchHopSeconds) - this.offset;
  }

  /** Frames above `threshold`, as buzzdetect's `detections_*` column counts them. */
  countDetections(classIndex: number, threshold: number): number {
    let n = 0;
    for (let p = 0; p < this.filled; p++) {
      if (this.get(p, classIndex) > threshold) n++;
    }
    return n;
  }

  /** Largest value held for a class, for auto-ranging the axis. */
  range(classIndex: number): { min: number; max: number } {
    let min = Infinity;
    let max = -Infinity;
    for (let p = 0; p < this.filled; p++) {
      const v = this.get(p, classIndex);
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return Number.isFinite(min) ? { min, max } : { min: -4, max: 2 };
  }
}

export class ActivationView {
  readonly canvas: HTMLCanvasElement;
  readonly axis: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private axisCtx: CanvasRenderingContext2D;
  private dpr = 1;
  private width = 0;
  private height = 0;

  /** Value axis, in the model's activation units. */
  min = -4;
  max = 2;

  constructor(container: HTMLElement, axis: HTMLCanvasElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'act-canvas';
    container.append(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
    this.axis = axis;
    this.axisCtx = axis.getContext('2d')!;
  }

  resize(width: number, height: number, axisWidth: number): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.width = width;
    this.height = height;
    for (const [c, w] of [
      [this.canvas, width],
      [this.axis, axisWidth],
    ] as Array<[HTMLCanvasElement, number]>) {
      c.width = Math.max(1, Math.round(w * this.dpr));
      c.height = Math.max(1, Math.round(height * this.dpr));
      c.style.width = `${w}px`;
      c.style.height = `${height}px`;
    }
  }

  private yOf(v: number): number {
    const usable = this.height - PAD_TOP - PAD_BOTTOM;
    return PAD_TOP + (1 - (v - this.min) / (this.max - this.min)) * usable;
  }

  draw(
    store: ActivationStore,
    series: readonly Series[],
    viewport: Viewport,
    playhead: number | null,
    selection: { from: number; to: number } | null,
  ): void {
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = PANEL_BG;
    ctx.fillRect(0, 0, this.width, this.height);

    this.drawAxis();
    if (store.total === 0) return;

    const hop = store.patchHopSeconds;
    const framePx = hop * viewport.pixelsPerSecond;

    // Frames visible, clamped to what has actually arrived.
    const first = Math.max(0, store.patchAt(viewport.start) - 1);
    const last = Math.min(store.filled - 1, store.patchAt(viewport.end) + 1);
    if (last < first) return;

    // Whether any shown series clears its own threshold in each visible frame,
    // shared below by the background wash and by hollow dots (so a hollow dot
    // sitting inside a detected frame -- another series' detection -- reads
    // against that frame's actual background instead of the panel's).
    const detectedFrame = new Uint8Array(last - first + 1);
    for (let p = first; p <= last; p++) {
      for (const s of series) {
        if (store.get(p, s.index) > s.threshold) {
          detectedFrame[p - first] = 1;
          break;
        }
      }
    }

    // Detected-frame background: the span between a frame's boundaries gets a
    // brighter fill wherever it was detected, so a detection reads as a wash
    // you can see at a glance and not just the dot sitting on top of it.
    ctx.fillStyle = DETECTED_BG;
    for (let p = first; p <= last; p++) {
      if (!detectedFrame[p - first]) continue;
      const x0 = viewport.timeToX(store.timeOf(p));
      const x1 = viewport.timeToX(store.timeOf(p) + hop);
      if (x1 < 0 || x0 > this.width) continue;
      ctx.fillRect(x0, 0, x1 - x0, this.height);
    }

    if (selection) {
      ctx.fillStyle = 'rgba(226, 232, 240, 0.045)';
      const x0 = viewport.timeToX(selection.from);
      ctx.fillRect(x0, 0, viewport.timeToX(selection.to) - x0, this.height);
    }

    // Frame boundaries, so the model's 0.96 s grid is visible rather than
    // implied. Both edges of each frame are drawn, which is what makes a gap
    // read as a gap.
    if (framePx >= MIN_BOUNDARY_PX) {
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let p = first; p <= last + 1; p++) {
        const x = Math.round(viewport.timeToX(store.timeOf(p))) + 0.5;
        if (x < 0 || x > this.width) continue;
        ctx.moveTo(x, 0);
        ctx.lineTo(x, this.height);
      }
      ctx.stroke();
    }

    // Every threshold first, so one series' cutoff never lies on top of
    // another's data. Dashed, in the series colour at reduced alpha.
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    for (const s of series) {
      const ty = this.yOf(s.threshold);
      ctx.strokeStyle = `${s.color}66`;
      ctx.beginPath();
      ctx.moveTo(0, ty);
      ctx.lineTo(this.width, ty);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Dots only when frames are wide enough for them to read, and only while
    // few enough lines are up that they do not pile onto each other.
    const dots = framePx >= MIN_DOT_PX && series.length <= MAX_DOT_SERIES;

    for (const s of series) {
      // The polyline, one vertex per frame centre when frames are wide enough to
      // matter, otherwise one per pixel holding that pixel's peak -- a mean would
      // erase exactly the brief events being looked for.
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();

      if (framePx >= 1) {
        for (let p = first; p <= last; p++) {
          const x = viewport.timeToX(store.timeOf(p) + hop / 2);
          const y = this.yOf(store.get(p, s.index));
          if (p === first) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      } else {
        const secondsPerPixel = 1 / viewport.pixelsPerSecond;
        let started = false;
        for (let x = 0; x < this.width; x++) {
          const p0 = Math.max(first, store.patchAt(viewport.xToTime(x)));
          const p1 = Math.min(last, store.patchAt(viewport.xToTime(x) + secondsPerPixel));
          if (p1 < p0) continue;
          let peak = -Infinity;
          for (let p = p0; p <= p1; p++) {
            const v = store.get(p, s.index);
            if (v > peak) peak = v;
          }
          const y = this.yOf(peak);
          if (started) ctx.lineTo(x, y);
          else {
            ctx.moveTo(x, y);
            started = true;
          }
        }
      }
      ctx.stroke();

      // One dot per frame: filled above this series' threshold, hollow below.
      // A hollow dot's fill matches the frame's own background -- plain panel
      // background, or the detected wash if another series fired there --
      // so the line doesn't show through either way.
      if (!dots) continue;
      for (let p = first; p <= last; p++) {
        const cx = viewport.timeToX(store.timeOf(p) + hop / 2);
        if (cx < -DOT_RADIUS || cx > this.width + DOT_RADIUS) continue;
        const v = store.get(p, s.index);
        const cy = this.yOf(v);
        ctx.beginPath();
        ctx.arc(cx, cy, DOT_RADIUS, 0, Math.PI * 2);
        if (v > s.threshold) {
          ctx.fillStyle = s.color;
          ctx.fill();
        } else {
          ctx.fillStyle = detectedFrame[p - first] ? DETECTED_BG : PANEL_BG;
          ctx.fill();
          ctx.strokeStyle = s.color;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    if (playhead !== null) {
      const x = viewport.timeToX(playhead);
      ctx.strokeStyle = 'rgba(255,255,255,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, 0);
      ctx.lineTo(Math.round(x) + 0.5, this.height);
      ctx.stroke();
    }
  }

  /** Value labels in the left gutter, aligned with the spectrogram's. */
  private drawAxis(): void {
    const ctx = this.axisCtx;
    const w = this.axis.width / this.dpr;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, this.height);
    ctx.fillStyle = 'rgba(11,18,32,0.85)';
    ctx.fillRect(0, 0, w, this.height);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w - 0.5, 0);
    ctx.lineTo(w - 0.5, this.height);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const TICKS = 4;
    for (let k = 0; k <= TICKS; k++) {
      const v = this.min + (k / TICKS) * (this.max - this.min);
      const y = this.yOf(v);
      if (y < 8 || y > this.height - 6) continue;
      ctx.fillText(v.toFixed(1), w - 6, y);
    }
  }
}
