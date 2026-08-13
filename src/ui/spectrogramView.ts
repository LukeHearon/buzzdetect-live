/**
 * Canvas spectrogram: draws the mel-band byte buffer from dsp/spectrogram.ts
 * through the shared viewport.
 *
 * Two layers, because they change at very different rates. The spectrogram is
 * redrawn when the view or the settings change; the playhead and selection sit
 * on a transparent canvas above it, redrawn every frame during playback.
 * Compositing two canvases costs the GPU nothing and saves recomputing the
 * image for a moving one-pixel line.
 *
 * The vertical axis is linear in mel between the displayed frequency limits, so
 * the low end -- where a buzz lives -- gets the height, and the picture matches
 * the features the model works from.
 *
 * The pixel loop caps how many columns it reads per screen pixel. Without that,
 * zooming out on a long file turns one repaint into tens of millions of reads
 * and the gesture stutters on a slow machine.
 */

import { bandForHz, DB_CEIL, DB_FLOOR, type SpectrogramData } from '../dsp/spectrogram';
import { hertzToMel, melToHertz } from '../dsp/melspec';
import { colormap, levelTable, type ColormapName } from './colormap';
import { Viewport, type Selection } from './viewport';

/** Most spectrogram columns read per screen pixel when zoomed out. */
const MAX_COLUMNS_PER_PIXEL = 8;

export interface SpectrogramSettings {
  /** Displayed frequency range, in Hz. */
  minHz: number;
  maxHz: number;
  /** Level window, in stored byte units (see dsp/spectrogram.ts). */
  minByte: number;
  maxByte: number;
  colormap: ColormapName;
}

export const DEFAULT_SETTINGS: SpectrogramSettings = {
  // Insect flight sits well under 6 kHz, and YAMNet's mel bands stop at 7.5 kHz;
  // showing more than this spends height on rows that are never the answer.
  minHz: 0,
  maxHz: 6000,
  // -120..0 dBFS maps to 0..255; field recordings sit low, so the default
  // window opens at about -95 dBFS rather than showing mostly black.
  minByte: 53,
  maxByte: 200,
  colormap: 'magma',
};

export class SpectrogramView {
  readonly base: HTMLCanvasElement;
  readonly overlay: HTMLCanvasElement;
  readonly axis: HTMLCanvasElement;
  readonly colorscale: HTMLCanvasElement;
  readonly viewport = new Viewport();
  settings: SpectrogramSettings = { ...DEFAULT_SETTINGS };

  private data: SpectrogramData | null = null;
  private baseCtx: CanvasRenderingContext2D;
  private overlayCtx: CanvasRenderingContext2D;
  private axisCtx: CanvasRenderingContext2D;
  private colorscaleCtx: CanvasRenderingContext2D;
  private image: ImageData | null = null;
  private bandForRow = new Int32Array(0);
  private dpr = 1;
  private baseDirty = true;
  /** contentKey() of the image currently on the canvas; '' when there is none. */
  private drawnKey = '';
  /** Time at the left edge of that image, which the blit keeps exact. */
  private drawnStart = 0;

  constructor(container: HTMLElement, axis: HTMLCanvasElement, colorscale: HTMLCanvasElement) {
    this.base = document.createElement('canvas');
    this.overlay = document.createElement('canvas');
    this.base.className = 'spec-base';
    this.overlay.className = 'spec-overlay';
    container.append(this.base, this.overlay);

    // `alpha: false` lets the compositor skip blending the opaque bottom layer.
    this.baseCtx = this.base.getContext('2d', { alpha: false })!;
    this.overlayCtx = this.overlay.getContext('2d')!;
    this.axis = axis;
    this.axisCtx = axis.getContext('2d')!;
    this.colorscale = colorscale;
    this.colorscaleCtx = colorscale.getContext('2d')!;
  }

  setData(data: SpectrogramData | null): void {
    this.data = data;
    this.baseDirty = true;
    this.drawnKey = '';
  }

  invalidate(): void {
    this.baseDirty = true;
  }

  /** Highest frequency the loaded audio can represent. */
  get nyquist(): number {
    return this.data ? this.data.sampleRate / 2 : 8000;
  }

  /** Resizes the backing stores to the element's box at the current DPR. */
  resize(cssWidth: number, cssHeight: number, axisWidth: number): void {
    // Cap DPR: a 3x retina buffer triples the app's hottest loop for a
    // spectrogram that already looks smooth at 2x.
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(cssWidth * this.dpr));
    const h = Math.max(1, Math.round(cssHeight * this.dpr));

    for (const c of [this.base, this.overlay]) {
      c.width = w;
      c.height = h;
      c.style.width = `${cssWidth}px`;
      c.style.height = `${cssHeight}px`;
    }

    this.image = this.baseCtx.createImageData(w, h);
    // Fill alpha once; the draw loop only ever writes RGB.
    const d = this.image.data;
    for (let i = 3; i < d.length; i += 4) d[i] = 255;

    this.axis.width = Math.max(1, Math.round(axisWidth * this.dpr));
    this.axis.height = h;
    this.axis.style.width = `${axisWidth}px`;
    this.axis.style.height = `${cssHeight}px`;

    // The colorscale's width comes from CSS (it sits in the fixed-width
    // contrast rail). Its height is deliberately less than the full column,
    // centred by `align-self: center`, so the end labels have somewhere to
    // sit that isn't the rail's own clipped edge.
    const colorscaleWidth = this.colorscale.clientWidth;
    const colorscaleCssHeight = Math.max(20, cssHeight - 40);
    this.colorscale.width = Math.max(1, Math.round(colorscaleWidth * this.dpr));
    this.colorscale.height = Math.max(1, Math.round(colorscaleCssHeight * this.dpr));
    this.colorscale.style.height = `${colorscaleCssHeight}px`;

    this.bandForRow = new Int32Array(h);
    this.viewport.width = cssWidth;
    this.baseDirty = true;
    // The old image is gone with the old backing store; nothing to scroll from.
    this.drawnKey = '';
  }

  /** Row -> mel band table. Cheap; rebuilt on every base draw. */
  private updateRowTable(): void {
    const data = this.data;
    if (!data) return;
    const h = this.bandForRow.length;
    const { minHz, maxHz } = this.settings;
    const bank = { bands: data.binCount, nyquist: data.sampleRate / 2 } as Parameters<
      typeof bandForHz
    >[1];

    const loMel = hertzToMel(minHz);
    const hiMel = hertzToMel(maxHz);
    const last = data.binCount - 1;

    for (let y = 0; y < h; y++) {
      // Row 0 is the top of the canvas and the highest frequency. Interpolating
      // in mel is what makes the axis mel-spaced.
      const mel = hiMel - ((hiMel - loMel) * y) / Math.max(1, h - 1);
      const band = Math.round(bandForHz(melToHertz(mel), bank));
      this.bandForRow[y] = band < 0 ? 0 : band > last ? last : band;
    }
  }

  /**
   * Everything except the view's start position that decides what a pixel is.
   * Two draws that agree on this differ only by a horizontal shift, which is
   * what makes the scroll path below safe. `columns` and `startSeconds` are in
   * here because the live scope mutates its buffer in place -- the object
   * identity never changes, so watching the object would miss new audio.
   */
  private contentKey(): string {
    const s = this.settings;
    const d = this.data;
    return [
      s.minHz,
      s.maxHz,
      s.minByte,
      s.maxByte,
      s.colormap,
      this.base.width,
      this.base.height,
      this.viewport.pixelsPerSecond,
      d ? `${d.columns}|${d.startSeconds}|${d.secondsPerColumn}|${d.binCount}` : 'empty',
    ].join('|');
  }

  drawBase(): void {
    if (!this.baseDirty) return;
    this.baseDirty = false;

    const { base, baseCtx, image, data } = this;
    if (!image) return;

    if (!data || data.columns === 0) {
      this.drawAxis();
      this.drawColorscale();
      baseCtx.fillStyle = '#0b1220';
      baseCtx.fillRect(0, 0, base.width, base.height);
      this.drawnKey = '';
      return;
    }

    const w = base.width;
    const h = base.height;
    const key = this.contentKey();
    const devicePixelsPerSecond = this.viewport.pixelsPerSecond * this.dpr;

    // Scroll path: the image already on the canvas is still correct, just in
    // the wrong place. Shifting it and filling in the strip that scrolled into
    // view costs a blit plus a handful of columns, where a full repaint is
    // width*height peak-holds -- the difference between comfortably inside a
    // frame budget and over it. The playhead moves every frame, so this is the
    // common case during playback and live capture.
    if (key === this.drawnKey) {
      // Device pixels the content must move right by. Integer, because a blit
      // cannot shift by a fraction of a pixel without resampling.
      const shift = Math.round((this.drawnStart - this.viewport.start) * devicePixelsPerSecond);
      if (shift === 0) return;

      if (Math.abs(shift) < w) {
        this.updateRowTable();
        baseCtx.drawImage(base, shift, 0);
        // The blit moved by whole pixels, so the image now starts at whatever
        // time that lands on rather than exactly at viewport.start. Tracking
        // the real value keeps the error under one pixel forever instead of
        // letting rounding accumulate over thousands of frames.
        this.drawnStart -= shift / devicePixelsPerSecond;

        const from = shift > 0 ? 0 : w + shift;
        const to = shift > 0 ? shift : w;
        this.renderColumns(from, to, this.drawnStart);
        // Only the new strip is uploaded; the rest of `image` is stale and
        // deliberately never read.
        baseCtx.putImageData(image, 0, 0, from, 0, to - from, h);
        return;
      }
    }

    // Full repaint: something other than the view's position changed, or the
    // view jumped further than its own width.
    this.drawAxis();
    this.drawColorscale();
    this.updateRowTable();
    this.renderColumns(0, w, this.viewport.start);
    baseCtx.putImageData(image, 0, 0);
    this.drawnKey = key;
    this.drawnStart = this.viewport.start;
  }

  /**
   * Renders device-pixel columns [xFrom, xTo) into the image buffer, with
   * `imageStart` as the time at x = 0. Split out from drawBase so the scroll
   * path can fill a strip on exactly the same terms as a full repaint.
   */
  private renderColumns(xFrom: number, xTo: number, imageStart: number): void {
    const { image, data } = this;
    if (!image || !data) return;

    const w = this.base.width;
    const h = this.base.height;
    const px = image.data;
    const lut = colormap(this.settings.colormap);
    const levels = levelTable(this.settings.minByte, this.settings.maxByte);
    const bins = data.bins;
    const binCount = data.binCount;
    const rows = this.bandForRow;

    const secondsPerPixel = 1 / (this.viewport.pixelsPerSecond * this.dpr);
    const columnsPerPixel = secondsPerPixel / data.secondsPerColumn;
    const stride = Math.max(1, Math.ceil(columnsPerPixel / MAX_COLUMNS_PER_PIXEL));
    const lastColumn = data.columns - 1;

    for (let x = xFrom; x < xTo; x++) {
      // Columns are indexed from the buffer's own start, which is 0 for a file
      // but walks forward for the live scope as it drops old history.
      const t = imageStart + x * secondsPerPixel - data.startSeconds;
      let c0 = Math.floor(t / data.secondsPerColumn);
      let c1 = Math.floor((t + secondsPerPixel) / data.secondsPerColumn);

      if (c0 > lastColumn || c1 < 0) {
        // Past the ends of the audio: leave the background showing.
        for (let y = 0; y < h; y++) {
          const o = (y * w + x) * 4;
          px[o] = 11;
          px[o + 1] = 18;
          px[o + 2] = 32;
        }
        continue;
      }

      if (c0 < 0) c0 = 0;
      if (c1 > lastColumn) c1 = lastColumn;
      if (c1 < c0) c1 = c0;

      for (let y = 0; y < h; y++) {
        const band = rows[y];
        // Peak-hold across the columns behind this pixel. Averaging would wash
        // out exactly the brief, narrow events the tool is about.
        let peak = 0;
        for (let c = c0; c <= c1; c += stride) {
          const v = bins[c * binCount + band];
          if (v > peak) peak = v;
        }
        const idx = levels[peak] * 3;
        const o = (y * w + x) * 4;
        px[o] = lut[idx];
        px[o + 1] = lut[idx + 1];
        px[o + 2] = lut[idx + 2];
      }
    }
  }

  /** Draws the playhead, the selection and the hover cursor. */
  drawOverlay(playhead: number | null, selection: Selection | null, hoverX: number | null): void {
    const ctx = this.overlayCtx;
    const w = this.overlay.width;
    const h = this.overlay.height;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const cssW = w / this.dpr;
    const cssH = h / this.dpr;

    if (selection) {
      const x0 = this.viewport.timeToX(selection.from);
      const x1 = this.viewport.timeToX(selection.to);
      ctx.fillStyle = 'rgba(226, 232, 240, 0.10)';
      ctx.fillRect(x0, 0, x1 - x0, cssH);
      ctx.strokeStyle = 'rgba(148, 197, 253, 0.9)';
      ctx.lineWidth = 1;
      for (const x of [x0, x1]) {
        ctx.beginPath();
        ctx.moveTo(Math.round(x) + 0.5, 0);
        ctx.lineTo(Math.round(x) + 0.5, cssH);
        ctx.stroke();
      }
    }

    if (hoverX !== null && hoverX >= 0 && hoverX <= cssW) {
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(hoverX) + 0.5, 0);
      ctx.lineTo(Math.round(hoverX) + 0.5, cssH);
      ctx.stroke();
    }

    if (playhead !== null) {
      const x = this.viewport.timeToX(playhead);
      if (x >= -1 && x <= cssW + 1) {
        ctx.strokeStyle = 'rgba(255,255,255,0.85)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, cssH);
        ctx.stroke();
      }
    }
  }

  /** Frequency labels in the left gutter, placed on the mel axis. */
  drawAxis(): void {
    const ctx = this.axisCtx;
    const w = this.axis.width / this.dpr;
    const h = this.axis.height / this.dpr;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = 'rgba(11,18,32,0.85)';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w - 0.5, 0);
    ctx.lineTo(w - 0.5, h);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const { minHz, maxHz } = this.settings;
    // Round frequencies rather than even spacing, since the axis is warped and
    // even spacing would land on unreadable numbers. Strictly between the
    // endpoints: those are shown by the editable Hz boxes docked over the
    // axis, not redrawn here.
    const candidates = [0, 100, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000];
    let lastY = -Infinity;
    for (const hz of candidates) {
      if (hz <= minHz || hz >= maxHz) continue;
      const y = this.hzToY(hz, h);
      if (y < 20 || y > h - 20) continue;
      // Skip labels that would collide near the compressed top of the axis.
      if (Math.abs(y - lastY) < 13) continue;
      lastY = y;
      ctx.fillText(hz >= 1000 ? `${hz / 1000}k` : `${hz}`, w - 6, y);
    }
  }

  /**
   * dB colour legend beside the contrast rail: the same colormap and level
   * window (`levelTable`) used for the spectrogram pixels, so the bar always
   * shows exactly what the image's colours mean. Redrawn whenever the
   * contrast sliders move, since minByte/maxByte are part of `contentKey()`.
   */
  drawColorscale(): void {
    const ctx = this.colorscaleCtx;
    const w = this.colorscale.width / this.dpr;
    const h = this.colorscale.height / this.dpr;
    if (w === 0 || h === 0) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const barW = Math.max(1, Math.min(10, w - 24));
    const lut = colormap(this.settings.colormap);
    const { minByte, maxByte } = this.settings;
    const levels = levelTable(minByte, maxByte);

    // Byte 255 at the top, to match "loud is up" on the spectrogram itself.
    for (let y = 0; y < h; y++) {
      const byte = Math.round(255 * (1 - y / Math.max(1, h - 1)));
      const idx = levels[byte] * 3;
      ctx.fillStyle = `rgb(${lut[idx]}, ${lut[idx + 1]}, ${lut[idx + 2]})`;
      ctx.fillRect(0, y, barW, 1);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, barW - 1, h - 1);

    // Inset from the very top/bottom edge so the end labels' glyphs aren't
    // clipped by the canvas boundary.
    const inset = 5;
    const dbAt = (byte: number) => DB_FLOOR + (byte / 255) * (DB_CEIL - DB_FLOOR);
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '9px system-ui, sans-serif';
    ctx.textAlign = 'left';
    const labelX = barW + 4;
    ctx.textBaseline = 'top';
    ctx.fillText(`${DB_CEIL}`, labelX, inset);
    ctx.textBaseline = 'middle';
    ctx.fillText(`${Math.round(dbAt(128))}`, labelX, h / 2);
    ctx.textBaseline = 'bottom';
    ctx.fillText(`${DB_FLOOR}`, labelX, h - inset);

    this.colorscale.title = `Colour scale, in dBFS (window: ${Math.round(dbAt(minByte))} to ${Math.round(dbAt(maxByte))})`;
  }

  /** Canvas y for a frequency, on the mel axis. */
  hzToY(hz: number, cssHeight: number): number {
    const loMel = hertzToMel(this.settings.minHz);
    const hiMel = hertzToMel(this.settings.maxHz);
    return ((hiMel - hertzToMel(hz)) / (hiMel - loMel)) * cssHeight;
  }
}
