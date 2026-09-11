/**
 * Composites a PNG of the spectrogram and buzzdetect panels for download.
 *
 * Built by hand rather than pulling in html2canvas: everything visual here is
 * already either a <canvas> (spectrogram, gutters, colorscale, activations) or
 * a handful of small DOM elements (time axis ticks, legend), so a real DOM
 * rasteriser is overkill. Each source's on-screen `getBoundingClientRect()` is
 * the layout authority -- positions are taken from it rather than recomputed
 * from --gutter/--rail, so the screenshot can't drift out of alignment with
 * what CLAUDE.md requires the three rows to share on screen.
 *
 * The contrast rail's draggable thumbs are left out: they're a control, not
 * part of the result, and reproducing <button> chrome by hand would be a lot
 * of code for something that means nothing in a static image. The colour
 * scale strip itself is kept, since it's the legend for the spectrogram's
 * colours.
 *
 * The gutter/colorscale labels are baked into their canvases by SpectrogramView
 * and ActivationView at a fixed, small font size that reads fine as part of the
 * live UI (where it sits among full-size on-screen text) but reads as an
 * afterthought once it's the only text in a standalone image. Rather than
 * duplicate their tick-placement logic here, `axisFontScale` on both views is
 * bumped for one synchronous redraw, captured, then put back -- nothing repaints
 * in between since JS is single-threaded, so the live gutter never visibly
 * changes size.
 */

const AXIS_FONT_SCALE = 1.8;

const BG = '#070b14';
const PANEL = '#0b1220';
const LINE = '#1e293b';
const TEXT = '#e2e8f0';
const MUTED = '#94a3b8';
const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';

const PAD = 14;
const FOOTER_H = 24;

// Same ramp as .brand strong in style.css, entered at its purple mid-point.
const WORDMARK_STOPS: [number, string][] = [
  [0, '#7a2382'],
  [0.3, '#b5367a'],
  [0.58, '#e55064'],
  [0.8, '#fb8761'],
  [1, '#fec287'],
];

interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

function rectOf(el: Element): Rect {
  const r = el.getBoundingClientRect();
  return { left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height };
}

/** The subset of SpectrogramView/ActivationView this module needs. */
interface AxisSource {
  axisFontScale: number;
  drawAxis(): void;
}
interface SpecAxisSource extends AxisSource {
  drawColorscale(): void;
}

export interface ScreenshotSources {
  spec: SpecAxisSource;
  acts: AxisSource;
  specAxis: HTMLCanvasElement;
  specBase: HTMLCanvasElement;
  specOverlay: HTMLCanvasElement;
  colorscale: HTMLCanvasElement;
  timeaxis: HTMLElement;
  actsAxis: HTMLCanvasElement;
  actsCanvas: HTMLCanvasElement;
  legend: HTMLElement;
  filename: string;
  durationSeconds: number;
}

/** Renders the current view to a canvas at up to 2x device pixel ratio. */
export function renderScreenshot(src: ScreenshotSources): HTMLCanvasElement {
  const { specAxis, specBase, colorscale, actsAxis, actsCanvas } = src;

  src.spec.axisFontScale = AXIS_FONT_SCALE;
  src.acts.axisFontScale = AXIS_FONT_SCALE;
  src.spec.drawAxis();
  src.spec.drawColorscale();
  src.acts.drawAxis();

  const specTop = rectOf(specAxis).top;
  const specLeft = rectOf(specAxis).left;
  const railRight = rectOf(colorscale).right + PAD;
  const actsBottom = rectOf(actsCanvas).bottom;
  const actsAxisLeft = rectOf(actsAxis).left;

  const left = Math.min(specLeft, actsAxisLeft) - PAD;
  const right = railRight;
  const top = specTop - PAD;
  const bottom = actsBottom + PAD;

  const width = Math.max(1, Math.round(right - left));
  const bodyHeight = Math.max(1, Math.round(bottom - top));
  const height = bodyHeight + FOOTER_H;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);

  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, width, height);

  const bodyTop = 0;

  const toLocal = (r: Rect) => ({
    x: r.left - left,
    y: r.top - top + bodyTop,
    w: r.width,
    h: r.height,
  });

  const drawCanvas = (source: HTMLCanvasElement) => {
    const loc = toLocal(rectOf(source));
    if (loc.w <= 0 || loc.h <= 0) return;
    ctx.drawImage(source, loc.x, loc.y, loc.w, loc.h);
  };

  // Spectrogram row.
  drawCanvas(specAxis);
  drawCanvas(specBase);
  drawCanvas(src.specOverlay);
  drawCanvas(colorscale);

  // Time axis row: background band, then ticks + labels copied from the DOM.
  const axisRect = toLocal(rectOf(src.timeaxis));
  ctx.fillStyle = PANEL;
  ctx.fillRect(0, axisRect.y, width, axisRect.h);
  ctx.strokeStyle = LINE;
  ctx.beginPath();
  ctx.moveTo(0, axisRect.y);
  ctx.lineTo(width, axisRect.y);
  ctx.moveTo(0, axisRect.y + axisRect.h);
  ctx.lineTo(width, axisRect.y + axisRect.h);
  ctx.stroke();

  ctx.font = `16px ${MONO}`;
  ctx.fillStyle = MUTED;
  ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const span of Array.from(src.timeaxis.children) as HTMLElement[]) {
    const r = toLocal(rectOf(span));
    const x = r.x + r.w / 2;
    ctx.beginPath();
    ctx.moveTo(x, axisRect.y);
    ctx.lineTo(x, axisRect.y + 4);
    ctx.stroke();
    ctx.fillText(span.textContent ?? '', x, axisRect.y + 5);
  }

  // Activations row.
  drawCanvas(actsAxis);
  drawCanvas(actsCanvas);

  // Both canvases are captured now; put the live gutters back to normal size.
  src.spec.axisFontScale = 1;
  src.acts.axisFontScale = 1;
  src.spec.drawAxis();
  src.spec.drawColorscale();
  src.acts.drawAxis();

  // Legend, redrawn from the live DOM so it matches whatever is currently shown.
  const legendItems = Array.from(src.legend.children) as HTMLElement[];
  if (legendItems.length > 0) {
    const legendRect = toLocal(rectOf(src.legend));
    ctx.fillStyle = 'rgba(11, 18, 32, 0.72)';
    const rr = (x: number, y: number, w: number, h: number, radius: number) => {
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.arcTo(x + w, y, x + w, y + h, radius);
      ctx.arcTo(x + w, y + h, x, y + h, radius);
      ctx.arcTo(x, y + h, x, y, radius);
      ctx.arcTo(x, y, x + w, y, radius);
      ctx.closePath();
    };
    rr(legendRect.x, legendRect.y, legendRect.w, legendRect.h, 6);
    ctx.fill();

    ctx.font = `14px ${MONO}`;
    ctx.textBaseline = 'middle';
    for (const item of legendItems) {
      const r = toLocal(rectOf(item));
      const pip = item.querySelector('.pip') as HTMLElement | null;
      const name = item.querySelector('.legend-name') as HTMLElement | null;
      const value = item.querySelector('.legend-value') as HTMLElement | null;
      const color = item.style.color || TEXT;
      const dimmed = item.classList.contains('dimmed');
      ctx.globalAlpha = dimmed ? 0.35 : 1;

      let x = r.x;
      const cy = r.y + r.h / 2;
      if (pip) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x + 5, cy, 5, 0, Math.PI * 2);
        ctx.fill();
        x += 16;
      }
      if (name) {
        ctx.fillStyle = color;
        ctx.textAlign = 'left';
        ctx.fillText(name.textContent ?? '', x, cy);
      }
      if (value) {
        ctx.fillStyle = 'rgba(226, 232, 240, 0.85)';
        ctx.textAlign = 'right';
        ctx.fillText(value.textContent ?? '', r.x + r.w, cy);
      }
      ctx.globalAlpha = 1;
    }
  }

  // Footer banner: attribution, site link and filename on one line, all in the
  // same muted weight; the timestamp sits across from them.
  const footerTop = height - FOOTER_H;
  ctx.fillStyle = PANEL;
  ctx.fillRect(0, footerTop, width, FOOTER_H);
  ctx.strokeStyle = LINE;
  ctx.beginPath();
  ctx.moveTo(0, footerTop);
  ctx.lineTo(width, footerTop);
  ctx.stroke();

  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  const cy = footerTop + FOOTER_H / 2;

  const wordmark = 'buzzdetect live!';
  ctx.font = `bold 10px ${MONO}`;
  const wordmarkWidth = ctx.measureText(wordmark).width;
  const gradient = ctx.createLinearGradient(PAD, 0, PAD + wordmarkWidth, 0);
  for (const [stop, color] of WORDMARK_STOPS) gradient.addColorStop(stop, color);
  ctx.fillStyle = gradient;
  ctx.fillText(wordmark, PAD, cy);

  ctx.font = `10px ${MONO}`;
  ctx.fillStyle = MUTED;
  const rest = ' — lukehearon.com/buzzdetect-live';
  ctx.fillText(rest, PAD + wordmarkWidth, cy);
  let x = PAD + wordmarkWidth + ctx.measureText(rest).width;
  if (src.filename) {
    const withFilename = ` — ${src.filename}`;
    ctx.fillText(withFilename, x, cy);
    x += ctx.measureText(withFilename).width;
  }
  if (src.durationSeconds > 0) {
    ctx.fillText(` (${src.durationSeconds.toFixed(0)}s)`, x, cy);
  }
  ctx.textAlign = 'right';
  ctx.fillText(new Date().toLocaleString(), width - PAD, cy);

  return canvas;
}

/** Renders and triggers a download as a PNG. */
export function downloadScreenshot(src: ScreenshotSources): void {
  const canvas = renderScreenshot(src);
  // Browsers de-duplicate a repeat name themselves (e.g. "testbuzz (1).png"),
  // so there's no need to stamp one in ourselves.
  const base = src.filename.replace(/\.[^.]+$/, '') || 'buzzdetect';
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${base}.png`;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}
