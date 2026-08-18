/**
 * Wiring: file or microphone in, worker out, two aligned canvases and a
 * transport.
 *
 * File mode and live mode share the same spectrogram and the same activation
 * panel; the only differences are where the data comes from and whether the
 * viewport follows the right edge. Keeping one set of views means a live
 * reading and a recorded one are drawn by the same code, on the same axes, in
 * the same units -- and that a live session becomes an ordinary file the moment
 * capture stops.
 *
 * Two rules hold throughout:
 *   - nothing expensive runs on this thread (see workers/analysis.worker.ts);
 *   - all drawing happens in one requestAnimationFrame loop with dirty flags,
 *     so a burst of events costs one repaint per frame rather than one each.
 */

import { decodeForAnalysis, encodeWav, LONG_FILE_SECONDS } from './audio/decode';
import { Player, sliderToGain, gainToSlider } from './audio/player';
import { SpectrogramView } from './ui/spectrogramView';
import { ActivationStore, ActivationView } from './ui/activationView';
import { SERIES, enabledSeries, type Series } from './ui/series';
import { formatTimeShort, normalizeSelection, type Selection } from './ui/viewport';
import { LiveSession, LIVE_STORE_PATCHES } from './live/liveSession';
import { createDualRange } from './ui/rangeSlider';
import type { WorkerResponse } from './workers/analysis.worker';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const els = {
  file: $<HTMLInputElement>('file'),
  mic: $<HTMLButtonElement>('mic'),
  openFile: $('open-file'),
  help: $<HTMLButtonElement>('help'),
  helpDialog: $<HTMLDialogElement>('help-dialog'),
  micLabel: $('mic-label'),
  liveBadge: $('live-badge'),
  meterFill: $('meter-fill'),
  status: $('status'),
  filename: $('filename'),
  minHz: $<HTMLInputElement>('min-hz'),
  maxHz: $<HTMLInputElement>('max-hz'),
  contrast: $('contrast'),
  colorscale: $<HTMLCanvasElement>('colorscale'),
  spec: $('spec'),
  specAxis: $<HTMLCanvasElement>('spec-axis'),
  timeaxis: $('timeaxis'),
  acts: $('acts'),
  actsAxis: $<HTMLCanvasElement>('acts-axis'),
  detectionCount: $('detection-count'),
  settings: $<HTMLButtonElement>('settings'),
  settingsDialog: $<HTMLDialogElement>('settings-dialog'),
  classList: $('class-list'),
  legend: $('legend'),
  toStart: $<HTMLButtonElement>('to-start'),
  play: $<HTMLButtonElement>('play'),
  playIcon: document.getElementById('play-icon') as unknown as SVGUseElement,
  time: $<HTMLInputElement>('time'),
  volume: $<HTMLInputElement>('volume'),
  volumeOut: $<HTMLOutputElement>('volume-out'),
  micGainControl: $('mic-gain-control'),
  micGain: $<HTMLInputElement>('mic-gain'),
  micGainOut: $<HTMLOutputElement>('mic-gain-out'),
};

const spec = new SpectrogramView(els.spec, els.specAxis, els.colorscale);
const acts = new ActivationView(els.acts, els.actsAxis);
const store = new ActivationStore();
const player = new Player();

/**
 * The classes currently plotted. All thirteen are always computed -- the model
 * emits them in one pass whatever is shown -- so switching one on costs a
 * redraw and nothing else.
 */
let shown: Series[] = enabledSeries();

let mediaUrl: string | null = null;
let hasAudio = false;
let live: LiveSession | null = null;
let selection: Selection | null = null;
let hoverX: number | null = null;
let overlayDirty = true;
let actsDirty = true;

const worker = new Worker(new URL('./workers/analysis.worker.ts', import.meta.url), {
  type: 'module',
});

function setStatus(text: string, kind: '' | 'busy' | 'error' = ''): void {
  els.status.textContent = text;
  els.status.className = `status ${kind}`.trim();
}

// ---------------------------------------------------------------- worker ---

worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'ready':
      // The header starts on "Loading model…"; this is when it stops being true.
      setStatus('');
      break;

    case 'spectrogram':
      if (!live) {
        spec.setData(msg.data);
        actsDirty = true;
      }
      break;

    case 'progress':
      if (msg.stage === 'inference') {
        setStatus(`Analysing… ${Math.round(msg.fraction * 100)}%`, 'busy');
      }
      break;

    case 'results':
      if (store.total !== msg.totalPatches) store.reset(msg.totalPatches, msg.patchHopSeconds);
      store.put(msg.firstPatch, msg.activations);
      autoRange();
      actsDirty = true;
      break;

    case 'done': {
      const seconds = msg.elapsedMs / 1000;
      const rate = spec.viewport.duration / seconds;
      setStatus(
        `Analyzed ${spec.viewport.duration.toFixed(0)} seconds in ${seconds.toFixed(1)}s ` +
          `(${rate.toFixed(0)}× realtime)`,
      );
      updateDetectionCount();
      break;
    }

    case 'live':
      store.put(msg.patchIndex, msg.activations);
      autoRange();
      actsDirty = true;
      updateDetectionCount();
      break;

    case 'error':
      setStatus(msg.message, 'error');
      break;
  }
};

/**
 * Keeps the value axis around the data.
 *
 * Activations are unbounded, so a fixed axis either clips or wastes most of its
 * height. The range only grows during a session, so the plot does not rescale
 * under the cursor every time a loud frame arrives.
 */
function autoRange(): void {
  // Over the shown classes only: a class nobody is looking at should not be
  // able to flatten everything else against the axis.
  for (const s of shown) {
    const { min, max } = store.range(s.index);
    acts.min = Math.min(acts.min, Math.floor(min * 2) / 2);
    acts.max = Math.max(acts.max, Math.ceil(max * 2) / 2);
  }
}

// ------------------------------------------------------------------ file ---

els.file.addEventListener('change', async () => {
  const file = els.file.files?.[0];
  if (file) await loadFile(file);
  els.file.value = '';
});

document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', async (e) => {
  e.preventDefault();
  const file = e.dataTransfer?.files?.[0];
  if (file) await loadFile(file);
});

/**
 * Draws attention to the two source buttons while there is nothing to look at.
 * They are the only useful controls in an empty session, and the transport
 * beside them is disabled, so an untouched page otherwise gives no clue where
 * to start.
 */
function updateSourcePrompt(): void {
  const empty = !hasAudio && !live;
  els.mic.classList.toggle('needs-source', empty);
  els.openFile.classList.toggle('needs-source', empty);
}

/** Clears whatever is loaded, in either mode. */
function resetSession(): void {
  if (mediaUrl) URL.revokeObjectURL(mediaUrl);
  mediaUrl = null;
  hasAudio = false;
  store.reset(0, 0.96);
  spec.setData(null);
  selection = null;
  player.pause();
  acts.min = -4;
  acts.max = 2;
  els.play.disabled = true;
  els.toStart.disabled = true;
  actsDirty = true;
  overlayDirty = true;
  updateSourcePrompt();
}

async function loadFile(file: File): Promise<void> {
  try {
    await stopLive();
    setStatus('Decoding…', 'busy');
    resetSession();
    els.filename.textContent = file.name;

    const decoded = await decodeForAnalysis(file);
    if (decoded.peak < 1e-4) {
      setStatus('Decoded audio is silent; the browser may not support this file', 'error');
    }
    if (decoded.duration > LONG_FILE_SECONDS) {
      setStatus(`Long file: ${(decoded.duration / 60).toFixed(0)} min`, 'busy');
    }

    mediaUrl = decoded.url;
    player.load(decoded.url);
    hasAudio = true;
    updateSourcePrompt();
    spec.viewport.duration = decoded.duration;
    spec.viewport.fit();
    els.play.disabled = false;
    els.toStart.disabled = false;
    spec.invalidate();
    actsDirty = true;

    setStatus('Analysing…', 'busy');
    // Both arrays are transferred, so nothing is copied; neither may be read
    // after this point.
    const { wav, display } = decoded;
    worker.postMessage(
      { type: 'analyze', wav, display, displayRate: decoded.displayRate, framehopProp: 1 },
      display.buffer === wav.buffer ? [wav.buffer] : [wav.buffer, display.buffer],
    );
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err), 'error');
  }
}

// --------------------------------------------------------- interactions ---

/**
 * How long the pointer must be held before a drag counts as a selection.
 *
 * Without it, the few pixels of travel in an ordinary click land as a tiny
 * selection, and clicking to seek keeps leaving stray ranges behind. The anchor
 * is still the point where the button went down, so a deliberate press-and-drag
 * selects from where it started even though it only became a selection part-way
 * through -- what you get is what you aimed at, not what was left after the
 * delay elapsed.
 */
const SELECTION_HOLD_MS = 300;

/** Travel, in pixels, past which a touch drag is a pan and not a tap. */
const TAP_SLOP_PX = 8;

let dragging: {
  startTime: number;
  startedAt: number;
  selecting: boolean;
  onActs: boolean;
  /**
   * Touch only: one finger drags the timeline rather than selecting on it.
   * `touch-action: none` means the browser will not pan these panels for us,
   * and pinching alone leaves no way to move sideways. Selection stays a
   * mouse gesture; on touch a frame is still selected by tapping it.
   */
  panning: boolean;
  lastX: number;
  moved: boolean;
} | null = null;

/** A drag becomes a selection once it is both old enough and long enough. */
function selectionBegun(d: NonNullable<typeof dragging>, t: number): boolean {
  return performance.now() - d.startedAt >= SELECTION_HOLD_MS && Math.abs(t - d.startTime) > 0.002;
}

for (const area of [els.spec, els.acts]) {
  const onActs = area === els.acts;

  /**
   * Touch points currently down on this panel, by id, as x within the panel.
   *
   * Two of them is a pinch: the scale follows the ratio of their separation and
   * the view follows their midpoint, so zoom and pan come out of one gesture --
   * which is what a pinch on a timeline is expected to do. A pinch is anchored
   * afresh on every move rather than against where the fingers started, so the
   * midpoint sliding sideways pans by exactly that much.
   */
  const pointers = new Map<number, number>();
  let pinch: { distance: number; x: number } | null = null;

  /** Separation and midpoint of the first two pointers down, or null. */
  function spread(): { distance: number; x: number } | null {
    if (pointers.size < 2) return null;
    const [a, b] = [...pointers.values()];
    return { distance: Math.abs(a - b), x: (a + b) / 2 };
  }

  function viewChanged(): void {
    if (live) following = spec.viewport.end >= spec.viewport.duration - 0.5;
    spec.invalidate();
    actsDirty = true;
    overlayDirty = true;
  }

  area.addEventListener('pointerdown', (e) => {
    if (!hasData()) return;
    area.setPointerCapture(e.pointerId);
    const x = offsetX(e, area);
    pointers.set(e.pointerId, x);

    const second = spread();
    if (second) {
      // A second finger converts whatever the first was doing into a pinch,
      // including undoing a selection it had started to drag out.
      if (dragging?.selecting) selection = null;
      dragging = null;
      pinch = second;
      hoverX = null;
      actsDirty = true;
      overlayDirty = true;
      return;
    }

    dragging = {
      startTime: spec.viewport.xToTime(x),
      startedAt: performance.now(),
      selecting: false,
      onActs,
      panning: e.pointerType === 'touch',
      lastX: x,
      moved: false,
    };
  });

  area.addEventListener('pointermove', (e) => {
    const x = offsetX(e, area);
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, x);

    // A finger leaves no cursor, so it should leave no hover line either.
    hoverX = e.pointerType === 'touch' ? null : x;
    overlayDirty = true;

    if (pinch) {
      const now = spread();
      if (!now || now.distance <= 0) return;
      const vp = spec.viewport;
      vp.panBy((pinch.x - now.x) / vp.pixelsPerSecond);
      vp.setScaleAt(
        now.x,
        vp.pixelsPerSecond * (now.distance / pinch.distance),
        minPixelsPerSecond(),
      );
      pinch = now;
      viewChanged();
      return;
    }

    const t = spec.viewport.xToTime(x);

    if (dragging) {
      if (dragging.panning) {
        if (Math.abs(x - dragging.lastX) > 0) {
          const vp = spec.viewport;
          if (!dragging.moved && Math.abs(vp.timeToX(dragging.startTime) - x) > TAP_SLOP_PX) {
            dragging.moved = true;
          }
          vp.panBy((dragging.lastX - x) / vp.pixelsPerSecond);
          dragging.lastX = x;
          viewChanged();
        }
        return;
      }
      if (selectionBegun(dragging, t)) dragging.selecting = true;
      if (dragging.selecting) {
        selection = normalizeSelection({ from: dragging.startTime, to: t });
        actsDirty = true;
      }
    }
  });

  function endPointer(e: PointerEvent): void {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = null;
    if (e.pointerType === 'touch') {
      hoverX = null;
      overlayDirty = true;
    }
  }

  area.addEventListener('pointerup', (e) => {
    endPointer(e);
    if (!dragging) return;
    // A pan that went anywhere is not also a seek.
    if (dragging.panning && dragging.moved) {
      dragging = null;
      return;
    }
    // A tap has not moved, so the press point is the point meant -- and after a
    // pan the pixel under the finger no longer means what it did on the way down.
    const t = dragging.panning ? dragging.startTime : spec.viewport.xToTime(offsetX(e, area));
    // Judged again here: a drag that came to rest before the hold elapsed
    // sends no further pointermove, so this is the only place left to notice
    // that it has since become one.
    if (selectionBegun(dragging, t)) {
      dragging.selecting = true;
      selection = normalizeSelection({ from: dragging.startTime, to: t });
    }

    if (!dragging.selecting) {
      if (dragging.onActs && store.filled > 0) {
        // Clicking a frame selects that frame's audio, so the extent the model
        // scored is the extent you hear.
        const p = store.patchAt(t);
        if (p >= 0 && p < store.filled) {
          selection = { from: store.timeOf(p), to: store.timeOf(p) + store.patchHopSeconds };
          player.currentTime = selection.from;
        }
      } else {
        // A click on the spectrogram seeks, and clears any selection.
        selection = null;
        if (!live) player.currentTime = Math.max(0, t);
      }
    }

    dragging = null;
    overlayDirty = true;
    actsDirty = true;
  });

  area.addEventListener('pointercancel', (e) => {
    endPointer(e);
    dragging = null;
  });

  area.addEventListener('pointerleave', () => {
    hoverX = null;
    overlayDirty = true;
  });

  area.addEventListener(
    'wheel',
    (e) => {
      if (!hasData()) return;
      e.preventDefault();

      // Scrolling pans; shift+scroll zooms. A trackpad's pinch gesture also
      // arrives as a wheel event with ctrlKey set, so it zooms too regardless
      // of shift. A trackpad's horizontal swipe arrives as deltaX -- treat
      // the dominant axis as the pan direction, so a sideways swipe pans
      // sideways and a vertical scroll pans along time.
      if (e.shiftKey || e.ctrlKey) {
        const factor = Math.exp(-Math.max(-100, Math.min(100, e.deltaY)) * 0.004);
        spec.viewport.zoomAt(offsetX(e, area), factor, minPixelsPerSecond());
      } else {
        const horizontal = Math.abs(e.deltaX) > Math.abs(e.deltaY);
        const delta = horizontal ? e.deltaX : e.deltaY;
        spec.viewport.panBy(delta / spec.viewport.pixelsPerSecond);
        // Panning away from the right edge in live mode stops the auto-follow,
        // so you can look at something while capture continues.
        if (live) following = spec.viewport.end >= spec.viewport.duration - 0.5;
      }
      spec.invalidate();
      actsDirty = true;
      overlayDirty = true;
    },
    { passive: false },
  );
}

function hasData(): boolean {
  return hasAudio || live !== null;
}

function minPixelsPerSecond(): number {
  const d = spec.viewport.duration;
  return d > 0 ? spec.viewport.width / d : 1;
}

document.addEventListener('keydown', (e) => {
  const target = e.target as HTMLElement | null;
  if (target?.tagName === 'INPUT' && (target as HTMLInputElement).type !== 'range') return;
  // The dialog owns the keyboard while it is up: Escape should close it, not
  // also clear the selection behind it.
  if (els.helpDialog.open || els.settingsDialog.open) return;

  switch (e.key) {
    case ' ':
      if (!hasAudio) break;
      e.preventDefault();
      void player.toggle();
      break;
    case 'ArrowLeft':
      player.currentTime -= e.shiftKey ? 5 : 1;
      break;
    case 'ArrowRight':
      player.currentTime += e.shiftKey ? 5 : 1;
      break;
    case 'Escape':
      selection = null;
      actsDirty = true;
      overlayDirty = true;
      break;
  }
});

function offsetX(e: PointerEvent | WheelEvent, area: HTMLElement): number {
  return e.clientX - area.getBoundingClientRect().left;
}

// -------------------------------------------------------------- controls ---

// ------------------------------------------------------- class selection ---

/**
 * Builds the picker once, from the model's class list.
 *
 * Every class gets a row whether or not it is plotted, so turning one on is one
 * click rather than a search, and its threshold sits in the row it belongs to
 * rather than in a second list that has to be matched up by name.
 */
function buildClassList(): void {
  const frag = document.createDocumentFragment();
  for (const s of SERIES) {
    const row = document.createElement('label');
    row.className = 'class-row';

    const check = document.createElement('input');
    check.type = 'checkbox';
    check.checked = s.enabled;

    const pip = document.createElement('span');
    pip.className = 'pip';
    pip.style.background = s.color;

    const name = document.createElement('span');
    name.className = 'class-name';
    name.textContent = s.name;
    name.style.color = s.color;

    const threshold = document.createElement('input');
    threshold.type = 'number';
    threshold.className = 'num';
    threshold.step = '0.05';
    threshold.value = s.threshold.toFixed(2);
    threshold.title = `Threshold for ${s.name}`;
    // Inside a <label>: without this, clicking the field toggles the checkbox.
    threshold.addEventListener('click', (e) => e.stopPropagation());

    check.addEventListener('change', () => {
      s.enabled = check.checked;
      seriesChanged();
    });
    threshold.addEventListener('change', () => {
      const v = Number(threshold.value);
      if (!Number.isFinite(v)) {
        threshold.value = s.threshold.toFixed(2);
        return;
      }
      s.threshold = v;
      threshold.value = v.toFixed(2);
      actsDirty = true;
      updateDetectionCount();
    });

    row.append(check, pip, name, threshold);
    frag.append(row);
  }
  els.classList.replaceChildren(frag);
}

/** Legend: the plotted classes, over the plot's top-right corner. */
function drawLegend(): void {
  const frag = document.createDocumentFragment();
  for (const s of shown) {
    const item = document.createElement('span');
    item.className = 'legend-item';
    item.style.color = s.color;
    const pip = document.createElement('span');
    pip.className = 'pip';
    pip.style.background = s.color;
    item.append(pip, s.name);
    frag.append(item);
  }
  els.legend.replaceChildren(frag);
}

function seriesChanged(): void {
  shown = enabledSeries();
  autoRange();
  drawLegend();
  updateDetectionCount();
  actsDirty = true;
}

els.settings.addEventListener('click', () => els.settingsDialog.showModal());
buildClassList();

for (const el of [els.minHz, els.maxHz]) {
  el.addEventListener('change', () => {
    const lo = Math.max(0, Number(els.minHz.value) || 0);
    const hi = Math.min(spec.nyquist, Number(els.maxHz.value) || spec.nyquist);
    if (hi - lo < 100) return;
    spec.settings.minHz = lo;
    spec.settings.maxHz = hi;
    els.minHz.value = String(lo);
    els.maxHz.value = String(hi);
    // Band edges span the whole file, so a range change is a row lookup --
    // no recompute.
    spec.invalidate();
  });
}

// Contrast: one control for a window, rather than two sliders that can be
// dragged past each other. The gap is what the old code enforced by hand with
// a Math.min on every input.
els.contrast.append(
  createDualRange({
    min: 0,
    max: 255,
    step: 1,
    gap: 1,
    values: [spec.settings.minByte, spec.settings.maxByte],
    label: 'Contrast',
    onInput: (lowByte, highByte) => {
      spec.settings.minByte = lowByte;
      spec.settings.maxByte = highByte;
      spec.invalidate();
    },
  }).element,
);

/**
 * Accepts both of the forms the field can show and a person can mean: raw
 * seconds ("93.5"), MM:SS.SS as displayed, and H:MM:SS for a long recording.
 * Returns null rather than NaN for anything else, so a typo is refused instead
 * of seeking somewhere arbitrary.
 */
function parseTimestamp(text: string): number | null {
  const parts = text.trim().split(':');
  if (parts.length > 3) return null;
  let seconds = 0;
  for (const part of parts) {
    if (!/^\d+(\.\d+)?$/.test(part)) return null;
    seconds = seconds * 60 + Number(part);
  }
  return Number.isFinite(seconds) ? seconds : null;
}

/** Writes the readout, unless it is being typed into. */
function showTime(seconds: number): void {
  if (document.activeElement === els.time) return;
  els.time.value = formatTimeShort(seconds);
}

function commitTime(): boolean {
  const seconds = parseTimestamp(els.time.value);
  if (seconds === null) {
    els.time.classList.add('invalid');
    return false;
  }
  els.time.classList.remove('invalid');
  const vp = spec.viewport;
  player.currentTime = Math.max(0, Math.min(seconds, vp.duration));
  // Centre the view the way playback does, so a seek off-screen brings its
  // destination into view instead of leaving the playhead somewhere unseen.
  vp.start = player.currentTime - vp.visibleSeconds / 2;
  vp.clamp();
  spec.invalidate();
  actsDirty = true;
  overlayDirty = true;
  return true;
}

els.time.addEventListener('focus', () => els.time.select());
els.time.addEventListener('input', () => els.time.classList.remove('invalid'));
els.time.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    if (commitTime()) els.time.blur();
  } else if (e.key === 'Escape') {
    e.preventDefault();
    els.time.classList.remove('invalid');
    els.time.blur();
  }
});
els.time.addEventListener('blur', () => {
  // Leaving the field is not a way to enter a bad value: commit what parses,
  // and otherwise fall back to where the playhead actually is.
  if (!commitTime()) {
    els.time.classList.remove('invalid');
    els.time.value = formatTimeShort(player.currentTime);
  }
});

els.help.addEventListener('click', () => els.helpDialog.showModal());

els.play.addEventListener('click', () => void player.toggle());
els.toStart.addEventListener('click', () => {
  player.toStart();
  overlayDirty = true;
});

player.el.addEventListener('play', () => els.playIcon.setAttribute('href', '#i-pause'));
for (const ev of ['pause', 'ended']) {
  player.el.addEventListener(ev, () => els.playIcon.setAttribute('href', '#i-play'));
}
player.setSelection(() => selection);

els.volume.addEventListener('input', () => {
  const gain = sliderToGain(Number(els.volume.value));
  player.volume = gain;
  els.volumeOut.textContent = `${gain.toFixed(1)}×`;
});
els.volume.value = String(gainToSlider(1));

els.micGain.addEventListener('input', () => {
  const gain = Number(els.micGain.value);
  els.micGainOut.textContent = `${gain.toFixed(2)}×`;
  if (live) live.inputGain = gain;
});

function updateDetectionCount(): void {
  if (store.filled === 0 || shown.length === 0) {
    els.detectionCount.textContent = '';
    return;
  }
  // Per class, since each is counted against its own threshold; a total across
  // classes would not mean anything.
  const parts = shown.map((s) => `${s.name} ${store.countDetections(s.index, s.threshold)}`);
  els.detectionCount.textContent = `${parts.join(' · ')} — of ${store.filled} frames`;
}

// ------------------------------------------------------------- rendering ---

function resize(): void {
  const gutter = els.specAxis.clientWidth;
  spec.resize(els.spec.clientWidth, els.spec.clientHeight, gutter);
  acts.resize(els.acts.clientWidth, els.acts.clientHeight, gutter);
  spec.viewport.clamp();
  spec.invalidate();
  actsDirty = true;
  overlayDirty = true;
  drawTimeAxis();
}

new ResizeObserver(resize).observe(document.body);

/** Time-axis ticks, as DOM so the labels stay crisp. */
function drawTimeAxis(): void {
  const vp = spec.viewport;
  if (vp.duration === 0 || vp.width === 0) {
    els.timeaxis.replaceChildren();
    return;
  }

  // Aim for a tick every ~90 px, snapped to a round interval.
  const target = 90 / vp.pixelsPerSecond;
  const steps = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
  const step = steps.find((s) => s >= target) ?? steps[steps.length - 1];

  const frag = document.createDocumentFragment();
  const first = Math.ceil(vp.start / step) * step;
  for (let t = first; t <= vp.end; t += step) {
    const span = document.createElement('span');
    span.style.left = `${vp.timeToX(t)}px`;
    span.textContent = step < 1 ? t.toFixed(2) : String(Math.round(t));
    frag.append(span);
  }
  els.timeaxis.replaceChildren(frag);
}

let lastViewKey = '';
/** Playhead position as last drawn, in whole pixels; null when there is none. */
let lastHeadX: number | null = null;
/** Live only: whether the viewport is pinned to the right edge. */
let following = true;

function frame(): void {
  const vp = spec.viewport;

  if (live?.running) {
    vp.duration = live.elapsed;
    if (following) {
      // Keep "now" at the right edge; older audio pans off to the left.
      vp.start = Math.max(0, vp.duration - vp.visibleSeconds);
    }
    spec.invalidate();
    actsDirty = true;
    els.meterFill.style.width = `${Math.min(100, live.level * 140)}%`;
    showTime(vp.duration);
  }

  if (player.playing) {
    // Locked to the playhead: it sits in the middle and the audio moves past
    // it, so the view never jumps and never has to be chased.
    //
    // Advanced every frame. This used to be throttled because scrolling meant a
    // full repaint; SpectrogramView now scrolls by blitting what it already
    // drew, so the display simply runs at whatever rate the machine offers.
    vp.start = player.currentTime - vp.visibleSeconds / 2;
    vp.clamp();
    showTime(player.currentTime);
    overlayDirty = true;
  }

  const key = `${vp.start}|${vp.pixelsPerSecond}|${vp.width}`;
  if (key !== lastViewKey) {
    lastViewKey = key;
    drawTimeAxis();
    // The spectrogram scrolls with the view, so it is dirty whenever the view
    // is -- forgetting this is what pins it in place while everything else
    // moves.
    spec.invalidate();
    actsDirty = true;
    overlayDirty = true;
  }

  // The playhead moves on its own, without the view moving with it: whenever
  // playback does not scroll the view -- zoomed out far enough that the whole
  // media fits, or clamped at either end -- the view key is unchanged, so this
  // is the only thing that marks the panels dirty. The activation panel draws
  // its playhead into the same canvas as its data, so missing this leaves its
  // line parked where it was last painted while the spectrogram's own overlay,
  // redrawn every frame during playback, goes on advancing.
  const head = playhead();
  const headX = head === null ? null : Math.round(vp.timeToX(head));
  if (headX !== lastHeadX) {
    lastHeadX = headX;
    actsDirty = true;
    overlayDirty = true;
  }

  spec.drawBase();

  if (actsDirty) {
    actsDirty = false;
    acts.draw(store, shown, vp, head, selection);
  }

  if (overlayDirty) {
    overlayDirty = false;
    spec.drawOverlay(head, selection, hoverX);
  }

  requestAnimationFrame(frame);
}

function playhead(): number | null {
  if (live?.running) return following ? spec.viewport.duration : null;
  return hasAudio ? player.currentTime : null;
}

// ------------------------------------------------------------------ live ---

els.mic.addEventListener('click', async () => {
  if (live) {
    await stopLive();
    return;
  }

  try {
    resetSession();
    els.filename.textContent = 'Microphone';
    following = true;

    // The live store holds the same span of frames the spectrogram keeps, and
    // is compacted in step with it so both drop the same audio.
    store.reset(LIVE_STORE_PATCHES, 0.96);

    const session = new LiveSession({
      onSamples: (samples) => {
        // Copy: the worklet reuses its buffer and postMessage is async.
        const copy = samples.slice();
        worker.postMessage({ type: 'liveSamples', samples: copy }, [copy.buffer]);
      },
      onDrop: (patches) => store.dropOldest(patches),
    });

    live = session;
    updateSourcePrompt();
    spec.setData(session.data);
    spec.viewport.duration = 0;
    spec.viewport.pixelsPerSecond = 60;
    worker.postMessage({ type: 'liveStart' });

    await session.start();
    session.inputGain = Number(els.micGain.value);

    els.micLabel.textContent = 'Stop';
    els.mic.title = 'Stop recording';
    els.mic.classList.add('on');
    els.mic.querySelector('use')?.setAttribute('href', '#i-stop');
    els.liveBadge.hidden = false;
    els.micGainControl.hidden = false;
    setStatus('Listening…', 'busy');
  } catch (err) {
    setStatus(
      err instanceof Error ? `Microphone unavailable: ${err.message}` : String(err),
      'error',
    );
    await stopLive();
  }
});

/**
 * Stops capture and turns the session into something playable.
 *
 * The retained audio is written out as a WAV blob and handed to the player, and
 * the timeline is rebased so the kept history starts at zero -- from here on it
 * behaves exactly like a loaded file.
 */
async function stopLive(): Promise<void> {
  if (!live) return;
  const session = live;
  live = null;
  await session.stop();
  updateSourcePrompt();
  worker.postMessage({ type: 'liveStop' });

  const { samples, startSeconds, sampleRate } = session.takeRecording();
  const shifted = session.rebase();
  if (shifted > 0) store.offset = 0;

  if (samples.length > 0) {
    if (mediaUrl) URL.revokeObjectURL(mediaUrl);
    mediaUrl = URL.createObjectURL(encodeWav(samples, sampleRate));
    player.load(mediaUrl);
    hasAudio = true;
    updateSourcePrompt();
    els.play.disabled = false;
    els.toStart.disabled = false;
    spec.viewport.duration = samples.length / sampleRate;
    spec.viewport.clamp();
    // The live buffer is bounded, so say so when the earlier audio is gone.
    els.filename.textContent = startSeconds > 0 ? 'Recorded audio (last 5 min)' : 'Recorded audio';
  }

  els.micLabel.textContent = 'Record';
  els.mic.title = 'Record from microphone';
  els.mic.classList.remove('on');
  els.mic.querySelector('use')?.setAttribute('href', '#i-record');
  els.liveBadge.hidden = true;
  els.micGainControl.hidden = true;
  els.meterFill.style.width = '0%';
  setStatus('');
  spec.invalidate();
  actsDirty = true;
}

// ------------------------------------------------------------------ boot ---

resize();
drawLegend();
updateSourcePrompt();
requestAnimationFrame(frame);
