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
import { BUZZ_INDEX, CLASSES } from './model/session';
import { SpectrogramView } from './ui/spectrogramView';
import { ActivationStore, ActivationView, colorFor } from './ui/activationView';
import { formatTimeShort, normalizeSelection, type Selection } from './ui/viewport';
import { LiveSession, LIVE_STORE_PATCHES } from './live/liveSession';
import { createDualRange } from './ui/rangeSlider';
import type { WorkerResponse } from './workers/analysis.worker';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const els = {
  file: $<HTMLInputElement>('file'),
  mic: $<HTMLButtonElement>('mic'),
  micLabel: $('mic-label'),
  liveBadge: $('live-badge'),
  meterFill: $('meter-fill'),
  status: $('status'),
  filename: $('filename'),
  minHz: $<HTMLInputElement>('min-hz'),
  maxHz: $<HTMLInputElement>('max-hz'),
  contrast: $('contrast'),
  spec: $('spec'),
  specAxis: $<HTMLCanvasElement>('spec-axis'),
  timeaxis: $('timeaxis'),
  acts: $('acts'),
  actsAxis: $<HTMLCanvasElement>('acts-axis'),
  threshold: $<HTMLInputElement>('threshold'),
  detectionCount: $('detection-count'),
  toStart: $<HTMLButtonElement>('to-start'),
  play: $<HTMLButtonElement>('play'),
  playIcon: document.getElementById('play-icon') as unknown as SVGUseElement,
  time: $<HTMLInputElement>('time'),
  volume: $<HTMLInputElement>('volume'),
  volumeOut: $<HTMLOutputElement>('volume-out'),
  series: $('series'),
};

const spec = new SpectrogramView(els.spec, els.specAxis);
const acts = new ActivationView(els.acts, els.actsAxis);
const store = new ActivationStore();
const player = new Player();

/** The only class shown. buzzdetect's other twelve are still computed, just not plotted. */
const CLASS_INDEX = BUZZ_INDEX;
const SERIES_COLOR = colorFor(CLASSES[CLASS_INDEX]);
// One source of truth for the series colour: the label matches the polyline.
els.series.style.color = SERIES_COLOR;

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
  const { min, max } = store.range(CLASS_INDEX);
  acts.min = Math.min(acts.min, Math.floor(min * 2) / 2);
  acts.max = Math.max(acts.max, Math.ceil(max * 2) / 2);
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

let dragging: { startTime: number; moved: boolean; onActs: boolean } | null = null;

for (const area of [els.spec, els.acts]) {
  const onActs = area === els.acts;

  area.addEventListener('pointerdown', (e) => {
    if (!hasData()) return;
    area.setPointerCapture(e.pointerId);
    dragging = { startTime: spec.viewport.xToTime(offsetX(e, area)), moved: false, onActs };
  });

  area.addEventListener('pointermove', (e) => {
    const x = offsetX(e, area);
    hoverX = x;
    overlayDirty = true;

    const t = spec.viewport.xToTime(x);

    if (dragging) {
      if (Math.abs(t - dragging.startTime) > 0.002) dragging.moved = true;
      if (dragging.moved) {
        selection = normalizeSelection({ from: dragging.startTime, to: t });
        actsDirty = true;
      }
    }
  });

  area.addEventListener('pointerup', (e) => {
    if (!dragging) return;
    const t = spec.viewport.xToTime(offsetX(e, area));

    if (!dragging.moved) {
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

  area.addEventListener('pointerleave', () => {
    hoverX = null;
    overlayDirty = true;
  });

  area.addEventListener(
    'wheel',
    (e) => {
      if (!hasData()) return;
      e.preventDefault();

      // A trackpad's horizontal swipe arrives as deltaX; treat the dominant
      // axis as the gesture, so a sideways swipe pans and a pinch or vertical
      // wheel zooms. Shift+wheel pans too, for a mouse with one wheel.
      const horizontal = Math.abs(e.deltaX) > Math.abs(e.deltaY);
      if (horizontal || e.shiftKey) {
        const delta = horizontal ? e.deltaX : e.deltaY;
        spec.viewport.panBy(delta / spec.viewport.pixelsPerSecond);
        // Panning away from the right edge in live mode stops the auto-follow,
        // so you can look at something while capture continues.
        if (live) following = spec.viewport.end >= spec.viewport.duration - 0.5;
      } else {
        const factor = Math.exp(-Math.max(-100, Math.min(100, e.deltaY)) * 0.004);
        spec.viewport.zoomAt(offsetX(e, area), factor, minPixelsPerSecond());
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

els.threshold.addEventListener('change', () => {
  const v = Number(els.threshold.value);
  if (!Number.isFinite(v)) {
    els.threshold.value = acts.threshold.toFixed(2);
    return;
  }
  acts.threshold = v;
  els.threshold.value = v.toFixed(2);
  actsDirty = true;
  updateDetectionCount();
});

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

function updateDetectionCount(): void {
  if (store.filled === 0) {
    els.detectionCount.textContent = '';
    return;
  }
  const n = store.countDetections(CLASS_INDEX, acts.threshold);
  els.detectionCount.textContent = `${n} detections from ${store.filled} frames`;
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
/** Live only: whether the viewport is pinned to the right edge. */
let following = true;
const LIVE_REDRAW_MS = 33;
let lastLiveDraw = 0;
let lastScrollDraw = 0;

function frame(): void {
  const vp = spec.viewport;

  if (live?.running) {
    // A scrolling spectrogram is a full repaint, so it runs at ~30 Hz rather
    // than every frame. At the live scroll rate that is a sub-pixel difference
    // on screen and half the CPU.
    const now = performance.now();
    if (now - lastLiveDraw >= LIVE_REDRAW_MS) {
      lastLiveDraw = now;
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
  }

  if (player.playing) {
    // Locked to the playhead: it sits in the middle and the audio moves past
    // it, so the view never jumps and never has to be chased.
    //
    // Scrolling means a full spectrogram repaint, so the view is advanced at
    // ~30 Hz while the playhead line itself still moves every frame on the
    // cheap overlay. The difference is imperceptible and it halves the cost of
    // playing a file.
    const now = performance.now();
    if (now - lastScrollDraw >= LIVE_REDRAW_MS) {
      lastScrollDraw = now;
      vp.start = player.currentTime - vp.visibleSeconds / 2;
      vp.clamp();
    }
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

  spec.drawBase();

  if (actsDirty) {
    actsDirty = false;
    acts.draw(store, CLASS_INDEX, SERIES_COLOR, vp, playhead(), selection);
  }

  if (overlayDirty) {
    overlayDirty = false;
    spec.drawOverlay(playhead(), selection, hoverX);
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
    spec.setData(session.data);
    spec.viewport.duration = 0;
    spec.viewport.pixelsPerSecond = 60;
    worker.postMessage({ type: 'liveStart' });

    await session.start();

    els.micLabel.textContent = 'Stop';
    els.mic.title = 'Stop recording';
    els.mic.classList.add('on');
    els.mic.querySelector('use')?.setAttribute('href', '#i-stop');
    els.liveBadge.hidden = false;
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
  worker.postMessage({ type: 'liveStop' });

  const { samples, startSeconds, sampleRate } = session.takeRecording();
  const shifted = session.rebase();
  if (shifted > 0) store.offset = 0;

  if (samples.length > 0) {
    if (mediaUrl) URL.revokeObjectURL(mediaUrl);
    mediaUrl = URL.createObjectURL(encodeWav(samples, sampleRate));
    player.load(mediaUrl);
    hasAudio = true;
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
  els.meterFill.style.width = '0%';
  setStatus('');
  spec.invalidate();
  actsDirty = true;
}

// ------------------------------------------------------------------ boot ---

resize();
requestAnimationFrame(frame);
