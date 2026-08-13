/**
 * A two-thumb range control, built by hand because the platform has no such
 * input. The usual substitute -- two <input type="range"> overlaid and styled
 * to look like one -- was rejected: the pair fight over hit-testing wherever
 * the thumbs meet, which is exactly where a contrast window is adjusted.
 *
 * Vertical is the primary orientation here, so it is the only one implemented.
 * The value axis runs bottom-to-top to match the thing it sits beside: on a
 * spectrogram, loud is up.
 *
 * Pointer events rather than mouse/touch pairs, with capture on the thumb, so a
 * drag that leaves the track keeps tracking instead of stopping at the edge.
 * The thumbs are real <button>s so they are focusable and arrow-key adjustable
 * without a roving tabindex; each carries the ARIA slider role and its own
 * value, which is what a screen reader needs to read the pair as a range.
 */

export interface DualRangeOptions {
  min: number;
  max: number;
  step: number;
  /** Smallest allowed distance between the thumbs, in value units. */
  gap: number;
  /** Initial [low, high]. */
  values: [number, number];
  /** Accessible name, applied to both thumbs with "minimum"/"maximum". */
  label: string;
  onInput: (low: number, high: number) => void;
}

export interface DualRange {
  readonly element: HTMLElement;
  values(): [number, number];
  setValues(low: number, high: number): void;
}

export function createDualRange(options: DualRangeOptions): DualRange {
  const { min, max, step, gap, label, onInput } = options;
  let [low, high] = options.values;

  const element = document.createElement('div');
  element.className = 'dualrange';

  const track = document.createElement('div');
  track.className = 'dualrange-track';

  const fill = document.createElement('div');
  fill.className = 'dualrange-fill';

  const thumbs: Record<'low' | 'high', HTMLButtonElement> = {
    low: document.createElement('button'),
    high: document.createElement('button'),
  };
  for (const which of ['low', 'high'] as const) {
    const thumb = thumbs[which];
    thumb.type = 'button';
    thumb.className = 'dualrange-thumb';
    thumb.setAttribute('role', 'slider');
    thumb.setAttribute('aria-label', `${label} ${which === 'low' ? 'minimum' : 'maximum'}`);
    thumb.setAttribute('aria-valuemin', String(min));
    thumb.setAttribute('aria-valuemax', String(max));
  }

  element.append(track, fill, thumbs.low, thumbs.high);

  const clampStep = (v: number): number => {
    const snapped = Math.round((v - min) / step) * step + min;
    return Math.min(max, Math.max(min, snapped));
  };

  /** Fraction of the track, 0 at the bottom. */
  const fraction = (v: number): number => (v - min) / (max - min);

  function render(): void {
    const lowPct = fraction(low) * 100;
    const highPct = fraction(high) * 100;
    thumbs.low.style.bottom = `${lowPct}%`;
    thumbs.high.style.bottom = `${highPct}%`;
    fill.style.bottom = `${lowPct}%`;
    fill.style.height = `${highPct - lowPct}%`;
    thumbs.low.setAttribute('aria-valuenow', String(low));
    thumbs.high.setAttribute('aria-valuenow', String(high));
  }

  function set(which: 'low' | 'high', raw: number, notify: boolean): void {
    const v = clampStep(raw);
    if (which === 'low') {
      low = Math.min(v, high - gap);
    } else {
      high = Math.max(v, low + gap);
    }
    render();
    if (notify) onInput(low, high);
  }

  /** Value at a client Y, with the track's own bottom as the origin. */
  function valueAt(clientY: number): number {
    const box = element.getBoundingClientRect();
    if (box.height === 0) return min;
    const fromBottom = (box.bottom - clientY) / box.height;
    return min + fromBottom * (max - min);
  }

  for (const which of ['low', 'high'] as const) {
    const thumb = thumbs[which];
    thumb.addEventListener('pointerdown', (e: PointerEvent) => {
      e.preventDefault();
      thumb.focus();
      thumb.setPointerCapture(e.pointerId);
    });
    thumb.addEventListener('pointermove', (e: PointerEvent) => {
      if (!thumb.hasPointerCapture(e.pointerId)) return;
      set(which, valueAt(e.clientY), true);
    });
    thumb.addEventListener('keydown', (e: KeyboardEvent) => {
      // Page keys move by ten steps; Home/End run to the far end, which the
      // opposing thumb's gap then limits.
      const big = step * 10;
      const delta =
        e.key === 'ArrowUp' || e.key === 'ArrowRight'
          ? step
          : e.key === 'ArrowDown' || e.key === 'ArrowLeft'
            ? -step
            : e.key === 'PageUp'
              ? big
              : e.key === 'PageDown'
                ? -big
                : 0;
      if (delta !== 0) {
        e.preventDefault();
        set(which, (which === 'low' ? low : high) + delta, true);
      } else if (e.key === 'Home' || e.key === 'End') {
        e.preventDefault();
        set(which, e.key === 'Home' ? min : max, true);
      }
    });
  }

  // A press on the track moves whichever thumb is nearer, so the control can be
  // aimed at a level directly instead of dragged to it.
  element.addEventListener('pointerdown', (e: PointerEvent) => {
    if (e.target !== element && e.target !== track && e.target !== fill) return;
    const v = valueAt(e.clientY);
    const which = Math.abs(v - low) <= Math.abs(v - high) ? 'low' : 'high';
    thumbs[which].focus();
    set(which, v, true);
  });

  render();

  return {
    element,
    values: () => [low, high],
    setValues(nextLow: number, nextHigh: number): void {
      low = clampStep(nextLow);
      high = Math.max(clampStep(nextHigh), low + gap);
      render();
    },
  };
}
