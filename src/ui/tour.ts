/**
 * A guided walkthrough: a fixed-position spotlight ring around one element at
 * a time plus a bubble of text beside it, stepped with Next/Back. Built by
 * hand rather than as a library dependency -- the whole thing is positioning
 * two absolutely-placed boxes off `getBoundingClientRect`, which a dependency
 * would not make meaningfully simpler.
 *
 * The ring and bubble are recomputed on resize/scroll rather than cached,
 * since the panels reflow (the help dialog and settings dialog are the same
 * way). A step whose target is not currently in the DOM (or is offscreen) is
 * skipped rather than shown pointing at nothing.
 */

export interface TourOptions {
  /** Called right before the first step renders. */
  onStart?: () => void;
  /** Called once the tour ends, however it ends (Exit, Escape, or finishing the last step). */
  onStop?: () => void;
}

export interface TourStep {
  /**
   * Element id (or ids) to highlight -- an array rings the bounding box of all
   * of them. Omit for a sign-off step: no ring, bubble centred on screen.
   */
  target?: string | string[];
  title: string;
  body: string;
  /** Optional bullet list, e.g. suggestions on a closing step. */
  tips?: string[];
  /** Preferred side for the bubble; falls back automatically if there's no room. */
  placement?: 'top' | 'bottom' | 'left' | 'right';
}

export class Tour {
  private index = 0;
  private active = false;

  get isActive(): boolean {
    return this.active;
  }
  private readonly ring: HTMLDivElement;
  private readonly bubble: HTMLDivElement;
  private readonly backdrop: HTMLDivElement;
  private readonly reposition = () => this.render();

  constructor(
    private readonly steps: TourStep[],
    private readonly options: TourOptions = {},
  ) {
    this.backdrop = document.createElement('div');
    this.backdrop.className = 'tour-backdrop';

    this.ring = document.createElement('div');
    this.ring.className = 'tour-ring';

    this.bubble = document.createElement('div');
    this.bubble.className = 'tour-bubble';
  }

  start(): void {
    if (this.steps.length === 0) return;
    this.active = true;
    this.index = 0;
    this.options.onStart?.();
    document.body.append(this.backdrop, this.ring, this.bubble);
    window.addEventListener('resize', this.reposition);
    window.addEventListener('scroll', this.reposition, true);
    document.addEventListener('keydown', this.onKeydown);
    this.render();
  }

  stop(): void {
    if (!this.active) return;
    this.active = false;
    this.backdrop.remove();
    this.ring.remove();
    this.bubble.remove();
    window.removeEventListener('resize', this.reposition);
    window.removeEventListener('scroll', this.reposition, true);
    document.removeEventListener('keydown', this.onKeydown);
    this.options.onStop?.();
  }

  private readonly onKeydown = (e: KeyboardEvent) => {
    switch (e.key) {
      case 'Escape':
        this.stop();
        break;
      case 'ArrowRight':
      case 'Enter':
        e.preventDefault();
        this.next();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        this.back();
        break;
    }
  };

  private next(): void {
    if (this.index >= this.steps.length - 1) {
      this.stop();
      return;
    }
    this.index++;
    this.render();
  }

  private back(): void {
    if (this.index === 0) return;
    this.index--;
    this.render();
  }

  private render(): void {
    const step = this.steps[this.index];
    const hasTarget = step.target !== undefined;
    const ids = hasTarget ? (Array.isArray(step.target) ? step.target! : [step.target as string]) : [];
    const rects = ids.map((id) => document.getElementById(id)?.getBoundingClientRect()).filter((r): r is DOMRect => !!r);
    if (hasTarget && rects.length === 0) {
      // Target not present right now (e.g. a hidden control) -- skip it
      // rather than ring nothing.
      this.next();
      return;
    }

    // A sign-off step has nothing to ring, so the backdrop dims the whole
    // page itself rather than relying on the ring's spotlight cutout.
    this.ring.classList.toggle('tour-ring-hidden', rects.length === 0);
    this.backdrop.classList.toggle('tour-dim', rects.length === 0);
    let r = { left: 0, top: 0, right: 0, bottom: 0 };
    if (rects.length > 0) {
      r = {
        left: Math.min(...rects.map((x) => x.left)),
        top: Math.min(...rects.map((x) => x.top)),
        right: Math.max(...rects.map((x) => x.right)),
        bottom: Math.max(...rects.map((x) => x.bottom)),
      };
      const pad = 6;
      this.ring.style.left = `${r.left - pad}px`;
      this.ring.style.top = `${r.top - pad}px`;
      this.ring.style.width = `${r.right - r.left + pad * 2}px`;
      this.ring.style.height = `${r.bottom - r.top + pad * 2}px`;
    }
    const rWidth = r.right - r.left;
    const rHeight = r.bottom - r.top;

    this.bubble.replaceChildren();
    const h = document.createElement('h3');
    h.textContent = step.title;
    const p = document.createElement('p');
    p.textContent = step.body;
    this.bubble.append(h, p);
    if (step.tips && step.tips.length > 0) {
      const ul = document.createElement('ul');
      ul.className = 'tour-tips';
      for (const tip of step.tips) {
        const li = document.createElement('li');
        li.textContent = tip;
        ul.append(li);
      }
      this.bubble.append(ul);
    }
    const controls = document.createElement('div');
    controls.className = 'tour-controls';

    const count = document.createElement('span');
    count.className = 'tour-count mono small';
    count.textContent = `${this.index + 1} / ${this.steps.length}`;

    const skip = document.createElement('button');
    skip.type = 'button';
    skip.className = 'button small';
    skip.textContent = 'Exit';
    skip.addEventListener('click', () => this.stop());

    const back = document.createElement('button');
    back.type = 'button';
    back.className = 'button small';
    back.textContent = 'Back';
    back.disabled = this.index === 0;
    back.addEventListener('click', () => this.back());

    const nextBtn = document.createElement('button');
    nextBtn.type = 'button';
    nextBtn.className = 'button small';
    nextBtn.textContent = this.index === this.steps.length - 1 ? 'Done' : 'Next';
    nextBtn.addEventListener('click', () => this.next());

    controls.append(count, skip, back, nextBtn);
    this.bubble.append(controls);

    // Measure in place -- position:fixed sizing doesn't depend on left/top,
    // so there's no need to relocate the bubble first (which would fight the
    // position transition below with a visible jump to the corner).
    const bw = this.bubble.offsetWidth;
    const bh = this.bubble.offsetHeight;
    const gap = 14;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let pos: { x: number; y: number };
    if (rects.length === 0) {
      // No target: a sign-off step, centred on screen rather than pointing
      // at anything.
      pos = { x: vw / 2 - bw / 2, y: vh / 2 - bh / 2 };
    } else {
      const fits = (x: number, y: number) => x >= 8 && y >= 8 && x + bw <= vw - 8 && y + bh <= vh - 8;
      const candidates: Array<{ x: number; y: number }> = [];
      const order: Array<'top' | 'bottom' | 'left' | 'right'> = step.placement
        ? [step.placement, 'bottom', 'top', 'right', 'left']
        : ['bottom', 'top', 'right', 'left'];
      for (const side of order) {
        switch (side) {
          case 'bottom':
            candidates.push({ x: r.left + rWidth / 2 - bw / 2, y: r.bottom + gap });
            break;
          case 'top':
            candidates.push({ x: r.left + rWidth / 2 - bw / 2, y: r.top - gap - bh });
            break;
          case 'right':
            candidates.push({ x: r.right + gap, y: r.top + rHeight / 2 - bh / 2 });
            break;
          case 'left':
            candidates.push({ x: r.left - gap - bw, y: r.top + rHeight / 2 - bh / 2 });
            break;
        }
      }
      pos = candidates.find((c) => fits(c.x, c.y)) ?? candidates[0];
    }
    pos = { x: Math.min(Math.max(pos.x, 8), vw - bw - 8), y: Math.min(Math.max(pos.y, 8), vh - bh - 8) };

    this.bubble.style.left = `${pos.x}px`;
    this.bubble.style.top = `${pos.y}px`;
  }
}
