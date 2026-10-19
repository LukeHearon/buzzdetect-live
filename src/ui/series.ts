/**
 * Which of the current model's classes are selectable, in what colour, and
 * against what threshold.
 *
 * The models differ in what they output -- fast_v1 and balanced_v1 have three
 * classes, heavy_v1 fifteen -- so the series are rebuilt from each model's own
 * class list when it loads (see setModelClasses). A class is offered only if
 * it is in DISPLAY_ORDER: that is where it gets a colour, and it is how a class
 * whose zero means nothing (one buzzdetect-training could not center; see
 * DEFAULT_THRESHOLD) is kept out of the settings dialog.
 *
 * Colours are hard-coded per class rather than dealt out of a palette by index,
 * and they are grouped by what the class *is*: insects warm yellow, ambient
 * green, mechanical blue, other animals and people pink through rose. A flat
 * categorical palette leaves the reader matching swatches one at a time; a
 * family tells you what kind of thing spiked before you have found its name in
 * the legend. Within a family the hues are held apart by lightness, which
 * survives being drawn as a 1.5 px line.
 *
 * Thresholds are per class because they are per class in buzzdetect, and they
 * are unbounded model units, so there is no shared scale to normalise them
 * onto. They start at the model's suggestion on every switch rather than
 * carrying over: a number tuned against one model's scale says nothing about
 * another's.
 */

/**
 * buzzdetect's suggested cutoff, the same for every class: zero.
 *
 * buzzdetect-training's export subtracts each class's suggested threshold (its
 * 95%-precision point over the CV folds) from that class's output inside the
 * graph, so these models' results are on a shifted scale where positive means
 * detected at the suggestion. Only classes the folds could center get that
 * meaning; heavy_v1's ambient_scraping and mech_quadcopter ship uncentered,
 * and are not offered.
 */
export const DEFAULT_THRESHOLD = 0;

/** Colours are only defined for classes this app actually offers (DISPLAY_ORDER). */
const COLORS: Record<string, string> = {
  // Insects: warm yellows.
  ins_buzz: '#fcd34d',
  ins_trill: '#f59e0b',
  // Ambient: greens.
  ambient_rain: '#4ade80',
  ambient_thunder: '#a3e635',
  ambient_noise: '#15803d',
  ambient_background: '#86efac',
  ambient_music: '#2dd4bf',
  // Mechanical: blues and cyans.
  mech_plane: '#38bdf8',
  mech_auto: '#818cf8',
  mech_hum: '#22d3ee',
  mech_machinery: '#60a5fa',
  // Everything else alive: pinks through rose.
  human: '#fb7185',
  animal: '#f0abfc',
};

/**
 * The classes this app lets a user select, in display order for the settings
 * list and the legend: grouped, so the colour families read as families. Not
 * any model's own output order, which is arbitrary. A model offers whichever of
 * these it outputs.
 */
const DISPLAY_ORDER: string[] = [
  'ins_buzz',
  'ins_trill',
  'ambient_rain',
  'ambient_thunder',
  'ambient_noise',
  'ambient_background',
  'ambient_music',
  'mech_plane',
  'mech_auto',
  'mech_hum',
  'mech_machinery',
  'human',
  'animal',
];

export function colorFor(name: string): string {
  return COLORS[name] ?? '#94a3b8';
}

/** One plotted line: what to read out of the store, and how to draw it. */
export interface Series {
  name: string;
  index: number;
  color: string;
  threshold: number;
  enabled: boolean;
}

/**
 * Which classes are switched on, by name, across model switches -- so a class
 * picked on heavy_v1 is still picked on returning to it, even after a model
 * that lacks it. On at startup: the buzz the tool is for, plus people, plus a
 * mechanical class where the model has one.
 */
const enabledNames = new Set(['ins_buzz', 'human', 'mech_plane']);

/**
 * The live per-class state for the current model, in display order. One array,
 * mutated in place -- the views read it every frame and the settings dialog
 * writes it, so a copy would only be a second thing to keep in step.
 */
export const SERIES: Series[] = [];

/** Rebuilds SERIES for a model's output columns. */
export function setModelClasses(classes: string[]): void {
  const next = DISPLAY_ORDER.filter((name) => classes.includes(name)).map((name) => ({
    name,
    index: classes.indexOf(name),
    color: colorFor(name),
    threshold: DEFAULT_THRESHOLD,
    enabled: enabledNames.has(name),
  }));
  SERIES.splice(0, SERIES.length, ...next);
}

/** Switches a class on or off, remembered by name for other models too. */
export function setEnabled(s: Series, on: boolean): void {
  s.enabled = on;
  if (on) enabledNames.add(s.name);
  else enabledNames.delete(s.name);
}

export function enabledSeries(): Series[] {
  return SERIES.filter((s) => s.enabled);
}
