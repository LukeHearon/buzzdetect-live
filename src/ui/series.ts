/**
 * Which of the model's thirteen outputs are plotted, in what colour, and against
 * what threshold.
 *
 * Colours are hard-coded per class rather than dealt out of a palette by index,
 * and they are grouped by what the class *is*: insects warm yellow, ambient
 * green, mechanical blue, other animals and people pink through rose. With up to
 * thirteen lines on one axis, a flat categorical palette leaves the reader
 * matching swatches one at a time; a family tells you what kind of thing spiked
 * before you have found its name in the legend. Within a family the hues are
 * held apart by lightness, which survives being drawn as a 1.5 px line.
 *
 * Thresholds are per class because they are per class in buzzdetect: −1.2 is the
 * default for `ins_buzz`, and nothing says the same number means the same thing
 * for rain. They are unbounded model units, so there is no shared scale to
 * normalise them onto.
 */

import { CLASSES } from '../model/session';

export type ClassName = (typeof CLASSES)[number];

/** buzzdetect's own default cutoff, applied to every class until changed. */
export const DEFAULT_THRESHOLD = -1.2;

const COLORS: Record<ClassName, string> = {
  // Insects: warm yellows.
  ins_buzz: '#fcd34d',
  ins_trill: '#f59e0b',
  // Ambient: greens.
  ambient_rain: '#4ade80',
  ambient_noise: '#15803d',
  ambient_background: '#a7f3d0',
  // Mechanical: blues and cyans.
  mech_plane: '#38bdf8',
  mech_train: '#60a5fa',
  mech_auto: '#818cf8',
  mech_hum: '#22d3ee',
  mech_siren: '#c7d2fe',
  // Everything else alive: pinks through rose.
  bird_goose: '#f9a8d4',
  frog: '#e879f9',
  human: '#fb7185',
};

/**
 * Display order in the settings list and the legend: grouped, so the colour
 * families read as families. Not the model's output order, which is arbitrary.
 */
export const DISPLAY_ORDER: ClassName[] = [
  'ins_buzz',
  'ins_trill',
  'ambient_rain',
  'ambient_noise',
  'ambient_background',
  'mech_plane',
  'mech_train',
  'mech_auto',
  'mech_hum',
  'mech_siren',
  'bird_goose',
  'frog',
  'human',
];

/** On at startup: the buzz the tool is for, plus one ambient and one mechanical. */
const DEFAULT_ENABLED: ClassName[] = ['ins_buzz', 'ambient_rain', 'mech_plane'];

export function colorFor(name: string): string {
  return COLORS[name as ClassName] ?? '#94a3b8';
}

/** One plotted line: what to read out of the store, and how to draw it. */
export interface Series {
  name: ClassName;
  index: number;
  color: string;
  threshold: number;
  enabled: boolean;
}

/**
 * The live per-class state, in display order. One array, mutated in place --
 * the views read it every frame and the settings dialog writes it, so a copy
 * would only be a second thing to keep in step.
 */
export const SERIES: Series[] = DISPLAY_ORDER.map((name) => ({
  name,
  index: CLASSES.indexOf(name),
  color: colorFor(name),
  threshold: DEFAULT_THRESHOLD,
  enabled: DEFAULT_ENABLED.includes(name),
}));

export function enabledSeries(): Series[] {
  return SERIES.filter((s) => s.enabled);
}
