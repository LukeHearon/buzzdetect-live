/**
 * Colour lookup for the spectrogram.
 *
 * A 256-entry RGB table built once, so drawing a column is an array index per
 * pixel rather than arithmetic. The ramp is perceptually ordered (dark low,
 * bright high, monotonic in lightness) so apparent brightness tracks level
 * instead of the hue cycling of a rainbow map, which invents edges that are not
 * in the data.
 */

/** Control points: [position 0..1, r, g, b]. Interpolated in sRGB. */
const MAGMA_STOPS: Array<[number, number, number, number]> = [
  [0.0, 0, 0, 4],
  [0.13, 28, 16, 68],
  [0.25, 79, 18, 123],
  [0.38, 129, 37, 129],
  [0.5, 181, 54, 122],
  [0.63, 229, 80, 100],
  [0.75, 251, 135, 97],
  [0.88, 254, 194, 135],
  [1.0, 252, 253, 191],
];

const GREY_STOPS: Array<[number, number, number, number]> = [
  [0.0, 0, 0, 0],
  [1.0, 255, 255, 255],
];

export type ColormapName = 'magma' | 'grey';

function build(stops: Array<[number, number, number, number]>): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  let seg = 0;
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    while (seg < stops.length - 2 && t > stops[seg + 1][0]) seg++;
    const [t0, r0, g0, b0] = stops[seg];
    const [t1, r1, g1, b1] = stops[seg + 1];
    const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
    lut[i * 3] = r0 + (r1 - r0) * f;
    lut[i * 3 + 1] = g0 + (g1 - g0) * f;
    lut[i * 3 + 2] = b0 + (b1 - b0) * f;
  }
  return lut;
}

const CACHE = new Map<ColormapName, Uint8ClampedArray>();

export function colormap(name: ColormapName): Uint8ClampedArray {
  let lut = CACHE.get(name);
  if (!lut) {
    lut = build(name === 'grey' ? GREY_STOPS : MAGMA_STOPS);
    CACHE.set(name, lut);
  }
  return lut;
}

/**
 * Maps a stored byte through the user's level window to a colour index.
 *
 * Returns a 256-entry table so the render loop can index it directly instead of
 * clamping and rescaling per pixel.
 */
export function levelTable(minByte: number, maxByte: number): Uint8Array {
  const table = new Uint8Array(256);
  const span = Math.max(1, maxByte - minByte);
  for (let v = 0; v < 256; v++) {
    const t = ((v - minByte) / span) * 255;
    table[v] = t <= 0 ? 0 : t >= 255 ? 255 : t;
  }
  return table;
}
