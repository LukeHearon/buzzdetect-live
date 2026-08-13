/**
 * In-browser parity check (parity.html).
 *
 * The node test in test/model.test.ts already compares the front end and the
 * model against buzzdetect's numbers, but it does so with node reading a raw
 * float array off disk. This runs the same comparison through the path a user
 * actually takes -- fetch, `decodeAudioData`, the worker's code on this
 * machine's browser build of the wasm runtime -- so a difference introduced by
 * the browser itself cannot hide.
 */

import { decodeForAnalysis } from './audio/decode';
import { MelPatchExtractor, PATCH_VALUES } from './dsp/melspec';
import { loadModel, runBatch, CLASSES } from './model/session';

interface Reference {
  source: string;
  framehopProp: number;
  classes: string[];
  activations: number[][];
}

/** Difference at which a value would round differently in buzzdetect's 2-dp CSV. */
const VISIBLE = 5e-3;

function el(tag: string, className = '', text = ''): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

async function main(): Promise<void> {
  const out = document.getElementById('out')!;
  const base = document.baseURI;

  const reference: Reference = await (
    await fetch(new URL('sample/soybean_10s.reference.json', base))
  ).json();

  const blob = await (await fetch(new URL('sample/soybean_10s.wav', base))).blob();
  const decoded = await decodeForAnalysis(new File([blob], 'sample.wav', { type: 'audio/wav' }));

  const extractor = new MelPatchExtractor(reference.framehopProp);
  const n = reference.activations.length;
  const patches = new Float32Array(n * PATCH_VALUES);
  extractor.fillPatches(decoded.wav, 0, n, patches);

  const model = await loadModel();
  const got = await runBatch(model.session, patches, n);

  // Per-class worst case, plus the count of cells that would print differently.
  const nClasses = CLASSES.length;
  const perClass = CLASSES.map(() => ({ max: 0, sum: 0 }));
  let visible = 0;
  let roundedDiffer = 0;

  for (let p = 0; p < n; p++) {
    for (let c = 0; c < nClasses; c++) {
      const want = reference.activations[p][c];
      const d = Math.abs(got[p * nClasses + c] - want);
      const stat = perClass[c];
      if (d > stat.max) stat.max = d;
      stat.sum += d;
      if (d > VISIBLE) visible++;
      if (Math.round(got[p * nClasses + c] * 100) !== Math.round(want * 100)) roundedDiffer++;
    }
  }

  const table = el('table');
  const head = el('tr');
  head.append(
    el('th', 'name', 'class'),
    el('th', '', 'max |diff|'),
    el('th', '', 'mean |diff|'),
    el('th', '', 'range'),
  );
  table.append(head);

  for (const [c, name] of CLASSES.entries()) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let p = 0; p < n; p++) {
      const v = reference.activations[p][c];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const row = el('tr');
    const stat = perClass[c];
    row.append(
      el('td', 'name', name),
      el('td', stat.max > VISIBLE ? 'fail' : 'pass', stat.max.toExponential(2)),
      el('td', '', (stat.sum / n).toExponential(2)),
      el('td', '', `${lo.toFixed(2)} … ${hi.toFixed(2)}`),
    );
    table.append(row);
  }

  const overall = Math.max(...perClass.map((s) => s.max));
  const cells = n * nClasses;

  const summary = el('div');
  summary.id = 'summary';
  summary.append(
    el(
      'div',
      overall < VISIBLE ? 'pass' : 'fail',
      `${overall < VISIBLE ? 'PASS' : 'FAIL'} — worst activation differs by ` +
        `${overall.toExponential(2)} across ${cells} values ` +
        `(${n} frames x ${nClasses} classes)`,
    ),
    el(
      'div',
      '',
      `${roundedDiffer} of ${cells} would round differently at buzzdetect's ` +
        `2-decimal output precision; ${visible} exceed ${VISIBLE}.`,
    ),
  );

  const provenance = el('pre');
  provenance.textContent =
    `sample:  ${reference.source}\n` +
    `decoded: ${decoded.wav.length} samples, ${decoded.duration.toFixed(3)} s\n` +
    `runtime: ${model.threads} thread(s), model ${(model.bytes / 1e6).toFixed(2)} MB`;

  out.replaceChildren(summary, table, provenance);
}

main().catch((err) => {
  const out = document.getElementById('out')!;
  out.replaceChildren(el('p', 'fail', err instanceof Error ? err.message : String(err)));
});
