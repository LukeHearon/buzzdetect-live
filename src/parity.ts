/**
 * In-browser parity check (parity.html).
 *
 * The node test in test/model.test.ts already compares the app's framing and
 * the models against buzzdetect's numbers, but it does so with node reading a
 * WAV off disk. This runs the same comparison through the path a user actually
 * takes -- fetch, `decodeAudioData`, the worker's batching on this machine's
 * browser build of the wasm runtime -- so a difference introduced by the
 * browser itself cannot hide. Every model is checked, one after another.
 */

import { decodeForAnalysis } from './audio/decode';
import { frameCount } from './model/framing';
import { MODEL_NAMES, type ModelName } from './model/models';
import { loadModel, runFrames, scratchFor, BATCH_FRAMES } from './model/session';

interface ModelReference {
  classes: string[];
  activations: number[][];
}

interface Reference {
  source: string;
  samples: number;
  models: Record<ModelName, ModelReference>;
}

/** Difference at which a value would round differently in buzzdetect's 2-dp CSV. */
const VISIBLE = 5e-3;

function el(tag: string, className = '', text = ''): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

async function check(name: ModelName, wav: Float32Array, reference: ModelReference): Promise<HTMLElement[]> {
  const model = await loadModel(name);
  const { classes } = reference;
  const nClasses = classes.length;
  const n = frameCount(model.config, wav.length);

  const got = new Float32Array(n * nClasses);
  const scratch = scratchFor(model, BATCH_FRAMES);
  for (let first = 0; first < n; first += BATCH_FRAMES) {
    const count = Math.min(BATCH_FRAMES, n - first);
    got.set(await runFrames(model, wav, first, count, scratch), first * nClasses);
  }

  const heading = el('h2', '', name);
  if (n !== reference.activations.length || model.config.classes.join() !== classes.join()) {
    return [
      heading,
      el(
        'p',
        'fail',
        `FAIL — ${n} frames x ${model.config.classes.length} classes here, ` +
          `${reference.activations.length} x ${nClasses} in the reference`,
      ),
    ];
  }

  // Per-class worst case, plus the count of cells that would print differently.
  const perClass = classes.map(() => ({ max: 0, sum: 0 }));
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

  for (const [c, cls] of classes.entries()) {
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
      el('td', 'name', cls),
      el('td', stat.max > VISIBLE ? 'fail' : 'pass', stat.max.toExponential(2)),
      el('td', '', (stat.sum / n).toExponential(2)),
      el('td', '', `${lo.toFixed(2)} … ${hi.toFixed(2)}`),
    );
    table.append(row);
  }

  const overall = Math.max(...perClass.map((s) => s.max));
  const cells = n * nClasses;

  const summary = el('div', 'summary');
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
    `runtime: ${model.threads} thread(s), model ${(model.bytes / 1e6).toFixed(2)} MB`;

  model.session.release();
  return [heading, summary, table, provenance];
}

async function main(): Promise<void> {
  const out = document.getElementById('out')!;
  const base = document.baseURI;

  const reference: Reference = await (
    await fetch(new URL('sample/testbuzz.reference.json', base))
  ).json();

  const blob = await (await fetch(new URL('sample/testbuzz.wav', base))).blob();
  const decoded = await decodeForAnalysis(new File([blob], 'sample.wav', { type: 'audio/wav' }));

  const provenance = el('pre');
  provenance.textContent =
    `sample:  ${reference.source}\n` +
    `decoded: ${decoded.wav.length} samples (reference ${reference.samples}), ` +
    `${decoded.duration.toFixed(3)} s`;
  out.replaceChildren(provenance);

  for (const name of MODEL_NAMES) {
    const running = el('p', '', `Running ${name}…`);
    out.append(running);
    try {
      running.replaceWith(...(await check(name, decoded.wav, reference.models[name])));
    } catch (err) {
      running.replaceWith(el('h2', '', name), el('p', 'fail', err instanceof Error ? err.message : String(err)));
    }
  }
}

main().catch((err) => {
  const out = document.getElementById('out')!;
  out.replaceChildren(el('p', 'fail', err instanceof Error ? err.message : String(err)));
});
