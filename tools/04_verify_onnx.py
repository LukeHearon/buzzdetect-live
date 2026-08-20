"""
Checks the exported ONNX against the SavedModels buzzdetect actually runs, by
feeding it the reference patches from 01_reference.py and comparing both the
1024-d embeddings and the 13 activations.

Tolerances are the budget the shipped build was chosen against, not float32
exactness -- conv kernels are stored as float16 from block 4 onward (see
03_build_onnx.py), buying half the download for a known amount of error.

  * activations: 1e-2, one quantum of buzzdetect's 2-decimal CSV output
    (digits_results = 2). Below this, a printed value cannot move by more than
    one unit in its last place.
  * embeddings: 2e-2, informational. Embeddings are an intermediate; what
    survives into the activations is what matters.

Neither bound is the real acceptance test. That is 05_detections.py: whether any
frame crosses the threshold differently. Building with --precision fp32 drops
both figures to ~1e-5.
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')


def report(name, got, want, tol):
    d = np.abs(got - want)
    ok = d.max() < tol
    print(f'  {name:<12} max|diff| {d.max():.3e}   mean|diff| {d.mean():.3e}   '
          f'{"PASS" if ok else "FAIL"} (tol {tol:g})')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default='testbuzz')
    ap.add_argument('--model', default='buzzdetect_v3.onnx')
    args = ap.parse_args()

    ref = np.load(os.path.join(DIR, f'{args.ref}.npz'))
    patches = ref['patches'].astype(np.float32)

    path = os.path.join(DIR, args.model)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

    acts, embs = sess.run(['activations', 'embeddings'], {'patches': patches})

    print(f'{args.model} vs SavedModel, {len(patches)} patches from {ref["source"]}')
    ok = report('embeddings', embs, ref['embeddings'], 2e-2)
    ok &= report('activations', acts, ref['logits'], 1e-2)

    # what a user would actually see: buzzdetect rounds to 2 decimals
    r_got, r_want = acts.round(2), ref['logits'].round(2)
    n_diff = int((r_got != r_want).sum())
    print(f'  rounded to 2dp: {n_diff}/{r_got.size} cells differ '
          f'({100 * n_diff / r_got.size:.3f}%)')

    buzz = list(ref['classes']).index('ins_buzz')
    d_buzz = np.abs(acts[:, buzz] - ref['logits'][:, buzz])
    print(f'  ins_buzz column: max|diff| {d_buzz.max():.3e}, '
          f'range {ref["logits"][:, buzz].min():.2f}..{ref["logits"][:, buzz].max():.2f}')

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
