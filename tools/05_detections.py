"""Detection-count agreement at a fixed ins_buzz threshold.

Rounding error only matters where it moves an activation across the threshold,
so this counts flips against the SavedModel reference, not just totals.
"""
import argparse, os, glob
import numpy as np, onnxruntime as ort

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')

ap = argparse.ArgumentParser()
ap.add_argument('--ref', default='testbuzz')
ap.add_argument('--threshold', type=float, default=-0.75)
a = ap.parse_args()

ref = np.load(os.path.join(DIR, f'{a.ref}.npz'))
patches = ref['patches'].astype(np.float32)
buzz = list(ref['classes']).index('ins_buzz')

truth = ref['logits'][:, buzz] > a.threshold
print(f'{len(patches)} frames, threshold ins_buzz > {a.threshold}')
print(f'{"model":<28} {"MB":>6} {"detections":>11} {"flips":>7}   margin of nearest flip')
print(f'{"buzzdetect (SavedModel)":<28} {"-":>6} {truth.sum():>11} {"-":>7}')

for path in sorted(glob.glob(os.path.join(DIR, '*.onnx'))):
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    act = s.run(['activations'], {'patches': patches})[0][:, buzz]
    got = act > a.threshold
    flip = got != truth
    near = (f'{np.abs(ref["logits"][flip, buzz] - a.threshold).max():.4f}'
            if flip.any() else '-')
    print(f'{os.path.basename(path):<28} {os.path.getsize(path)/1e6:>6.2f} '
          f'{got.sum():>11} {flip.sum():>7}   {near}')
