"""
Copies buzzdetect's shipped models into the web app, and writes the reference
activations the parity checks compare against.

buzzdetect's models are already single ONNX graphs, waveform in and
predictions out, with the log-mel front end, the trunk and the head fused by
buzzdetect-training's export. So there is nothing to build: this copies each
model's `model.onnx` and `config_model.json` verbatim into public/model/<name>/,
after checking both against the sha256 in buzzdetect's own models.json, so what
the browser runs is byte-for-byte the released model.

The full-precision graph is shipped, not `model.fp16.onnx`. buzzdetect itself
only uses the fp16 sibling on a GPU, by request, and documents it as ~2e-2 off
the fp32 predictions -- far outside the parity this app is held to.

The references come from buzzdetect's own OnnxModel -- its session pinned to a
fixed length, its zero padding, its n_frames trim -- run over the shipped demo
clip, which is short enough to be one buzzdetect chunk. They are buzzdetect's
numbers, not a second implementation of them.

Run from the buzzdetect engine directory, with the engine's own venv (the
`buzzdetect` conda env predates the ONNX engine and has no onnxruntime):

    cd ~/Documents/bioacoustics/buzzdetect/engine
    .venv/bin/python ~/Tools/buzzdetect-live/tools/models.py
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.getcwd())
from src.inference.models import load_model  # noqa: E402

MODELS = ['balanced_v1', 'fast_v1', 'heavy_v1']
FILES = ['model.onnx', 'config_model.json']

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_MODEL_OUT = os.path.join(ROOT, 'public', 'model')
PATH_SAMPLE = os.path.join(ROOT, 'public', 'sample', 'testbuzz.wav')
PATH_REFERENCE = os.path.join(ROOT, 'public', 'sample', 'testbuzz.reference.json')
# buzzdetect's catalog of released models, one level above the engine.
PATH_CATALOG = os.path.join(os.path.dirname(os.getcwd()), 'models.json')

# Lengths at which the float32 frame count is worth pinning: each side of the
# first frame, a few exact multiples of the hop (where buzzdetect's float32
# reciprocal can round a quotient just over an integer), and the two lengths
# OnnxModel.n_frames' docstring names as ones float64 gets wrong.
LENGTHS = [0, 1, 15359, 15360, 15599, 15600, 15601, 30718, 30719, 30720,
           61680, 307200, 3210480, 16000 * 3600]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def copy_model(name, catalog):
    entry = next((m for m in catalog['models'] if m['name'] == name), None)
    if entry is None:
        raise SystemExit(f'{name} is not in {PATH_CATALOG}')
    model = load_model(name, framehop_prop=1, initialize=False)
    out = os.path.join(DIR_MODEL_OUT, name)
    os.makedirs(out, exist_ok=True)
    for fname in FILES:
        src = os.path.join(model.model_dir, fname)
        want = entry['files'][fname]['sha256']
        got = sha256(src)
        if got != want:
            raise SystemExit(f'{src}: sha256 {got} is not the released {want}')
        shutil.copyfile(src, os.path.join(out, fname))
    print(f'{name}: {os.path.getsize(os.path.join(out, "model.onnx")) / 1e6:.1f} MB')


def reference(name, samples):
    model = load_model(name, framehop_prop=1, initialize=False)
    # One chunk covering the whole clip, sized the way the analyzer sizes it.
    model.samples_session = model.session_length(len(samples) / model.samplerate)
    model.initialize()
    activations = model.predict(samples)
    return {
        'classes': model.config['classes'],
        'activations': [[round(float(v), 7) for v in row] for row in activations],
        'frameCounts': {str(n): model.n_frames(n) for n in LENGTHS},
    }


def main():
    with open(PATH_CATALOG) as f:
        catalog = json.load(f)
    for name in MODELS:
        copy_model(name, catalog)

    samples, rate = sf.read(PATH_SAMPLE, dtype='float32')
    if rate != 16000 or samples.ndim != 1:
        raise SystemExit(f'{PATH_SAMPLE} must be 16 kHz mono, got {rate} Hz, shape {samples.shape}')

    with open(PATH_REFERENCE, 'w') as f:
        json.dump({
            'source': 'testbuzz.wav',
            'note': "activations from buzzdetect's OnnxModel, CPU, fp32, one chunk",
            'samples': int(len(samples)),
            'models': {name: reference(name, samples) for name in MODELS},
        }, f)
    print(f'{PATH_REFERENCE}  {os.path.getsize(PATH_REFERENCE) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
