"""
Writes the demo clip the web app ships with, plus the reference activations for it.

The clip is written as 32-bit float WAV at 16 kHz mono -- the exact array
buzzdetect fed its embedder, straight out of the reference run. Because the
browser decodes it without resampling or lossy decoding, the web app's inputs
are bit-identical to buzzdetect's, so any difference in the activations is the
model export alone. That is what makes public/sample usable as an in-browser
parity check (test/browser-parity.html) and not just a demo file.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

DIR_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_OUT = os.path.join(ROOT, 'public', 'sample')


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'fixture10s'
    ref = np.load(os.path.join(DIR_ART, f'{tag}.npz'))
    os.makedirs(DIR_OUT, exist_ok=True)

    wav_path = os.path.join(DIR_OUT, 'testbuzz.wav')
    sf.write(wav_path, ref['waveform'], 16000, subtype='FLOAT')

    ref_path = os.path.join(DIR_OUT, 'testbuzz.reference.json')
    with open(ref_path, 'w') as f:
        json.dump({
            'source': os.path.basename(str(ref['source'])),
            'note': 'activations from buzzdetect yamnet_large_general via TensorFlow',
            'framehopProp': float(ref['framehop_prop']),
            'classes': [str(c) for c in ref['classes']],
            'activations': [[round(float(v), 6) for v in row] for row in ref['logits']],
        }, f)

    print(f"{wav_path}  {os.path.getsize(wav_path) / 1e6:.2f} MB")
    print(f"{ref_path}  {os.path.getsize(ref_path) / 1e3:.0f} kB, "
          f"{ref['logits'].shape[0]} frames")


if __name__ == '__main__':
    main()
