"""
Runs inside the `buzzdetect` conda env, from the buzzdetect repo root.

Turns a reference .npz into flat binary fixtures the TypeScript tests read, so
the browser front end is checked against TensorFlow's own numbers rather than
against a second Python reimplementation of them.

Also dumps the dense mel filterbank straight from
`tf.signal.linear_to_mel_weight_matrix`, which localises failures: if the
filterbank matches but the patches do not, the bug is in the STFT, not the mel
projection.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import tensorflow as tf

from embedders.yamnet.params import Params

DIR_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
DIR_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'test', 'fixtures')


def write(name, arr, dtype=np.float32):
    path = os.path.join(DIR_OUT, name)
    arr.astype(dtype).ravel().tofile(path)
    return os.path.getsize(path)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'fixture10s'
    ref = np.load(os.path.join(DIR_ART, f'{tag}.npz'))
    os.makedirs(DIR_OUT, exist_ok=True)

    p = Params()
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=p.mel_bands,
        num_spectrogram_bins=(2 ** int(np.ceil(np.log2(
            round(p.sample_rate * p.stft_window_seconds))))) // 2 + 1,
        sample_rate=p.sample_rate,
        lower_edge_hertz=p.mel_min_hz,
        upper_edge_hertz=p.mel_max_hz).numpy()

    total = 0
    total += write('waveform.f32', ref['waveform'])
    total += write('patches.f32', ref['patches'])
    total += write('activations.f32', ref['logits'])
    total += write('mel_matrix.f32', mel_matrix)

    meta = {
        'source': str(ref['source']),
        'framehopProp': float(ref['framehop_prop']),
        'sampleRate': 16000,
        'numSamples': int(ref['waveform'].shape[0]),
        'numPatches': int(ref['patches'].shape[0]),
        'patchShape': list(map(int, ref['patches'].shape[1:])),
        'melMatrixShape': list(mel_matrix.shape),
        'classes': [str(c) for c in ref['classes']],
    }
    with open(os.path.join(DIR_OUT, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    print(f'wrote fixtures to {DIR_OUT} ({total / 1e6:.2f} MB)')
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
