"""
Runs inside the `buzzdetect` conda env, from the buzzdetect repo root.

Pulls the raw layer weights out of yamnet.keras and model_general_v3 and writes
them to a plain .npz + .json spec, so the ONNX builder needs nothing but numpy.
No arithmetic happens here -- BatchNorm folding is the builder's job, and is
checked against these untouched values.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

import tensorflow as tf
from embedders.yamnet.yamnet import WaveformFeatures  # noqa: F401

DIR_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')


def main():
    km = tf.keras.models.load_model('embedders/yamnet/yamnet.keras', compile=False)

    spec = []
    arrays = {}

    # layers[0] = waveform input, [1] = WaveformFeatures, [2] = reshape to (96,64,1).
    # Everything from [3] on is the conv stack we want, and it is strictly
    # sequential, so a straight walk reproduces the graph.
    for layer in km.layers[3:]:
        kind = type(layer).__name__
        name = layer.name

        if kind in ('Conv2D', 'DepthwiseConv2D'):
            w = layer.get_weights()
            assert len(w) == 1, f'{name}: expected a bias-free conv, got {len(w)} weights'
            arrays[f'{name}/kernel'] = w[0]
            spec.append({
                'op': kind, 'name': name,
                'strides': list(layer.strides),
                'padding': layer.padding,
                'kernel': list(w[0].shape),
            })
        elif kind == 'BatchNormalization':
            assert layer.scale is False, f'{name}: gamma present, folding assumes scale=False'
            assert layer.center is True, f'{name}: no beta'
            beta, mean, var = layer.get_weights()
            arrays[f'{name}/beta'] = beta
            arrays[f'{name}/mean'] = mean
            arrays[f'{name}/var'] = var
            spec.append({'op': 'BatchNorm', 'name': name, 'epsilon': float(layer.epsilon)})
        elif kind == 'ReLU':
            spec.append({'op': 'ReLU', 'name': name})
        elif kind == 'GlobalAveragePooling2D':
            spec.append({'op': 'GlobalAvgPool', 'name': name})
        else:
            raise RuntimeError(f'unhandled layer {name} ({kind})')

    # the buzzdetect head: one bias-ed Dense, no activation
    head = tf.saved_model.load('models/model_general_v3')
    by_name = {v.name: v.numpy() for v in head.variables}
    arrays['head/kernel'] = by_name['dense/kernel:0']
    arrays['head/bias'] = by_name['dense/bias:0']
    spec.append({'op': 'Dense', 'name': 'head',
                 'kernel': list(arrays['head/kernel'].shape)})

    with open('models/model_general_v3/config_model.json') as f:
        classes = json.load(f)['classes']

    os.makedirs(DIR_OUT, exist_ok=True)
    np.savez(os.path.join(DIR_OUT, 'weights.npz'), **arrays)
    with open(os.path.join(DIR_OUT, 'spec.json'), 'w') as f:
        json.dump({'layers': spec, 'classes': classes}, f, indent=1)

    n_params = sum(a.size for a in arrays.values())
    print(f'{len(spec)} ops, {n_params:,} parameters ({n_params * 4 / 1e6:.1f} MB as fp32)')
    for s in spec[:4]:
        print(' ', s)


if __name__ == '__main__':
    main()
