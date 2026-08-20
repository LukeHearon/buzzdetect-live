"""
Runs inside the `buzzdetect` conda env, from the buzzdetect repo root.

Produces the ground truth every later step is checked against:
  - the exact 16 kHz waveform buzzdetect feeds its embedder
  - YAMNet's log-mel patches for that waveform
  - the 1024-d embeddings from the REAL embedder buzzdetect runs
  - the activations from the REAL head, `--model` (models/<model>/)

Two embedders are supported, distinguished by `embeddername` in the model's
own config_model.json:
  - 'yamnet_k2': a separate SavedModel. Embeddings come from it directly, and
    this also asserts that embedders/yamnet/yamnet.keras (whose layer weights
    we actually export to ONNX) is numerically the same network -- if that
    assertion fails, the ONNX export is invalid.
  - 'yamnet': the model IS embedders/yamnet/yamnet.keras, so its embeddings
    are used as-is with nothing to cross-check.

The head itself is loaded either as a Keras single-file model (models/<model>/
model.keras) or, for older models, a SavedModel directory.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

import librosa
import soundfile as sf
import tensorflow as tf
import keras

from embedders.yamnet.params import Params
import embedders.yamnet.features as features_lib
from embedders.yamnet.yamnet import WaveformFeatures  # noqa: F401  (needed to deserialize yamnet.keras)

SR = 16000
DIR_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')


def load_waveform(path, seconds=None):
    """Mirror WorkerStreamer: read native, then librosa.resample to 16 kHz."""
    samples, sr_native = sf.read(path, dtype='float32', always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if seconds is not None:
        samples = samples[:int(seconds * sr_native)]
    samples = librosa.resample(y=samples, orig_sr=sr_native, target_sr=SR)
    return samples.astype(np.float32), sr_native


def load_head(model_dir):
    """A head model saved either as model.keras or an older SavedModel dir."""
    keras_path = os.path.join(model_dir, 'model.keras')
    if os.path.exists(keras_path):
        head = keras.saving.load_model(keras_path, compile=False)
        return lambda embeddings: np.asarray(head(embeddings, training=False))

    saved = tf.saved_model.load(model_dir)
    return lambda embeddings: saved.signatures['serving_default'](
        input=tf.constant(embeddings))['dense'].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('--model', default='yamnet_large_general',
                    help='directory name under models/')
    ap.add_argument('--seconds', type=float, default=None)
    ap.add_argument('--tag', default='ref')
    ap.add_argument('--framehop_prop', type=float, default=1.0,
                    help='1.0 = contiguous 0.96s frames (buzzdetect default); 0.5 = half-overlap')
    args = ap.parse_args()

    model_dir = os.path.join('models', args.model)
    with open(os.path.join(model_dir, 'config_model.json')) as f:
        model_config = json.load(f)
    embeddername = model_config['embeddername']
    classes = model_config['classes']

    # The patch hop feeds pad_waveform as well as the framing, so it has to be
    # set here rather than by subsampling patches afterwards -- a wholehop run
    # pads the tail to a 0.96s boundary and can emit one more patch than a
    # halfhop run does at every other index.
    params = Params(patch_hop_seconds=0.96 * args.framehop_prop)
    wav, sr_native = load_waveform(args.audio, args.seconds)
    print(f'{args.audio}: {sr_native} Hz native -> {len(wav)} samples @ {SR} Hz '
          f'({len(wav) / SR:.2f}s)')

    # --- features: pad + log-mel + patch framing, exactly as YAMNet does -----
    padded = features_lib.pad_waveform(tf.constant(wav), params)
    logmel, patches = features_lib.waveform_to_log_mel_spectrogram_patches(padded, params)
    logmel = logmel.numpy()
    patches = patches.numpy()          # [n_patches, 96, 64]
    print(f'log-mel {logmel.shape}, patches {patches.shape}')

    # --- the Keras model we export from, fed the SAME patches ----------------
    km = tf.keras.models.load_model('embedders/yamnet/yamnet.keras', compile=False)
    core = tf.keras.Model(inputs=km.layers[2].input, outputs=km.outputs)  # skip waveform+features
    emb_keras = core.predict(patches, verbose=0)

    if embeddername == 'yamnet_k2':
        subdir = 'yamnet_wholehop' if args.framehop_prop == 1.0 else 'yamnet_halfhop'
        k2 = tf.saved_model.load(f'embedders/yamnet_k2/models/{subdir}')
        embeddings = k2.signatures['serving_default'](
            input_1=tf.constant(wav))['global_average_pooling2d'].numpy()

        n = min(len(embeddings), len(emb_keras))
        d = np.abs(embeddings[:n] - emb_keras[:n])
        print(f'\nyamnet_k2 vs yamnet.keras embeddings: shapes {embeddings.shape} / {emb_keras.shape}')
        print(f'  max|diff| = {d.max():.3e}   mean|diff| = {d.mean():.3e}')
        assert embeddings.shape == emb_keras.shape, 'patch counts disagree'
        assert d.max() < 1e-3, 'yamnet.keras is NOT the same network as yamnet_k2'
    elif embeddername == 'yamnet':
        # The model IS embedders/yamnet/yamnet.keras -- nothing to cross-check.
        embeddings = emb_keras
    else:
        raise ValueError(f'unknown embeddername {embeddername!r}')

    logits = load_head(model_dir)(embeddings)

    os.makedirs(DIR_OUT, exist_ok=True)
    out = os.path.join(DIR_OUT, f'{args.tag}.npz')
    np.savez_compressed(
        out,
        waveform=wav, logmel=logmel, patches=patches,
        framehop_prop=np.float32(args.framehop_prop),
        embeddings=embeddings, logits=logits, classes=np.array(classes),
        source=np.array(os.path.abspath(args.audio)), sr_native=np.int64(sr_native),
    )
    print(f'\nwrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)')
    print('logits range', logits.min(), logits.max())


if __name__ == '__main__':
    main()
