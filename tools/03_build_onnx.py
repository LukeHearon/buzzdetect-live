"""
Builds one ONNX graph: log-mel patches -> 13 buzzdetect activations.

YAMNet's conv stack and yamnet_large_general's dense head are fused into a single
file so the browser loads one model and pays one inference call per batch.

Two size/speed reductions happen here, both exact or near-exact:

  * BatchNorm folding. Every BN in YAMNet has scale=False, so it is
    y = (x - mean)/sqrt(var + eps) + beta, an affine map that absorbs into the
    preceding bias-free convolution: W' = W * s, b' = beta - mean * s, where
    s = 1/sqrt(var + eps). This removes 27 ops from the graph and costs nothing
    in accuracy beyond float rounding.

  * Explicit padding. Patch shape is always 96x64, so every spatial dimension is
    static and TF's asymmetric SAME padding can be resolved to literal pad
    values at build time. Avoids relying on auto_pad, which some ONNX Runtime
    Web kernels handle on a slower path.

The graph keeps a dynamic batch dimension, so one session serves both the
file-analysis batches and the single-patch live microphone calls.
"""
import argparse
import json
import math
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
PATCH_FRAMES, MEL_BANDS = 96, 64
OPSET = 13


def same_pads(in_h, in_w, k_h, k_w, s_h, s_w):
    """TF 'SAME' padding resolved to explicit ONNX pads [t, l, b, r].

    TF puts the extra pixel at the end when the total padding is odd, which is
    ONNX's SAME_UPPER convention.
    """
    out_h, out_w = math.ceil(in_h / s_h), math.ceil(in_w / s_w)
    pad_h = max((out_h - 1) * s_h + k_h - in_h, 0)
    pad_w = max((out_w - 1) * s_w + k_w - in_w, 0)
    return [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2], out_h, out_w


def quantize_per_channel(w):
    """Symmetric int8 quantization of a conv kernel, one scale per output channel.

    Per-channel is not optional here: YAMNet is a MobileNet, and its depthwise
    kernels have channels whose magnitudes differ by orders of magnitude. A
    single tensor-wide scale collapses the small ones to zero.
    """
    absmax = np.abs(w.reshape(w.shape[0], -1)).max(axis=1)
    scale = np.maximum(absmax / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.rint(w / scale.reshape(-1, 1, 1, 1)), -127, 127).astype(np.int8)
    return q, scale


class Builder:
    def __init__(self):
        self.nodes = []
        self.inits = []

    def const(self, name, array):
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    def node(self, op, ins, outs, **attrs):
        self.nodes.append(helper.make_node(op, ins, outs, **attrs))
        return outs[0]


def fold_bn(kernel_oihw, beta, mean, var, eps):
    """Absorb a scale=False BatchNorm into the conv that feeds it."""
    scale = 1.0 / np.sqrt(var + eps)
    w = kernel_oihw * scale.reshape(-1, 1, 1, 1)
    b = beta - mean * scale
    return w.astype(np.float32), b.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--precision', choices=('fp32', 'fp16', 'int8', 'mixed'), default='fp32',
                    help='how conv kernels are STORED; compute stays fp32 either way')
    ap.add_argument('--mixed-from', type=int, default=8,
                    help="with --precision mixed: first block index stored as fp16")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    quantize = args.precision == 'int8'

    w = np.load(os.path.join(DIR, 'weights.npz'))
    with open(os.path.join(DIR, 'spec.json')) as f:
        spec = json.load(f)
    layers, classes = spec['layers'], spec['classes']

    b = Builder()

    # [N, 96, 64] -> [N, 1, 96, 64]; a single channel, so this is a free reshape
    b.const('shape_nchw', np.array([-1, 1, PATCH_FRAMES, MEL_BANDS], dtype=np.int64))
    x = b.node('Reshape', ['patches', 'shape_nchw'], ['x0'])

    h, wd = PATCH_FRAMES, MEL_BANDS
    i = 0
    n_folded = 0
    n_conv = 0
    n_half = 0
    while i < len(layers):
        L = layers[i]

        if L['op'] in ('Conv2D', 'DepthwiseConv2D'):
            bn = layers[i + 1]
            assert bn['op'] == 'BatchNorm', f"{L['name']} is not followed by BatchNorm"
            assert layers[i + 2]['op'] == 'ReLU', f"{L['name']} block has no ReLU"

            k = w[f"{L['name']}/kernel"]
            if L['op'] == 'Conv2D':
                # Keras HWIO -> ONNX OIHW
                kern = k.transpose(3, 2, 0, 1)
                group = 1
            else:
                # Keras depthwise HWCM (M=1) -> ONNX grouped OIHW with O = C
                assert k.shape[3] == 1, 'depth_multiplier != 1 not handled'
                kern = k.transpose(2, 3, 0, 1)
                group = k.shape[2]

            kern, bias = fold_bn(kern, w[f"{bn['name']}/beta"], w[f"{bn['name']}/mean"],
                                 w[f"{bn['name']}/var"], bn['epsilon'])
            n_folded += 1

            s_h, s_w = L['strides']
            k_h, k_w = kern.shape[2], kern.shape[3]
            pads, h, wd = same_pads(h, wd, k_h, k_w, s_h, s_w)

            name = L['name']
            # Rounding error introduced at a conv is amplified by every conv after
            # it, so the cheap half-precision wins are at the END of the stack --
            # which is also where the parameters are (the 512/1024-channel
            # pointwise convs hold most of the model). 'mixed' takes that trade:
            # fp16 from block --mixed-from onward, fp32 before it.
            half = (args.precision == 'fp16'
                    or (args.precision == 'mixed' and n_conv >= args.mixed_from))
            n_conv += 1
            if half:
                n_half += 1
                # Weights travel as float16 and are cast back to float32 by a
                # Cast on constant input, which ONNX Runtime folds at load time.
                # Half the download, unchanged inference cost, and a RELATIVE
                # rounding error (~5e-4) rather than int8's absolute one -- which
                # matters through 27 MobileNet layers, see step 05.
                b.const(f'{name}_Wh', kern.astype(np.float16))
                b.node('Cast', [f'{name}_Wh'], [f'{name}_W'], to=int(TensorProto.FLOAT))
            elif quantize:
                # Weight-only quantization: the kernel travels as int8 and is
                # expanded back to float by a DequantizeLinear on constant input,
                # which ONNX Runtime constant-folds at session load. So this buys
                # a ~4x smaller download at no inference cost and no change to
                # the arithmetic beyond the weight rounding measured in step 05.
                q, scale = quantize_per_channel(kern)
                b.const(f'{name}_Wq', q)
                b.const(f'{name}_Ws', scale)
                b.node('DequantizeLinear', [f'{name}_Wq', f'{name}_Ws'], [f'{name}_W'], axis=0)
            else:
                b.const(f'{name}_W', kern)
            b.const(f'{name}_B', bias)
            x = b.node('Conv', [x, f'{name}_W', f'{name}_B'], [f'{name}_out'],
                       kernel_shape=[k_h, k_w], strides=[s_h, s_w], pads=pads, group=group)
            x = b.node('Relu', [x], [f'{name}_relu'])
            i += 3

        elif L['op'] == 'GlobalAvgPool':
            # ReduceMean over H and W straight to [N, 1024]; cheaper than
            # GlobalAveragePool + Flatten and one node fewer.
            x = b.node('ReduceMean', [x], ['embeddings'], axes=[2, 3], keepdims=0)
            i += 1

        elif L['op'] == 'Dense':
            b.const('head_W', w['head/kernel'].astype(np.float32))
            b.const('head_B', w['head/bias'].astype(np.float32))
            x = b.node('Gemm', [x, 'head_W', 'head_B'], ['activations'])
            i += 1

        else:
            raise RuntimeError(f"unexpected op {L['op']}")

    inp = helper.make_tensor_value_info(
        'patches', TensorProto.FLOAT, ['batch', PATCH_FRAMES, MEL_BANDS])
    out = helper.make_tensor_value_info('activations', TensorProto.FLOAT, ['batch', len(classes)])
    emb = helper.make_tensor_value_info('embeddings', TensorProto.FLOAT, ['batch', 1024])

    graph = helper.make_graph(b.nodes, 'buzzdetect_general_v3', [inp], [out, emb], b.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)],
                              producer_name='buzzdetect-web')
    model.doc_string = ('YAMNet conv stack (BatchNorm folded) + yamnet_large_general dense head. '
                        'Input: log-mel patches [N,96,64]. Output: 13 class activations.')
    onnx.checker.check_model(model)

    path = os.path.join(DIR, args.out or f'buzzdetect_v3_{args.precision}.onnx')
    onnx.save(model, path)
    print(f'{args.precision}: {len(b.nodes)} nodes ({n_folded} BatchNorms folded away, '
          f'{n_half}/{n_conv} convs stored half-precision), '
          f'{os.path.getsize(path) / 1e6:.2f} MB')
    print('classes:', ', '.join(classes))


if __name__ == '__main__':
    main()
