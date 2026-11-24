"""The JZDL graph of our detector, built from the network definition (no template file).

The persondet topology as a JZDL v5 model: layer types, wiring (bottom/top blob ids) and the
structural parameters of each layer (kernel, stride, padding, bit widths, blob sizes). Weight
blobs are left empty; export.py fills them from a trained Detector. This makes the exporter
independent of Ingenic's persondet file, so a model trained from scratch contains nothing of it.

Layer parameter layouts (see docs/jzdl-format.md): for conv types the fields are
  [cout, kh, kw, sh?, sw?, stride_h, stride_w, f7, pad_t, pad_b, pad_l, pad_r, weight_bytes,
   f13, bits_in, bits_out, dtype(20), f17]
with type-specific variants below. The constants (dtype 20, the first layer's f7 = -233 and
14 = 4, 15 = 8) are format values the runtime expects for these layer types.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'jzdl'))
import jzdl  # noqa: E402

DT = 20          # activation data type code used by every layer of this graph


def _conv_params(conv):
    c = conv.cout
    if conv.kind == 'first':
        return 46, [c, 3, 3, 1, 1, 2, 2, -233, 0, 1, 0, 1, 0, 0, 4, 8, conv.bits_out, DT, 1]
    if conv.kind == 'dw':
        return 73, [c, 3, 3, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, conv.bits_in, conv.bits_out, DT, 1]
    if conv.kind == 'pwpool':
        return 74, [c, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, conv.bits_in, conv.bits_out, DT, 1]
    if conv.kind == 'pw':
        return 53, [c, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, conv.bits_in, conv.bits_out, DT, 1]
    return 51, [c, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, conv.bits_in, 32, 21, 0]       # head (float out)


def skeleton(net, in_h, in_w):
    """jzdl.Model for net (a qat.net.Detector) with empty weight blobs; layer order = LAYER_INDEX."""
    m = jzdl.Model()
    m.version, m.input_blob, m.output_blob = 5, 0, 29
    m.in_h, m.in_w, m.in_c = in_h, in_w, 3
    m.blob_count = 30
    L = net.L

    def conv(name, bottom, top):
        t, params = _conv_params(L[name])
        return jzdl.Layer(t, [bottom], [top], params)

    m.layers = [
        jzdl.Layer(16, [], [0], [3, 67, 67]),                                   # input
        conv('l1', 0, 1), conv('l2', 1, 2), conv('l3', 2, 3), conv('l4', 3, 4), conv('l5', 4, 5),
        conv('l6', 5, 6), conv('l7', 6, 7), conv('l8', 7, 8), conv('l9', 8, 9),
        jzdl.Layer(33, [9], [10, 11], [DT]),                                    # split (b9)
        jzdl.Layer(75, [10], [12], [0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 4, DT]),      # maxpool 2x2
        conv('l12', 12, 13), conv('l13', 13, 14), conv('l14', 14, 15), conv('l15', 15, 16),
        conv('l16', 16, 17),
        jzdl.Layer(33, [17], [18, 19], [DT]),                                   # split (b17)
        conv('l18', 19, 20), conv('l19', 20, 21), conv('l20', 21, 22),           # stride-32 head
        conv('l21', 18, 23),
        jzdl.Layer(69, [23], [24], [DT, 4, 2, 2, 0]),                           # zero-insert upsample x2
        jzdl.Layer(71, [24, 11], [25], [DT]),                                   # concat
        conv('l24', 25, 26), conv('l25', 26, 27), conv('l26', 27, 28),           # stride-16 head
        jzdl.Layer(76, [22, 28], [29], []),                                     # output (both heads)
    ]
    return m
