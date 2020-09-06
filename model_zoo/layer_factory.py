import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Conv3D, Pool2D, BatchNorm, Linear, Dropout


LAYER_BUILDER_DICT=dict()


def parse_expr(expr):
    parts = expr.split('<=')
    return parts[0].split(','), parts[1], parts[2].split(',')


def get_basic_layer(info, channels=None, conv_bias=False, num_segments=4):
    id = info['id']

    attr = info['attrs'] if 'attrs' in info else dict()
    if 'kernel_d' in attr.keys():
        if isinstance(attr["kernel_d"], str):
            div_num = int(attr["kernel_d"].split("/")[-1])
            attr['kernel_d'] = int(num_segments / div_num)

    out, op, in_vars = parse_expr(info['expr'])
    assert(len(out) == 1)
    assert(len(in_vars) == 1)
    mod, out_channel, = LAYER_BUILDER_DICT[op](attr, channels, conv_bias)

    return id, out[0], mod, out_channel, in_vars[0]


def build_conv(attr, channels=None, conv_bias=False):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_h'], attr['kernel_w'])
    if 'pad' in attr or 'pad_w' in attr and 'pad_h' in attr:
        padding = attr['pad'] if 'pad' in attr else (attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if 'stride' in attr or 'stride_w' in attr and 'stride_h' in attr:
        stride = attr['stride'] if 'stride' in attr else (attr['stride_h'], attr['stride_w'])
    else:
        stride = 1

    conv = Conv2D(channels, out_channels, ks, stride, padding,
    param_attr=fluid.ParamAttr(learning_rate=1.0), bias_attr=fluid.ParamAttr(learning_rate=2.0))

    return conv, out_channels


def build_pooling(attr, channels=None, conv_bias=False):
    method = attr['mode']
    pad = attr['pad'] if 'pad' in attr else 0
    if method == 'max':
        pool = Pool2D(attr['kernel_size'], 'max', attr['stride'], pad,
                            ceil_mode=True) # all Caffe pooling use ceil model
    elif method == 'ave':
        pool = Pool2D(attr['kernel_size'], 'avg', attr['stride'], pad,
                            ceil_mode=True)  # all Caffe pooling use ceil model
    else:
        raise ValueError("Unknown pooling method: {}".format(method))

    return pool, channels


def build_relu(attr, channels=None, conv_bias=False):
    return None, channels


def build_bn(attr, channels=None, conv_bias=False):
    bn = BatchNorm(channels, param_attr=fluid.ParamAttr(learning_rate=1.0), 
    bias_attr=fluid.ParamAttr(learning_rate=1.0))
    return bn, channels


def build_linear(attr, channels=None, conv_bias=False):
    linear = Linear(channels, attr['num_output'], param_attr=fluid.ParamAttr(learning_rate=1.0), 
    bias_attr=fluid.ParamAttr(learning_rate=2.0))
    return linear, attr['num_output']


def build_dropout(attr, channels=None, conv_bias=False):
    return Dropout(p=attr['dropout_ratio']), channels

def build_conv3d(attr, channels=None, conv_bias=False):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_d'], attr['kernel_h'], attr['kernel_w'])
    if ('pad' in attr) or ('pad_d' in attr and 'pad_w' in attr and 'pad_h' in attr):
        padding = attr['pad'] if 'pad' in attr else (attr['pad_d'], attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if ('stride' in attr) or ('stride_d' in attr and 'stride_w' in attr and 'stride_h' in attr):
        stride = attr['stride'] if 'stride' in attr else (attr['stride_d'], attr['stride_h'], attr['stride_w'])
    else:
        stride = 1

    conv = Conv3D(channels, out_channels, ks, stride, padding,
    param_attr=fluid.ParamAttr(learning_rate=5.0), bias_attr=fluid.ParamAttr(learning_rate=10.0))

    return conv, out_channels

def build_pooling3d(attr, channels=None, conv_bias=False):
    method = attr['mode']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_d'], attr['kernel_h'], attr['kernel_w'])
    if ('pad' in attr) or ('pad_d' in attr and 'pad_w' in attr and 'pad_h' in attr):
        padding = attr['pad'] if 'pad' in attr else (attr['pad_d'], attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if ('stride' in attr) or ('stride_d' in attr and 'stride_w' in attr and 'stride_h' in attr):
        stride = attr['stride'] if 'stride' in attr else (attr['stride_d'], attr['stride_h'], attr['stride_w'])
    else:
        stride = 1

    return (method, ks, stride, padding), channels

def build_bn3d(attr, channels=None, conv_bias=False):
    bn = BatchNorm(channels, param_attr=fluid.ParamAttr(learning_rate=1.0), 
    bias_attr=fluid.ParamAttr(learning_rate=1.0))
    return bn, channels


LAYER_BUILDER_DICT['Convolution'] = build_conv

LAYER_BUILDER_DICT['Pooling'] = build_pooling

LAYER_BUILDER_DICT['ReLU'] = build_relu

LAYER_BUILDER_DICT['Dropout'] = build_dropout

LAYER_BUILDER_DICT['BN'] = build_bn

LAYER_BUILDER_DICT['InnerProduct'] = build_linear

LAYER_BUILDER_DICT['Conv3d'] = build_conv3d

LAYER_BUILDER_DICT['Pooling3d'] = build_pooling3d

LAYER_BUILDER_DICT['BN3d'] = build_bn3d

