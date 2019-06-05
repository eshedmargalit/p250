"""
Defines CNN architectures
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tfutils import model_tool


def alexnet(images, train=True, norm=True, seed=0, **kwargs):
    """
    Alexnet
    """
    m = model_tool.ConvNet(seed=seed)

    conv_kwargs = {"add_bn": False, "init": "xavier", "weight_decay": 0.0001}
    pool_kwargs = {"pool_type": "maxpool"}
    fc_kwargs = {"init": "trunc_norm", "weight_decay": 0.0001, "stddev": 0.01}

    dropout = 0.5 if train else None

    m.conv(96, 11, 4, padding="VALID", layer="conv1", in_layer=images, **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=0.0001, beta=0.75, layer="lrn1")
    m.pool(3, 2, layer="pool1", **pool_kwargs)

    m.conv(256, 5, 1, layer="conv2", **conv_kwargs)
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=0.0001, beta=0.75, layer="lrn2")
    m.pool(3, 2, layer="pool2", **pool_kwargs)

    m.conv(384, 3, 1, layer="conv3", **conv_kwargs)
    m.conv(384, 3, 1, layer="conv4", **conv_kwargs)

    m.conv(256, 3, 1, layer="conv5", **conv_kwargs)
    m.pool(3, 2, layer="pool5", **pool_kwargs)

    m.fc(4096, dropout=dropout, bias=0.1, layer="fc6", **fc_kwargs)
    m.fc(4096, dropout=dropout, bias=0.1, layer="fc7", **fc_kwargs)
    m.fc(1000, activation=None, dropout=None, bias=0, layer="fc8", **fc_kwargs)

    return m


def alexnet_wrapper(inputs, **kwargs):
    """
    tfutils wrapper around alexnet
    """
    outputs = {}
    if isinstance(inputs, dict):
        images = inputs["images"]
    else:
        images = inputs
    m = alexnet(images, **kwargs)

    logit_key = kwargs.get("logit_key", "logits")
    outputs[logit_key] = m.output

    return_params = kwargs.get("return_params")
    if return_params:
        return outputs, m.params

    return outputs["logits"]


def vgg_19_wrapper(inputs, **kwargs):
    """
    tfutils wrapper around vgg_19 in slim
    """
    model_fn = nets.vgg.vgg_19

    outputs = {}
    if isinstance(inputs, dict):
        images = inputs["images"]
    else:
        images = inputs

    with slim.arg_scope(nets.vgg.vgg_arg_scope()):
        net, endpoints = model_fn(
            images, num_classes=kwargs["num_classes"], spatial_squeeze=True
        )

    # ipdb.set_trace()
    logit_key = kwargs.get("logit_key", "logits")
    endpoint_key = "model_0/__var_copy_0/vgg_19/fc8"
    outputs[logit_key] = endpoints[endpoint_key]

    return_params = kwargs.get("return_params")
    if return_params:
        params = {"cfg_final": kwargs["cfg_final"]}
        return outputs, params

    return outputs[logit_key]


def tnn_model_func(inputs, **kwargs):
    if isinstance(inputs, dict):
        images = inputs["images"]
    else:
        images = inputs

    base_name = kwargs["json_fpath"]
    ims = tf.identity(tf.cast(images, dtype=tf.float32), name="split")
    batch_size = kwargs["batch_size"]

    with tf.variable_scope("tnn_model"):
        if ".json" not in base_name:
            base_name += ".json"

        G = tnn_main.graph_from_json(base_name)

        # initialize graph structure
        tnn_main.init_nodes(
            G, input_nodes=["conv1"], batch_size=batch_size, channel_op="concat"
        )

        # unroll graph
        tnn_main.unroll_tf(G, input_seq={"conv1": ims}, ntimes=1)

        outputs = {}
        for l in ["conv" + str(t) for t in range(1, 11)]:
            outputs[l] = G.node[l]["outputs"][-1]

        outputs["logits"] = G.node["imnetds"]["outputs"][-1]

    return outputs
