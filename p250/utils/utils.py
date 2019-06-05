"""
Utility functions written in TensorFlow
"""
import tensorflow as tf

def loss_and_in_top_k(inputs, outputs, target):
    return {
        "loss": tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=outputs["logits"], labels=inputs[target]
        ),
        "top1": tf.nn.in_top_k(outputs["logits"], inputs[target], 1),
        "top5": tf.nn.in_top_k(outputs["logits"], inputs[target], 5),
    }
