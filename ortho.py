import tensorflow as tf
import numpy as np

def get_ortho_weights(var, gain):
    ''' compute the orthogonal initialization, this is only an approximate '''
    num_rows = 1
    for dim in var.shape[:-1]:
      num_rows *= dim
    num_cols = var.shape[-1]
    flat_shape = (num_cols, num_rows) if num_rows < num_cols else (num_rows, num_cols)
    a = tf.reshape(tf.nn.l2_normalize(var), flat_shape)
    # using svd would be better approximation but tf.qr seems faster
    q, r = tf.qr(a, full_matrices=False)
    d = tf.diag_part(r)
    q *= tf.sign(d)
    if num_rows < num_cols:
      q = tf.matrix_transpose(q)
    # gain is used to scale the new weights, needed for deeper networks
    return tf.reshape(gain*q, var.shape)
