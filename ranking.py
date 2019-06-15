import numpy as np
import tensorflow as tf

def conv_to_yeta(name):
    names = name.split('/')
    name = names[:-1] + ['yeta:0']
    return '/'.join(name)

def get_feed_dict(sorted_vars):
    ''' helper function to make the binary mask for the given convnet,
    uses the name of the variables to find conv filters '''
    beta_dict = {}
    for w in tf.get_collection('yeta'):
        # by default all of the filters are enabled
        filters = w.get_shape().as_list()[-1]
        beta_dict[w.name] = [1]*filters
    for weights, filter in sorted_vars:
        # this helper function converts weight name to binary masks
        yeta_name = conv_to_yeta(weights.name)
        try:
            beta_dict[yeta_name][filter]=0
        except:
            try:
                # this messy nested block is because of how tf does naming of the variables
                y = yeta_name.split('/')
                y = '/'.join([y[0], y[0]] + y[1:])
                beta_dict[y][filter]=0
            except:
                print('Not pruning {}'.format(yeta_name))
    return beta_dict

def inter_ortho(vars_to_prune, acc_fn):
    ''' inter-filter orthogonality ranking '''
    scores = {}
    for weights in vars_to_prune:
        weights_flat = tf.reshape(weights, (-1,weights.shape[-1]))
        norm_weight = tf.nn.l2_normalize(weights_flat)
        # normalizing weight is necessary for comparing different filters
        projection   = tf.matmul(norm_weight, norm_weight, transpose_a=True)
        # remove the diagonal elements as they are self-projection
        identity     = tf.diag(tf.diag_part(projection))
        ortho_proj   = tf.reduce_mean(projection-identity, axis=0)
        for filter in range(ortho_proj.get_shape().as_list()[-1]):
            # ortho aggregation is done per layer 
            v = tf.abs(ortho_proj[filter])
            scores[(weights,filter)] = float(acc_fn(v.eval()))
    return sorted(scores.iteritems(), key=lambda (k,v): v, reverse=True)


def rank(p):
    def fn(x, axis=None):
        # aggregative function composition was inspired from Molchanov paper
        f = np.mean
        g = np.abs
        if axis is not None:
            return f(g(x),axis)
        return g(f(x))

    var_name = 'weights'
    vars_to_prune=[]
    for weights in tf.trainable_variables():
        # hardcoded names are used to find conv filters to rank the filters
        if 'conv' not in weights.name.lower() or var_name not in weights.name.lower() or 'logits' in weights.name.lower():
                continue
        vars_to_prune.append(weights)
    sorted_vars = inter_ortho(vars_to_prune, fn)
    cutoff = int(len(sorted_vars)*p/100)
    # cutff-off is global across the full network
    return [s[0] for s in sorted_vars[:cutoff]]
