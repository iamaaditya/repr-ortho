import time
import tensorflow as tf

def get_scope_variable(scope, var, shape=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        v = tf.get_variable(var, shape)
    return v

def avg(l):
    return sum(l)/len(l)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

def measure_time(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print '%r (%r, %r) %2.2f sec' % \
              (f.__name__, args, kw, te-ts)
        return result
    return timed

def replace_none_with_zero(l):
    return [0 if i==None else i for i in l]
