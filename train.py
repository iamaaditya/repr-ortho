from __future__ import print_function, division
import os, sys, argparse
import tensorflow as tf
import numpy as np
import random
random.seed(1337)
tf.set_random_seed(1337)
np.random.seed(1337)
from util import *
from ranking import *
from ortho import *
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--standard', dest='standard', action='store_true', help='Train standard instead of RePr')
parser.add_argument('-learning_rate', type=float, default=0.05, help='Initial learning Rate for SGD')
parser.add_argument('-epochs', type=int, default=200, help='Number of epochs to train')
parser.add_argument('-tuner', type=int, default=20, help='Number of epochs to train the sub-network')
parser.add_argument('-freq', type=int, default=20, help='Number of epochs to train the full-network')
parser.add_argument('-batch_size', type=int, default=64, help='Batch size')
parser.add_argument('-rank', type=int, default=50, help='Percentage of filters to drop')
parser.add_argument('-gain', type=float, default=1.00, help='Multiplier for the ortho initializer')
args = parser.parse_args()
print(args)


path = 'data/'
nb_classes = 10
X, X_test = np.load(path+'x.npy'), np.load(path+'x_test.npy')
Y, Y_test = np.load(path+'y.npy'), np.load(path+'y_test.npy')
X_train, X_val = X[:45000], X[45000:]
Y_train, Y_val = Y[:45000], Y[45000:]
# X_train, X_val = X, X
# Y_train, Y_val = Y, Y
# validation split was used to run the oracle ranking
train_indices   = list(xrange(0,len(X_train)))
test_indices    = list(xrange(0,len(X_test)))
val_indices     = list(xrange(0,len(X_val)))

x         = tf.placeholder("float", [None, 32, 32, 3])
y         = tf.placeholder("float", [None, nb_classes])
logits, conv_activations  = convnet(x, nb_classes)

def test(sorted_vars):
    ''' single epoch over the test data '''
    test_epoch_acc   = []
    beta_dict = get_feed_dict(sorted_vars)
    for ind in chunker(test_indices, args.batch_size):
        feed_dict = {x: X_test[ind], y: Y_test[ind]}
        test_epoch_acc.append(accuracy.eval(dict(feed_dict.items() + beta_dict.items())))
    return avg(test_epoch_acc)

def train(epoch, sorted_vars):
    ''' single epoch for the training '''
    random.shuffle(train_indices)
    train_epoch_cost = []
    train_epoch_acc  = []
    var_list = tf.trainable_variables()
    # beta is the binary mask which is used to turn on/off the filters 
    beta_dict = get_feed_dict(sorted_vars)
    for ind in chunker(train_indices, args.batch_size):
        if len(ind) != args.batch_size: continue
        feed_dict={x: X_train[ind], y: Y_train[ind]}
        _, lgt, lr, c,acc = sess.run([optimizer, logits, learning_rate, cost, accuracy], 
                feed_dict=dict(feed_dict.items() + beta_dict.items()))
        train_epoch_cost.append(c)
        train_epoch_acc.append(acc)
    return avg(train_epoch_cost), avg(train_epoch_acc), lr

def one_epoch(epoch, sorted_vars):
    ''' wrapper for a single epoch of training '''
    train_cost, train_acc, lr = train(epoch, sorted_vars)
    print("Epoch:{:3d} lr={:.3f} cost={:.4f} Training accuracy={:.3f} Test accuracy={:.3f}"
            .format(epoch,lr, train_cost, train_acc, test(sorted_vars)))

def prune_with_train():
    ''' RePr training which trains the sub-network '''
    sorted_vars = rank(args.rank)
    # rank function returns a binary mask
    for iepoch in xrange(args.tuner):
        one_epoch(iepoch, sorted_vars)
    # after doing the sub-network training, re-initialize the filters
    # to be orthogoanl (approximately)
    def reinitialize_pruned():
        ops = []
        for weights, filters in sorted_vars:
        # sorted_vars maintains the filters that needs to be dropped
            c = get_ortho_weights(weights, args.gain)[:,:,:,filters]
            ops.append(weights[:,:,:,filters].assign(c))
        sess.run(ops)
    reinitialize_pruned()

with tf.Session() as sess:
    cost      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
    prediction= tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy  = tf.reduce_mean(tf.cast(prediction, tf.float32))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay( args.learning_rate, global_step, 25000, 0.50, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)
    sess.run(tf.global_variables_initializer())

    for epoch in range(1,args.epochs):
        # train the full-network
        one_epoch(epoch, sorted_vars=[])
        if epoch % args.freq == 0 and epoch >= 1:
            if args.rank and not args.standard:
                # train the sub-network with pruning
                prune_with_train()
            else:
                # if running standard mode, do normal training instead
                for iepoch in xrange(args.tuner):
                    one_epoch(iepoch, sorted_vars=[])
