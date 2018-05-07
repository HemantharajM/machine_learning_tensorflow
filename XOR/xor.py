#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:27:30 2018

@author: Hemantharaj M
Tensorflow code using deep learning concept to implement xor logic gate.
Two input node
one output node
1 hidden layer with two node
"""
import tensorflow as tf

tf.reset_default_graph()
data = [[1],[0],[0]]
#Initialize the target and label

X = tf.Variable([[1.0],[0.0],[1.0]],tf.float32, name='X')
Y = tf.Variable([[1.0]],tf.float32, name='Y')

#initialize the weight and finding intermediate node

W1 = tf.Variable(tf.random_normal([2,3]), name='first_layer_weights')
W2 = tf.Variable(tf.random_normal([1,3]), name='second_layer_weights')

Delta2 = tf.Variable(tf.zeros([3,1],tf.float32))
Delta3 = tf.Variable(tf.zeros([3,1], tf.float32))

z2 = tf.matmul(W1, X)
a2_bias = tf.sigmoid(z2, name='activation_first_layer')
a2 = tf.concat([[[1]], a2_bias],0)
z3 = tf.matmul(W2, a2)
a3 = tf.sigmoid(z3, name='activation_second')

#backpropagation value
delta3 = tf.subtract(a3, Y)
del2 = tf.matmul(tf.transpose(W2), delta3)
g2 = tf.multiply(a2, tf.subtract(1.0,a2))
delta2 = tf.multiply(del2, g2)

delta1_assign = Delta2.assign_add(tf.multiply(a2, delta2))
delta3_assign = Delta3.assign_add(tf.multiply(a3, delta3))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a3))
    print(W2.eval())
    print(W1.eval())
    print(sess.run(delta3))
    print(sess.run(delta2))
    print(Delta3.eval())
    print(Delta2.eval())
