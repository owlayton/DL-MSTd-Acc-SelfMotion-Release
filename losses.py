'''losses.py
Custom loss functions used to train neural networks
Oliver W. Layton
Last updated: Aug 2024
'''
import math
import tensorflow as tf


class CircularLoss(tf.keras.losses.Loss):
    def __init__(self, exp):
        super().__init__()

    def call(self, y_true, y_pred):
        # y_true and y_pred range: [-0.5, +0.5]
        # diff range: [-1, +1]
        diff = y_pred - y_true
        # Circular loss
        return tf.reduce_mean(tf.math.abs(tf.math.maximum(0.5*(1 - tf.math.cos(math.pi*diff)), 1e-10)))


class MSE(tf.keras.losses.Loss):
    def __init__(self, exp):
        super().__init__()

    def call(self, y_true, y_pred):
        # y_true and y_pred range: [-0.5, +0.5]
        # diff range: [-1, +1]
        diff = y_pred - y_true
        # MSE loss:
        return tf.reduce_mean(tf.math.abs(diff)**2)
