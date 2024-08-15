'''losses.py
Custom neural network weight initialization functions
Oliver W. Layton
Last updated: Aug 2024
'''
import tensorflow as tf


class GlorotUniformNonNegative(tf.keras.initializers.Initializer):
    '''Glorot uniform initialization adapted for non-neg weight constraint'''

    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def __call__(self, shape, dtype=None):
        # For Dense layer, fan in and out will be the 1st (and only) shape dims
        # But for CNN, kernel x and y will come 1st so they are the last 2 dims
        fan_in, fan_out = shape[-2], shape[-1]
        edge = tf.sqrt(6.0 / (fan_in + fan_out))

        wts = tf.math.abs(tf.random.uniform(shape=shape, minval=-edge, maxval=edge, seed=self.seed, dtype=dtype))
        return wts
