from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class TensorRNN(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=None, reuse=None):
        super(TensorRNN, self).__init__(_reuse=reuse)

        self._num_units = num_units
        self._activation = activation or tf.tanh

        self._kernel_U = self.add_variable(
            'U',
            shape=[100, self._num_units], dtype=tf.float32)
        self._kernel_W = self.add_variable(
            'W',
            shape=[self._num_units, self._num_units], dtype=tf.float32)
        self._kernel_b = self.add_variable(
            'b',
            shape=[self._num_units], dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        print("$$$$$$$$$$$$4$$4")
        print(inputs)

        a = tf.matmul(inputs, self._kernel_U)
        b = tf.matmul(state, self._kernel_W)
        output = a * b + self._kernel_b
        output = tf.nn.l2_normalize(output, dim=1)
        return output, output
    # @property
    # def state_size(self):
    #   return LSTMStateTuple(self._num_units, self._num_units)
