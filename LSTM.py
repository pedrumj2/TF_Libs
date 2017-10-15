import pymysql
import numpy as np
import tensorflow as tf


# This class gets data for a single user
class LSTM(object):
    def __init__(self, layers, hidden_layer_size, input_size, seed=-1 ):
        self._input_size = input_size
        self._hidden_layer_size = hidden_layer_size
        self.layers = layers
        self.seed = seed
        self._init_tensors()

    def _init_tensors(self):
        self.tf_wt = []
        self.tf_wi = []
        self.tf_wc = []
        self.tf_wo = []
        self.tf_bt = []
        self.tf_bi = []
        self.tf_bc = []
        self.tf_bo = []
        _size = self._hidden_layer_size + self._input_size
        self._init_tensor_block(self.layers, _size)

    def _init_tensor_block(self, layer, input_size):
        if self.seed != -1:
            self.tf_wt.append(tf.Variable(
                tf.random_normal([self._hidden_layer_size, input_size], stddev=0.35, seed=self.seed)))
            self.tf_wi.append(tf.Variable(
                tf.random_normal([self._hidden_layer_size, input_size], stddev=0.35, seed=self.seed)))
            self.tf_wc.append(tf.Variable(
                tf.random_normal([self._hidden_layer_size, input_size], stddev=0.35, seed=self.seed)))
            self.tf_wo.append(tf.Variable(
                tf.random_normal([self._hidden_layer_size, input_size], stddev=0.35, seed=self.seed)))
        else:
            self.tf_wt.append(tf.Variable(
                tf.random_normal([self._hidden_layer_size, input_size], stddev=0.35)))
            self.tf_wi.append(tf.Variable(
                tf.random_normal([self._hidden_layer_size, input_size], stddev=0.35)))
            self.tf_wc.append(tf.Variable(
                tf.random_normal([self._hidden_layer_size, input_size], stddev=0.35)))
            self.tf_wo.append(tf.Variable(
                tf.random_normal([self._hidden_layer_size, input_size], stddev=0.35)))
        self.tf_bt.append(tf.Variable(tf.zeros([self._hidden_layer_size, 1])))
        self.tf_bi.append(tf.Variable(tf.zeros([self._hidden_layer_size, 1])))
        self.tf_bc.append(tf.Variable(tf.zeros([self._hidden_layer_size, 1])))
        self.tf_bo.append(tf.Variable(tf.zeros([self._hidden_layer_size, 1])))

    def apply(self, input_data, state):
        state, output = self._block(input_data, state)
        return state, output

    def _block(self, input_data, state):
        state_input_date_conc = tf.concat([input_data, state], 0)
        tf_ft =  tf.nn.sigmoid(tf.matmul(self.tf_wt[0], state_input_date_conc) + self.tf_bt[0])
        tf_it = tf.nn.sigmoid(tf.matmul(self.tf_wi[0], state_input_date_conc) + self.tf_bi[0])
        tf_chat_t = tf.nn.tanh(tf.matmul(self.tf_wc[0], state_input_date_conc) + self.tf_bc[0])
        state = tf.multiply(tf_ft,state) + tf.multiply(tf_it,tf_chat_t)
        tf_ot = tf.nn.sigmoid(tf.matmul(self.tf_wo[0], state_input_date_conc) + self.tf_bo[0])
        output = tf.multiply(tf_ot, tf.nn.tanh(state))
        return state, output




