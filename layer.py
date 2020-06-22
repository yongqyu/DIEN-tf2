# https://github.com/zhougr1993/DeepInterestNetwork/blob/master/din/Dice.py
import tensorflow as tf
import tensorflow.keras.layers as nn

class attention(tf.keras.layers.Layer):
    def __init__(self, keys_dim, dim_layers):
        super(attention, self).__init__()
        self.keys_dim = keys_dim

        self.fc = tf.keras.Sequential()
        for dim_layer in dim_layers[:-1]:
            self.fc.add(nn.Dense(dim_layer, activation='sigmoid'))
        self.fc.add(nn.Dense(dim_layers[-1], activation=None))

    def call(self, queries, keys, keys_length):
        queries = tf.tile(tf.expand_dims(queries, 1), [1, tf.shape(keys)[1], 1])
        # outer product ?
        din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
        outputs = tf.transpose(self.fc(din_all), [0,2,1])

        # Mask
        key_masks = tf.sequence_mask(keys_length, max(keys_length), dtype=tf.bool)  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        outputs = outputs / (self.keys_dim ** 0.5)

        # Activation
        outputs = tf.keras.activations.softmax(outputs, -1)  # [B, 1, T]

        # Weighted sum
        outputs = tf.squeeze(tf.matmul(outputs, keys))  # [B, H]

        return outputs

class dice(tf.keras.layers.Layer):
    def __init__(self, feat_dim):
        super(dice, self).__init__()
        self.feat_dim = feat_dim
        self.alphas= tf.Variable(tf.zeros([feat_dim]), dtype=tf.float32)
        self.beta  = tf.Variable(tf.zeros([feat_dim]), dtype=tf.float32)

        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, _x, axis=-1, epsilon=0.000000001):

        reduction_axes = list(range(len(_x.get_shape())))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(_x.get_shape())
        broadcast_shape[axis] = self.feat_dim

        mean = tf.reduce_mean(_x, axis=reduction_axes)
        brodcast_mean = tf.reshape(mean, broadcast_shape)
        std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
        std = tf.sqrt(std)
        brodcast_std = tf.reshape(std, broadcast_shape)

        x_normed = self.bn(_x)
        x_p = tf.keras.activations.sigmoid(self.beta * x_normed)

        return self.alphas * (1.0 - x_p) * _x + x_p * _x

def parametric_relu(_x):
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

class Bilinear(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Bilinear, self).__init__()
        self.linear_act = nn.Dense(units, activation=None, use_bias=True)
        self.linear_noact = nn.Dense(units, activation=None, use_bias=False)

    def call(self, a, b, gate_b=None):
        if gate_b is None:
            return tf.keras.activations.sigmoid(self.linear_act(a) + self.linear_noact(b))
        else:
            return tf.keras.activations.tanh(self.linear_act(a) + tf.math.multiply(gate_b, self.linear_noact(b)))

class AUGRU(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AUGRU, self).__init__()

        self.u_gate = Bilinear(units)
        self.r_gate = Bilinear(units)
        self.c_memo = Bilinear(units)

    def call(self, inputs, state, att_score):
        u = self.u_gate(inputs, state)
        r = self.r_gate(inputs, state)
        c = self.c_memo(inputs, state, r)

        u_= att_score * u
        final = (1 - u_) * state + u_ * c

        return final
