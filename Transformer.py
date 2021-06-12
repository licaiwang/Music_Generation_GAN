import tensorflow as tf
from tensorflow.keras.layers import Reshape, Input, Conv1D, Add, Activation
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class MyConvld(tf.keras.layers.Layer):
    def __init__(self, nx, nf, start):
        super(MyConvld, self).__init__()
        self.nx = nx
        self.nf = nf
        self.start = start
        w_initializer = tf.random_normal_initializer(stddev=0.02)
        b_initializer = tf.constant_initializer(0)
        self.w = tf.Variable(w_initializer([1, nx, nf], dtype=tf.float32), name="w")
        self.b = tf.Variable(b_initializer([nf], dtype=tf.float32), name="b")

    def call(self, inputs):
        i = tf.reshape(inputs, [-1, self.nx])
        wn = tf.reshape(self.w, [-1, self.nf])
        return tf.reshape(tf.matmul(i, wn) + self.b, self.start + [self.nf])


"""
MEMORY EFFICIENT IMPLEMENTATION OF RELATIVE POSITION-BASED ATTENTION
(Music Transformer, Cheng-Zhi Anna Huang et al. 2018)
"""


class ATTN(tf.keras.layers.Layer):
    def __init__(self, n_state, n_head, seq):
        super(ATTN, self).__init__()
        self.n_state = n_state * 3
        self.n_head = n_head
        # TODO:
        E_initializer = tf.constant_initializer(0)
        self.E = tf.Variable(
            E_initializer(shape=[16, seq, 32], dtype=tf.float32), name="E"
        )

    def split_heads(self, x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(self.split_states(x, self.n_head), [0, 2, 1, 3])

    def split_states(self, x, n):
        """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
        *start, m = shape_list(x)
        return tf.reshape(x, start + [n, m // n])

    def merge_heads(self, x):
        # Reverse of split_heads
        return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = self.attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def merge_states(self, x):
        """Smash the last two dimensions of x into a single dimension."""
        *start, a, b = shape_list(x)
        return tf.reshape(x, start + [a * b])

    def attention_mask(self, nd, ns, *, dtype):
        """1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def relative_attn(self, q):
        # q have shape [batch, heads, sequence, features]
        batch, heads, sequence, features = shape_list(q)
        # [heads, batch, sequence, features]
        q_ = tf.transpose(q, [1, 0, 2, 3])
        # [heads, batch * sequence, features]
        q_ = tf.reshape(q_, [heads, batch * sequence, features])
        # [heads, batch * sequence, sequence]
        rel = tf.matmul(q_, self.E, transpose_b=True)
        # [heads, batch, sequence, sequence]
        rel = tf.reshape(rel, [heads, batch, sequence, sequence])
        # [heads, batch, sequence, 1+sequence]
        rel = tf.pad(rel, ((0, 0), (0, 0), (0, 0), (1, 0)))
        # [heads, batch, sequence+1, sequence]
        rel = tf.reshape(rel, (heads, batch, sequence + 1, sequence))
        # [heads, batch, sequence, sequence]
        rel = rel[:, :, 1:]
        # [batch, heads, sequence, sequence]
        rel = tf.transpose(rel, [1, 0, 2, 3])
        return rel

    def multihead_attn(self, q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w + self.relative_attn(q)
        w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))
        w = self.mask_attn_weights(w)
        w = tf.nn.softmax(w, axis=-1)
        a = tf.matmul(w, v)
        return a

    def call(self, inputs):
        q, k, v = map(self.split_heads, tf.split(inputs, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        a = self.multihead_attn(q, k, v)
        a = self.merge_heads(a)
        return a, present


class NormalizeDiagonal(tf.keras.layers.Layer):
    def __init__(self, n_state):
        super(NormalizeDiagonal, self).__init__()
        self.n_state = n_state
        g_initializer = tf.constant_initializer(1)
        b_initializer = tf.constant_initializer(0)
        self.g = tf.Variable(g_initializer([n_state], dtype=tf.float32), name="g")
        self.b = tf.Variable(b_initializer([n_state], dtype=tf.float32), name="b")

    def call(self, inputs):
        u = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        s = tf.reduce_mean(tf.square(inputs - u), axis=-1, keepdims=True)
        x = (inputs - u) * tf.math.rsqrt(s + 1e-5)
        return x * self.g + self.b


class WTE(tf.keras.layers.Layer):
    def __init__(self, n_vocab, n_embd):
        super(WTE, self).__init__()
        initializer = tf.random_normal_initializer(stddev=0.02)
        self.wte = tf.Variable(initializer([n_vocab, n_embd]), name="wte")

    def call(self, inputs):
        return [tf.gather(self.wte, inputs), self.wte]

def dilated_causal_Conv1D(dilation_r,activation_fn,dilated_num=0,num_stack=0):
    return Conv1D(filters=512,
    kernel_size=2,
    strides=1,
    padding="causal", #valid
    dilation_rate=dilation_r,
    activation=activation_fn,
    use_bias=True,
    name='%s_stack%d_dilated%d_causal_conv_layer'%(activation_fn,num_stack,dilated_num))
    
    
    
def TransformerGenerator(hparams, input_shape):

    n_vocab = hparams["EventDim"]
    n_embd = hparams["EmbeddingDim"]
    n_layer = hparams["Layers"]
    n_head = hparams["Heads"]
    n_sequence = hparams["Time"]
    
    batch_size = 1
    inputs = Input(shape=input_shape, dtype=tf.float32)


    h = dilated_causal_Conv1D(1,None,-1,-1)(inputs)
    nx = 512
    
    # Transformer
    for layer in range(n_layer):
        ## ATTN ###
        nor = NormalizeDiagonal(n_embd)(h)
        a = MyConvld(nx, nx * 3, [batch_size, n_sequence])(nor)
        a, present = ATTN(nx, n_head,n_sequence)(a)
        a = MyConvld(nx, nx, [batch_size, n_sequence])(a)
        ##########
        h = Add()([h, a])
        ###########
        ##  MLP  ##
        nor = NormalizeDiagonal(n_embd)(h)
        a = MyConvld(nx, nx * 4, [batch_size, n_sequence])(nor)
        a = Activation("gelu")(a)
        m = MyConvld(nx * 4, nx, [batch_size, n_sequence])(a)
        ###########
        h = Add()([h, m])
        ###########

    ### output ###
    h = NormalizeDiagonal(n_embd)(h)
    ### back to 0~1
    h = Dense(n_sequence)(h)
    h = GRU(256)(h)    
    h = Activation("sigmoid")(h)
    h = Reshape((256,1))(h)
    return Model(inputs, h)

def TransformerDiscriminator(hparams, input_shape):

    n_vocab = hparams["EventDim"]
    n_embd = hparams["EmbeddingDim"]
    n_layer = int(hparams["Layers"])
    n_head = hparams["Heads"]
    n_sequence = hparams["Time"]
    batch_size = 1
    inputs = Input(shape=input_shape, dtype=tf.float32)

    batch = 1

    h = dilated_causal_Conv1D(1,None,-1,-1)(inputs)
    nx = 512

    # Transformer
    for layer in range(n_layer):
        ## ATTN ###
        nor = NormalizeDiagonal(512)(h)
        a = MyConvld(nx, nx * 3, [batch_size, n_sequence])(nor)
        a, present = ATTN(nx, n_head,n_sequence)(a)
        a = MyConvld(nx, nx, [batch_size, n_sequence])(a)
        ##########
        h = Add()([h, a])
        ###########
        ##  MLP  ##
        nor = NormalizeDiagonal(512)(h)
        a = MyConvld(nx, nx * 4, [batch_size, n_sequence])(nor)
        a = Activation("gelu")(a)
        m = MyConvld(nx * 4, nx, [batch_size, n_sequence])(a)
        ###########
        h = Add()([h, m])
        ###########
    
    
    ### output ###
    h = NormalizeDiagonal(n_embd)(h)
    ### back to 0~1
    h = Dense(n_sequence)(h)
    h = GRU(256)(h)
    h = Dense((1))(h)

    return Model(inputs, h)
