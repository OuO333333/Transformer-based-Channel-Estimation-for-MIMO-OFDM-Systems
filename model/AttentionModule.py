import tensorflow as tf
from keras.layers import Layer

class SelfAttention(Layer):
    def __init__(self, output_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.WQ = self.add_weight(shape=(input_shape[-1],self.latent_dim),initializer="random_normal",trainable=True,name='wq')
        self.WK = self.add_weight(shape=(input_shape[-1],self.latent_dim),initializer="random_normal",trainable=True,name='wk')
        self.WV = self.add_weight(shape=(input_shape[-1],self.output_dim),initializer="random_normal",trainable=True,name='wv')
       
    def call(self, inputs):
        Q, K, V = tf.matmul(inputs, self.WQ), tf.matmul(inputs, self.WK), tf.matmul(inputs, self.WV)
        A = tf.matmul(Q, K, transpose_b=True)
        A_norm = tf.nn.softmax(A/(K.shape[-1]**0.5))
        outputs = tf.matmul(A_norm,V)
        return outputs
    
class Multiheadattention(tf.keras.layers.Layer):
    def __init__(self, output_dim, latent_dim, heads):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.heads = heads

    def build(self, input_shape):
        self.WQ = self.add_weight(shape=(self.heads, input_shape[-1], self.latent_dim),initializer="random_normal",trainable=True,name='wq')
        self.WK = self.add_weight(shape=(self.heads, input_shape[-1], self.latent_dim),initializer="random_normal",trainable=True,name='wk')
        self.WV = self.add_weight(shape=(self.heads, input_shape[-1], self.output_dim),initializer="random_normal",trainable=True,name='wv')
        self.WO = self.add_weight(shape=(self.heads*self.output_dim, self.output_dim),initializer="random_normal",trainable=True,name='wo')

    def call(self, inputs):
        Q, K, V = tf.matmul(inputs, self.WQ[0]), tf.matmul(inputs, self.WK[0]), tf.matmul(inputs, self.WV[0])
        A = tf.matmul(Q, K, transpose_b=True)
        A_norm = tf.nn.softmax(A/(K.shape[-1]**0.5))
        output = tf.matmul(A_norm,V)
        for head in range(1, self.heads):
            Q, K, V = tf.matmul(inputs, self.WQ[head]), tf.matmul(inputs, self.WK[head]), tf.matmul(inputs, self.WV[head])
            A = tf.matmul(Q, K, transpose_b=True)
            A_norm = tf.nn.softmax(A/(K.shape[-1]**0.5))
            o = tf.matmul(A_norm,V)
            output = tf.concat([output, o], axis=-1)
        outputs = tf.matmul(output,self.WO)
        return outputs