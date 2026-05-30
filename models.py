##En este archivo se guardan todos los modelos

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential
import math
from keras import ops

def get_timestep_embedding(timesteps, embedding_dim=256):
    half_dim = embedding_dim // 2
    
    exponent = -math.log(10000) * ops.arange(start=0, stop=half_dim, dtype="float32") / half_dim
    emb = ops.exp(exponent)
    
    timesteps = ops.cast(timesteps, "float32")
    args = timesteps * emb[None, :]
    
    embedding = ops.concatenate([ops.sin(args), ops.cos(args)], axis=-1)
    return embedding
def Upsample(filters,use_drop=False):
    init = tf.random_normal_initializer(0,0.02)
    layer = Sequential()
    layer.add(Conv2DTranspose(filters,4,2,padding="same",kernel_initializer=init))
    layer.add(BatchNormalization())
    if use_drop:
        layer.add(Dropout(0.5))
    layer.add(ReLU())
    return layer

def Downsample(filters,use_batch=True):
    init = tf.random_normal_initializer(0,0.02)
    layer = Sequential()
    layer.add(Conv2D(filters,4,2,padding="same",kernel_initializer=init))
    if use_batch:
        layer.add(BatchNormalization())
    layer.add(LeakyReLU())
    return layer

def ResidualBlock(filters,ni=0):
    inp = Input(shape=[None,None,filters])
    input_temb = Input(shape=[512])
    t_emb = Dense(filters)(input_temb)
    t_emb = Reshape((1, 1, filters))(t_emb)
    if not ni==0:
        ains = [Input(shape=[None,None,filters]) for i in range(ni)]
    
    x = inp
    xi = inp
    if not ni == 0:
        for i in range(ni):
            x = Concatenate()([x,ains[i]])
    
    l1 = Conv2D(filters,3,1,padding="same")
    ac1 = ReLU()
    x = l1(x)
    x = x+t_emb
    x = ac1(x)
    x = x+xi
    if not ni == 0:
        return Model([inp,input_temb]+ains,x)
    else:
        return Model([inp,input_temb],x)
    

def SelfAttentionBlock(filters, num_heads=4):
    inp = Input(shape=[None, None, filters])
    x = LayerNormalization()(inp)
    
    x_flat = Reshape((-1,filters))(x)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=filters // num_heads)(x_flat, x_flat)
    
    orig_shape = ops.shape(inp)
    attn_out = Reshape((16, 16, filters))(attn)
    out = inp + attn_out
    return Model(inp, out)

#Se pueden modificar las capas internas
#En particular, se pueden añadir mas bloques residuales
def Net():
    xi = Input(shape=[512,512,3],name="low_res_image")
    yt = Input(shape=[512,512,3],name="noisy_image")
    t_input = Input(shape=[1],name="timestep")

    t_emb = get_timestep_embedding(t_input, embedding_dim=256)
    t_emb = Dense(512, activation=tf.nn.swish)(t_emb)
    t_emb = Dense(512, activation=tf.nn.swish)(t_emb)
    
    con = Concatenate()([xi,yt])
    
    down_stack = [
        Downsample(64), #256,256,64
        Downsample(128),#128,128,128
        Downsample(256),#64 ,64 ,256
        Downsample(512),#32 ,32 ,512
        Downsample(512),#16 ,16 ,512
    ]
    rs_stack = [
        ResidualBlock(512),
        ResidualBlock(512,1),
        ResidualBlock(512,2),
        ResidualBlock(512,3),
    ]
    attention = SelfAttentionBlock(512, num_heads=4)
    
    up_stack = [
        Upsample(512),               #32 ,32 ,512
        Upsample(256),               #64 ,64 ,256
        Upsample(128),               #128,128,128
        Upsample(64),                #256,256,64
    ]
    last = Conv2DTranspose(3,4,2,padding="same")
    
    x = con
    s = []
    for layer in down_stack:
        x = layer(x)
        s.append(x)
    rsi = []
    for rs in rs_stack:
        if len(rsi) == 0:
            x = rs([x,t_emb])
            rsi.append(x)
        else:
            x = rs([x,t_emb]+rsi)
            rsi.append(x)
    
    x = attention(x)
    s = reversed(s[:-1])
    for layer,sk in zip(up_stack,s):
        x = layer(x)
        x = Concatenate()([x,sk])
    last = last(x)
    return Model([xi,yt,t_input],last)