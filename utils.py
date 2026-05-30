#Leer con cuidado

import tensorflow as tf
import numpy as np

'''
---Funcion de entrenamiento---
img --> Imagen en alta resolucion       #bs,512,512,3
timg --> Imagen en baja resolucion      #bs,512,512,3
net --> Modelo de super resolucion
optimizer --> Optimizador
prod --> calculo de p(y)
'''
@tf.function()
def train_step(img,timg,net,optimizer,alphas_bar_tensor):
    batch_size = tf.shape(img)[0]
    im_shape = tf.shape(img)
    
    n_steps = tf.shape(alphas_bar_tensor)[0]
    t = tf.random.uniform([batch_size], minval=0, maxval=n_steps, dtype=tf.int32)
    
    prod = tf.gather(alphas_bar_tensor,t)
    prod_spatial = tf.reshape(prod,[batch_size,1,1,1])

    noise = tf.random.normal(tf.shape(img),dtype=tf.float32)
    noisy_img = tf.sqrt(prod_spatial)*img + tf.sqrt(1.0-prod_spatial)*noise

    t_input = tf.cast(tf.expand_dims(t, axis=-1), tf.float32)
    with tf.GradientTape() as tape:
        #Prediccion del modelo
        pred = net([timg,noisy_img,t_input],training=True)
        #Calculo del error
        loss = tf.reduce_mean(tf.abs(pred-noise))
        #Calculamos gradientes
        grads = tape.gradient(loss,net.trainable_variables)
    #Aplicamos gradientes
    optimizer.apply_gradients(zip(grads,net.trainable_variables))
    
    return loss

'''
img --> Imagen en baja resolucion
net --> Modelo de super resolucion
numbs --> Hiperparametros(entre 0 y 1)
n_steps --> Cuantos pasos de reduccion de ruido seran
'''

def inference(img, net, numbs, alphas_bar, n_steps=100, get_frames=False):
    if len(img.shape) == 3:
        img = tf.expand_dims(img, axis=0)
        
    yt = tf.random.normal([1, 512, 512, 3])
    if get_frames:
        hist = [yt.numpy()]
        
    for i in reversed(range(0, n_steps)):
        print("Iteracion {}".format(n_steps-i))
        z = tf.random.normal([1, 512, 512, 3]) if i > 0 else tf.zeros([1, 512, 512, 3])
        t_tensor = tf.constant([[i]], dtype=tf.float32)
        
        pred = net([img, yt, t_tensor], training=False)
        alpha_bar_t = np.clip(alphas_bar[i], 1e-12, 1.0 - 1e-12)
        
        a = (1.0 / np.sqrt(numbs[i]))
        b = ((1.0 - numbs[i]) / np.sqrt(1.0 - alpha_bar_t))
        yt_1 = a * (yt - b * pred) + z * np.sqrt(1.0 - numbs[i])
        
        if get_frames:
            hist.append(yt_1.numpy())
        yt = yt_1
        
    if get_frames:
        return hist, yt
    else:
        return yt

#Calculamos p(y) segun visto en el paper de SR3
def calc_p(arr,steps):
    prod = np.prod(arr[:steps + 1])
    prod = np.clip(prod, 1e-12, 1.0 - 1e-12)
    
    return prod

#Generamos hiperparametros, pueden ser diferentes
def generate_params(steps, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, steps)
    alphas = 1.0 - betas
    
    return alphas
def resize(img,width,height):
    return tf.image.resize(img,[width,height],method=tf.image.ResizeMethod.BICUBIC)

def open_image(path,train=False):
    img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(path)),tf.float32)[...,:3]
    sp = tf.shape(img)
    img = img/255
    #Imagen en alta resolucion
    img = resize(img,512,512)
    
    if train:
        #Obtenemos la imagen en baja resolucion y la aumentamos
        timg = resize(img,128,128)
        timg = resize(timg,512,512)
        
        #Las retornamos para el entrenamiento
        return img,timg
    else:
        return img,(sp[0],sp[1])
    
