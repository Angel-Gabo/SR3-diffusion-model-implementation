import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential
from utils import *
import os
from models import Net 
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",type=str,default="")
parser.add_argument("-i","--train_steps",type=int,default=2000)
parser.add_argument("-t","--steps",type=int,default=2000)
parser.add_argument("-c","--ckpt_path",type=str,default="checkpoints/")
parser.add_argument("-b","--batch_size",type=int,default=32)

args = parser.parse_args()
path = args.path
train_steps = args.train_steps
steps = args.steps
ckpt_path = args.ckpt_path
batch_size = args.batch_size

urls = os.listdir(path)
print('''
    Tamaño del set de datos {}
    Batch size {}
'''.format(len(urls),batch_size))

def load_img(url):
    return open_image(path+"/"+url,train=True)

print("Elaborando set de datos...")

train_dataset =  tf.data.Dataset.from_tensor_slices(urls)
train_dataset = train_dataset.map(load_img,num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(32).batch(batch_size)

print("Haciendo red...")
net = Net()
optimizer = tf.keras.optimizers.Adam(1e-4,beta_1=0.5)

numbs = generate_params(steps,1e-6)
alphas_bar = np.cumprod(numbs, axis=0) 
alphas_bar_tf = tf.constant(alphas_bar, dtype=tf.float32)

ckpt_prefix = os.path.join(ckpt_path,"checkpoint")
ckpt = tf.train.Checkpoint(
    net=net,
    optimizer=optimizer,
)

print("Restaurando datos...")
ckpt.restore(tf.train.latest_checkpoint(ckpt_path))

print("Entrenando...")
for epoch in range(train_steps):
    hist = []
    for img,timg in train_dataset:
        l = train_step(img,timg,net,optimizer,alphas_bar_tf)
        hist.append(l.numpy())
    print("Paso {}/{}, error {}".format(epoch+1,steps,np.mean(hist)))
print("Entrenamiento finalizado.")
print("Guardando datos...")
ckpt.save(ckpt_path)
print("Datos guardados en {}".format(ckpt_path))