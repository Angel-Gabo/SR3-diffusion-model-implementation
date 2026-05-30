import tensorflow as tf
import numpy as np
from models import Net
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import inference,calc_p,generate_params,open_image
import cv2
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",type=str,default="")
parser.add_argument("-i","--infer_steps",type=int,default=100)
parser.add_argument("-t","--steps",type=int,default=2000)
parser.add_argument("-o","--out_file",type=str,default="out_image.jpg")
parser.add_argument("-v","--out_vid",type=str,default="video.mp4")

args = parser.parse_args()
path = args.path
infer_steps = args.infer_steps
steps = args.steps
out_path = args.out_file
pvid = args.out_vid

if path == "":
    print("Debes especificar la ruta del archivo")
    sys.exit()
else:
    if not os.path.exists(path):
        print("Debe existir el archivo")
        sys.exit()

#Hacemos el modelo
net = Net()

#Cargamos los pesos del modelo
checkpoint_path = "checkpoints/"
checkpoint = tf.train.Checkpoint(
    net = net
)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()



numbs = generate_params(steps,beta_end=0.005)
alphas_bar = np.cumprod(numbs, axis=0)

img,(width,height) = open_image(path)
img = tf.expand_dims(img,0)

preds,pred = inference(img, net, numbs, alphas_bar, n_steps=steps,get_frames=True)

fig = plt.figure()
ax = fig.add_subplot()

out_img = tf.squeeze(pred)

if not out_path == "":
    out_img_arr = np.array(out_img,"float32")*255
    cv_img = cv2.cvtColor(out_img_arr,cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_path,cv_img)


def animate(i):
    t = i%(steps+1)
    imk = tf.squeeze(preds[t])
    ax.clear()
    ax.axis("off")
    ax.imshow(imk)

ani = FuncAnimation(fig,animate,frames=steps)
ani.save(pvid,"ffmpeg",fps=60)
plt.show()