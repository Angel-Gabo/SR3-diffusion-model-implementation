# SR3-diffusion-model-implementation
Implementation of SR3 to increase the resolution of images from 64x64 to 512x512 with images of cats.

<picture>
<img src="image.png" width=250>
</picture>
<picture>
<img src="output.jpg" width=250>
</picture>

The comments are in Spanish.

The code is the implementation of SR3 [<https://iterative-refinement.github.io/>].

File structure
|File| use|
|-----:|---------|
|train.py| Train the model| 
|test.py| Get a prediction from the model|
|animation.py| Create a video of how you are reducing noise to the image|

## Test
In order to test the model, you need the checkpoints, which are at: <https://drive.google.com/drive/folders/1f8nCr30DnkJaPPwX4iiNlx0eMJzhQm3P?usp=drive_link>

They should be in a folder called checkpoints

### Use of test.py
The way to use the file is:
```bash
python3 test.py -p <path image> -i <inference steps> -t <steps model> -o <out file>
```

Example:
``` bash
python3 test.py -p image.jpg -i 100 -o out_image.jpg
```

### Use of animation.py
[![-](out_image.jpg)](GIF.gif)


The way to use the file is: 
```bash
python3 animation.py -p <path image> -i <inference steps> -t <steps model> -o <out image> -v <out video>
```

Example:
```bash
python3 animation.py -p image.jpg -i 100 -o out_img.jpg -v video.mp4
```

