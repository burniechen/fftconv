FFTConv
===
# Env.
```=
python = 3.8.10
pytorch = 1.10.0+cu113
```

# MNIST
## Train model
```=
cd mnist_cnn
python3 train/main.py --save-model
```
## Test model
```=
cd mnist_cnn
python3 test/main.py
```

# VGG16
## Download dataset
[imagenette2](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz)

## Train model
```=
cd vgg16_cnn
python3 train/main.py
```
## Test model
```=
cd vgg16_cnn
python3 test/main.py
```

# Expected dir.
```=
fftconv
|__ data/
	|__ MNIST/
	|__ imagenet2/

|__ mnist_cnn/
	|__ train/
		|__ main.py
		|__ mnist_cnn.pt
	|__ test/
		|__ main.py
		|__ net.py

|__ vgg16_cnn/
	|__ train/
		|__ main.py
		|__ vgg16.py
		|__ vgg16_cnn.pt
	|__ test/
		|__ main.py
		|__ net.py
```
