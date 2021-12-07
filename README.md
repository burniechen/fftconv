FFTConv
===
# env
1. python = 3.8.10
2. pytorch = 1.10.0+cu113

# train model
```
python3 train/main.py --save-model
```
# test model
```
python3 test/main.py
```

# expected folers
fftconv
|__ data/
|__ train/
	|__ main.py	
	|__ mnist_cnn.pt
|__ test/
	|__ main.py
	|__ net.py
