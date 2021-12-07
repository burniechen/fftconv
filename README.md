FFTConv
===
# Env.
```=
python = 3.8.10
pytorch = 1.10.0+cu113
```

# Train model
```=
python3 train/main.py --save-model
```
# Test model
```=
python3 test/main.py
```

# Expected dir.
```=
fftconv
|__ data/
|__ train/
    |__ main.py
    |__ mnist_cnn.pt
|__ test/
    |__ main.py
    |__ net.py
```
