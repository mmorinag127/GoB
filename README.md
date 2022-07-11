

# GoB
## pypoetry
```

```


## ALPA

A training script uses (alpa)[https://github.com/alpa-projects/alpa] which can accelerate multi-GPU training on JAX.
You need to install a few dependencies before adding it.
(Document)[https://alpa-projects.github.io/install.html] is here.

### 1. CUDA toolkit:
Follow the official guides to install CUDA and cuDNN. Alpa requires CUDA >= 11.1 and cuDNN >= 8.0.5.

### 2. Install the ILP solver used by Alpa:
```
sudo apt install coinor-cbc
```
or if you don't have permission to do `sudo`, then from (here)[https://github.com/coin-or/Cbc#DownloadandInstall].
Below instruction is building from source.
```
wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
chmod u+x coinbrew
./coinbrew fetch Cbc@master
./coinbrew build Cbc
```

`export LD_LIBRARY_PATH=/home/morinaga/local/src/coinbrew/dist/lib` to your ~/.bashrc (Linux)

### 3. Update pip version and install cupy:
```
poetry add cupy-cuda114
```
Then, check whether your system already has NCCL installed.
```
poetry shell
python -c "from cupy.cuda import nccl"
```
### 3.1 NCLL
check if the below command fails or not, if there is no output it succeeds.
```
python3 -c "from cupy.cuda import nccl"
```

### Alpa install from source

```
git clone --recursive git@github.com:alpa-projects/alpa.git
cd alpa
pip3 install -e ".[dev]"  # Note that the suffix `[dev]` is required to build custom modules.
```
then build `jaxlib` with some fixes.
```
cd alpa/build_jaxlib
python3 build/build.py --enable_cuda --dev_install --tf_path=$(pwd)/../third_party/tensorflow-alpa --cuda_compute_capabilities "3.5,5.2,6.0,7.0,7.5,8.0" --cudnn_path=/home/morinaga/local/cuda/8.2.4 --cuda_path=/usr/local/cuda-11
cd dist

#pip3 install -e .
python setup.py bdist_wheel

```


