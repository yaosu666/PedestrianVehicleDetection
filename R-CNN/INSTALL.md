## Installation

### Requirements:
- PyTorch 1.1
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- opencv


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-contrib-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/torch_stable.html

export INSTALL_DIR=$PWD

# install pycocotools
pip install pycocotools

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR
```