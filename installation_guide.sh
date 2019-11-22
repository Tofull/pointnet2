# Example to setup a GPU instance with pointnet++ from scratch.
# Author: Loïc Messal

# In this example, we will use Ubuntu 18.04 / Cuda 10.0 / Cudnn 7.6.5 / Tensorflow 1.14 / Python 2.7. 
# If you want to use different version, please refer to the pointnet++ comment : https://github.com/charlesq34/pointnet2/issues/152#issuecomment-544785566

# Be sure that your computer has at least one nvidia GPU
lspci | grep -i nvidia


########################
##  NVIDIA libraries  ##
########################

# Install libraries to use the GPU
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64  # This is important to find libcudnn7 package.

# Install NVIDIA packages
# Install cuda library
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y cuda-10-0

# A reboot is required
sudo shutdown -r now

# Use some commands to confirm the installation of the cuda library
nvidia-smi
/usr/local/cuda-10.0/bin/nvcc --version

# Install cudnn library
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    libcudnn7=7.6.5.32-1+cuda10.0





########################
##  tensorflow setup  ##
########################

# Install python
sudo apt-get update
sudo apt-get install -y python-dev python-pip

# Install tensorflow
pip install tensorflow-gpu==1.14

# Test tensorflow
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"





########################
##  pointnet++ setup  ##
########################

# Install Pointnet++

# Install pointnet++ dependency
sudo apt-get update
sudo apt-get install -y \
    git \
    g++-4.8

git clone https://github.com/Tofull/pointnet2
cd pointnet2



# Compile Tensorflow operator
# retrieve path to tensorflow library for compilation
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_LINK=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_link_flags()[0])')
TF_FRAMEWORK=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_link_flags()[1])')


# Compile sampling tensorflow operator
cd ./tf_ops/sampling
/usr/local/cuda-10.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++-4.8 -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -lcudart -L /usr/local/cuda-10.0/lib64/ ${TF_LINK} ${TF_FRAMEWORK} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ../..


# Compile grouping tensorflow operator
cd ./tf_ops/grouping
/usr/local/cuda-10.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++-4.8 -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -lcudart -L /usr/local/cuda-10.0/lib64/ ${TF_LINK} ${TF_FRAMEWORK} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ../..


# Compile 3d interpolate function
cd ./tf_ops/3d_interpolation
g++-4.8 -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -lcudart -L /usr/local/cuda-10.0/lib64/ ${TF_LINK} ${TF_FRAMEWORK} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ../..






########################
##  pointnet++ usage  ##
########################

# Use pointnet++

# Train
cd part_seg
python train.py --model pointnet2_part_seg --log_dir log --gpu 0 --max_epoch 201
cd ..


# Evaluate
cd part_seg
python evaluate.py --model pointnet2_part_seg --gpu 0
cd ..


# Infer
cd part_seg
python test.py --model pointnet2_part_seg --gpu 0
cd ..





########################
##  transfer of data  ##
########################

# If you want to test this project out, you can use a GPU instance on any cloud service.
# You can send some data to that instance through scp command
scp --recurse ./data instance-pointnet-gpu:pointnet2/data

# And get back the infered point cloud
scp --recurse instance-pointnet-gpu:pointnet2/part_seg/infer ./infer
