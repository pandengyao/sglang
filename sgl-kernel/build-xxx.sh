# 设置 CMake 版本
CMAKE_VERSION_MAJOR=3.31
CMAKE_VERSION_MINOR=1

# 下载并安装 CMake
echo "Downloading CMake from: https://cmake.org/files/v3.31/cmake-3.31.1-linux-x86_64.tar.gz"
wget https://cmake.org/files/v3.31/cmake-3.31.1-linux-x86_64.tar.gz
tar -xzf cmake-3.31.1-linux-x86_64.tar.gz
mv cmake-3.31.1-linux-x86_64 /opt/cmake
export PATH=/opt/cmake/bin:$PATH
export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH

# Debug CMake
echo "PATH: $PATH"
which cmake
cmake --version

# 安装系统依赖
yum install numactl-devel -y
yum install libibverbs -y --nogpgcheck
ln -sv /usr/lib64/libibverbs.so.1 /usr/lib64/libibverbs.so

# 安装 PyTorch 和 Python 依赖
/opt/python/cp310-cp310/bin/pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/test/cu126
/opt/python/cp310-cp310/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core

# 设置 CUDA 环境
export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.9 9.0+PTX'
export CUDA_VERSION=12.6
mkdir -p /usr/lib/x86_64-linux-gnu/
ln -s /usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so

# 构建 Python Wheel
cd /sgl-kernel
ls -la /opt/python/cp310-cp310/lib/python3.10/site-packages/wheel/
PYTHONPATH=/opt/python/cp310-cp310/lib/python3.10/site-packages /opt/python/cp310-cp310/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation

# 重命名生成的 wheels
./rename_wheels.sh
