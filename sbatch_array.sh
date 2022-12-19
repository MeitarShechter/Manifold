#!/bin/bash
#
# #SBATCH --mail-user=meitars@uchicago.edu
# #SBATCH --mail-type=ALL
#SBATCH --output=/net/projects/ranalab/meitars/out/preprocess-%j.%N.%a.stdout
#SBATCH --error=/net/projects/ranalab/meitars/out/preprocess-%j.%N.%a.stderr
#SBATCH --chdir=/net/projects/ranalab/meitars/diffusion-3d/data/Manifold/
#SBATCH --partition=threedle-owned
#SBATCH --job-name=preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-4:00:00
#SBATCH --array 5-6

eval "$(conda shell.bash hook)"
conda activate diffusion-3d

echo "starting with cuda version:"
update-alternatives --query cuda
# the path of the installation to use
cuda_path="/usr/local/cuda-11.3"
# filter out those CUDA entries from the PATH that are not needed anymore
path_elements=(${PATH//:/ })
new_path="${cuda_path}/bin"
for p in "${path_elements[@]}"; do
    if [[ ! ${p} =~ ^/usr/local/cuda ]]; then
        new_path="${new_path}:${p}"
    fi
done
# filter out those CUDA entries from the LD_LIBRARY_PATH that are not needed anymore
ld_path_elements=(${LD_LIBRARY_PATH//:/ })
new_ld_path="${cuda_path}/lib64:${cuda_path}/extras/CUPTI/lib64"
for p in "${ld_path_elements[@]}"; do
    if [[ ! ${p} =~ ^/usr/local/cuda ]]; then
        new_ld_path="${new_ld_path}:${p}"
    fi
done
# update environment variables
export CUDA_HOME="${cuda_path}"
export CUDA_ROOT="${cuda_path}"
export LD_LIBRARY_PATH="${new_ld_path}"
export PATH="${new_path}"

echo "current cuda version:"
update-alternatives --query cuda

cmd=`sed "${SLURM_ARRAY_TASK_ID}q;d" ./array_args_pre_process.txt`
echo 'Starting the program:'
echo ${cmd}
eval ${cmd}
# eval `sed "${SLURM_ARRAY_TASK_ID}q;d" ./array_args_pre_process.txt`