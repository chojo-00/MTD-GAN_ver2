#!/bin/bash
#SBATCH -J mtd_gan_train_ver1
#SBATCH -t 7-00:00:00
#SBATCH -o logs/%x_%A_%N.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND,BEGIN
#SBATCH --mail-user chobyeongcheon00@gmail.com
#SBATCH -p A6000
#SBATCH -w gpu120
#SBATCH --gres=gpu:1


# export HTTP_PROXY="http://192.168.45.108:3128"
# export HTTPS_PROXY="http://192.168.45.108:3128"

# Define vars
JOB_NAME="mtd_gan_train_ver1"
DOCKER_IMAGE_NAME="bc_cho/${JOB_NAME}"
DOCKER_CONTAINER_NAME="bc_cho${JOB_NAME}"
PORT_NUM=4966



# Paths inside the container
CODE_DIR="/workspace/bc_cho/0_Project/2_LDCT2NDCT/MTD-GAN_ver1"
CHECKPOINT_DIR="${CODE_DIR}/checkpoints"
SAVE_DIR="${CODE_DIR}/predictions"


# Run containers
docker build -t ${DOCKER_IMAGE_NAME} -f Dockerfile .

# Stop running container
if docker ps -q --filter "name=${DOCKER_CONTAINER_NAME}" | grep -q .; then
    echo "Stopping running container: ${DOCKER_CONTAINER_NAME}"
    docker stop ${DOCKER_CONTAINER_NAME}
fi

# Remove existing container
if docker ps -a -q --filter "name=${DOCKER_CONTAINER_NAME}" | grep -q .; then
    echo "Removing stopped container: ${DOCKER_CONTAINER_NAME}"
    docker rm ${DOCKER_CONTAINER_NAME}
fi

# gpu 
echo "=== Check GPU on Host Node ==="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


# Run containers
docker run --rm \
        --name ${DOCKER_CONTAINER_NAME} \
        --shm-size 1TB \
        --device nvidia.com/gpu=all \
        -v /mnt/nas100/forGPU/bc_cho:/workspace/bc_cho \
        -v /mnt/nas69/ds_WBCT/IDs/cychoi/ver_1:/workspace/bc_cho/0_Project/2_LDCT2NDCT/dataset/nas69 \
        -v /mnt/nas206/ds_WBCT_share:/workspace/bc_cho/0_Project/2_LDCT2NDCT/dataset/nas206 \
        ${DOCKER_IMAGE_NAME} \
        bash -c "
            cd ${CODE_DIR} && \
            python3 train.py \
                --dataset my_chest_data \
                --dataset-type-train 'window_patch' \
                --dataset-type-valid 'window' \
                --batch-size 20 \
                --train-num-workers 16 \
                --valid-num-workers 16 \
                --model 'MTD_GAN_Method' \
                --loss 'L1 Loss' \
                --method 'pcgrad' \
                --optimizer 'adamw' \
                --scheduler 'poly_lr' \
                --epochs 500 \
                --warmup-epochs 10 \
                --lr 1e-4 \
                --min-lr 1e-6 \
                --print-freq 10 \
                --save-checkpoint-every 1 \
                --checkpoint-dir ${CHECKPOINT_DIR} \
                --save-dir ${SAVE_DIR} 
        "