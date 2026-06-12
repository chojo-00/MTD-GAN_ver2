#!/bin/bash
#SBATCH -J mtd_gan_test_ver2_denoising_60
#SBATCH -t 7-00:00:00
#SBATCH -o logs/%x_%A_%N.out
#SBATCH --mail-type BEGIN,END
#SBATCH --mail-user chobyeongcheon00@gmail.com
#SBATCH -p RTX3090
#SBATCH -w gpu32
#SBATCH --gres=gpu:1


export HTTP_PROXY="http://192.168.45.108:3128"
export HTTPS_PROXY="http://192.168.45.108:3128"

# Define vars
JOB_NAME="mtd_gan_test_ver2_denoising_60"
DOCKER_IMAGE_NAME="bc_cho/${JOB_NAME}"
DOCKER_CONTAINER_NAME="bc_cho${JOB_NAME}"
PORT_NUM=4150



# Paths inside the container
CODE_DIR="/workspace/bc_cho/0_Project/2_LDCT2NDCT/MTD-GAN_ver2"
CHECKPOINT_DIR="${CODE_DIR}/checkpoints/denoising_60"
# 테스트 결과 이미지가 섞이지 않도록 별도의 폴더 지정
SAVE_DIR="${CODE_DIR}/test_results/denoising_60_new_new" 


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
            python3 test.py \
                --dataset my_chest_data \
                --dataset-type-test 'full' \
                --in-kernel 'B60f' \
                --task 'denoising' \
                --test-batch-size 1 \
                --test-num-workers 16 \
                --model 'MTD_GAN_Method' \
                --loss 'L1 Loss' \
                --multi-gpu-mode Single \
                --print-freq 10 \
                --resume ${CHECKPOINT_DIR}/epoch_9_checkpoint.pth \
                --checkpoint-dir ${CHECKPOINT_DIR} \
                --save-dir ${SAVE_DIR} \
                --epoch 10
        "


## parser 모음
##                 --gt-kernel 'B60f' \
##                 --dose 'full' \
