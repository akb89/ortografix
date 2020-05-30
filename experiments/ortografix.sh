#!/bin/sh

echo "I: full hostname: $(hostname -f)"

module load GCCcore/8.2.0
module load Singularity/3.4.0-Go-1.12

VERSION_CUDA='10.1.243'
module load CUDA/${VERSION_CUDA}

# if you need to know the allocated CUDA device, you can obtain it here:
echo "I: CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

echo "====="

srun singularity \
    exec \
    --bind $(readlink /home/kabbach/scratch) \
    --nv \
    /home/kabbach/ortografix-img/ortografix:latest.simg \
        ortografix train \
			--data /home/kabbach/ortografix/data/eng-fra.txt \
			--model-type gru \
			--with-attention \
			--shuffle \
			--max-seq-len 10 \
			--hidden-size 256 \
			--num-layers 1 \
			--bias \
			--dropout 0 \
			--learning-rate 0.01 \
			--epochs 5 \
			--print-every 1000 \
			--use-teacher-forcing \
			--teacher-forcing-ratio 0.5 \
			--output-dirpath /home/kabbach/ortografix/models/attention/
