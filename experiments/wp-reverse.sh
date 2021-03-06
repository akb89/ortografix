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
    /home/kabbach/ortografix:latest.simg \
        python /home/kabbach/ortografix/experiments/wp-reverse.py
