#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=00:30:00
#PBS -l select=1:ngpus=1:mem=64gb
#PBS -k oed

# check_model.py on a GPU. qsub -N rel5_check -o logs/pbs_check.out -e logs/pbs_check.err batch_check.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/EMRI_REL5_LM
singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u check_model.py > logs/check_model.log 2>&1
"
