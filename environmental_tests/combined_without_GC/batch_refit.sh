#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -k oed

# qsub -N <name> -o logs/<tag>.output -e logs/<tag>.error \
#      -v FISH_RUN=run_2,FISH_IDXS=0_2_3,TAG=<tag> batch_refit.sh
DIR=/home/svu/e1583490/EMRI_CV_bias/environmental_tests/combined_without_GC
cd $DIR
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $DIR
    FISH_RUN=$FISH_RUN FISH_IDXS='${FISH_IDXS//_/ }' python refit_no_gc.py > logs/$TAG.log 2>&1
    wait
"
