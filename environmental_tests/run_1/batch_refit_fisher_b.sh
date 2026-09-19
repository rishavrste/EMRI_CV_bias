#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N fish8_b
#PBS -l walltime=96:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/environmental_tests/run_1/fish8_b.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/environmental_tests/run_1/fish8_b.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/environmental_tests/run_1
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/environmental_tests/run_1
    ENV_FISH_IDXS='13 14 15 17 18 19 20 21 22 23 24' python refit_fisher.py > fish8_b.log 2>&1
    wait
"
