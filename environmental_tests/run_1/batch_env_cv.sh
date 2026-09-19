#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N env_cv
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/environmental_tests/env_cv.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/environmental_tests/env_cv.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/environmental_tests
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/environmental_tests
    python cv_population.py > env_cv.log 2>&1
    wait
"
