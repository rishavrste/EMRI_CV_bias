#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -N e2r0
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/environmental_tests/run_2/lm_e0fixed_redo0pa.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/environmental_tests/run_2/lm_e0fixed_redo0pa.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/environmental_tests/run_2
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/environmental_tests/run_2
    python lm_e0fixed_redo_0pa.py > lm_e0fixed_redo0pa.log 2>&1
    wait
"
