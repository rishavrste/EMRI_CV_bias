#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N h16_0pa
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/harder_environmental/hard16_0pa.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/harder_environmental/hard16_0pa.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
    python cv_de_cv.py --model 0PA --point-idx 16 --tag 0pa_s16 --fix-params e0 --full-phase-bounds --sigma-range 15 --de-maxiter 1100 --cv1-steps 8 --cv1-step-cap 2 --cv1-total-cap 5 --cp-margin 10 > hard16_0pa.log 2>&1
    wait
"
