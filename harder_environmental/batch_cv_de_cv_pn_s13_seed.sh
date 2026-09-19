#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N h13_pn_seed
#PBS -l walltime=96:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/harder_environmental/hard13_pn_seed.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/harder_environmental/hard13_pn_seed.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
    python cv_de_cv.py --model 0PA+PN --point-idx 13 --tag pn_s13_seed --seed-from-0pa /home/svu/e1583490/EMRI_CV_bias/harder_environmental/results_cv_de_cv_0pa_s13.json --fix-params e0 --full-phase-bounds --cp-margin 10 > hard13_pn_seed.log 2>&1
    wait
"
