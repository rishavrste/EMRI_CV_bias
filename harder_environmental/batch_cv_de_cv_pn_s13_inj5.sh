#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N h13_pn_inj5
#PBS -l walltime=96:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/harder_environmental/hard13_pn_inj5.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/harder_environmental/hard13_pn_inj5.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
    python cv_de_cv.py --model 0PA+PN --point-idx 13 --tag pn_s13_inj5 --fix-params e0 --full-phase-bounds --box-at-injection --sigma-range 5 --de-maxiter 2000 --cp-margin 10 > hard13_pn_inj5.log 2>&1
    wait
"
