#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N cv_imri_reclimb
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/src/imri_reclimb.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/src/imri_reclimb.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/src
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/src/IMRI
    python gauss_cv_imri_pn_reclimb.py > imri_pn_reclimb.log 2>&1
    wait
"
