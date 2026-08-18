#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N cv_bias_ko
#PBS -l walltime=48:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/src/imri.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/src/imri.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/src
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/src/EMRI
    python gauss_cv_emri_grid_diverse2_dev_from_0pa.py > emri_grid_diverse2_dev_from_0pa.log 2>&1
    wait
"