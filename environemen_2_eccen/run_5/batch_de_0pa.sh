#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N s5_0pa
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/run_5/de_0pa.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/run_5/de_0pa.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/run_5
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/run_5
    python de_run.py config_de_0pa.json > de_0pa.log 2>&1
    wait
"
