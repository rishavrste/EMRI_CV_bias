#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -N lm2g_ecc_b
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/second_gen_TDI/lm2g_b.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/second_gen_TDI/lm2g_b.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/second_gen_TDI
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/second_gen_TDI
    ENV_SETS='set 4,set 5' ENV_TAG=b python lm_second_gen.py > lm2g_b.log 2>&1
    wait
"
