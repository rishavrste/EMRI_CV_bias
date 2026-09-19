#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N fisher_devapa
#PBS -l walltime=6:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/Fisher_results/fisher_devapa.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/Fisher_results/fisher_devapa.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/Fisher_results
module load singularity

# Requires SuperKludge_r on the 'dev_a_pe' branch (covers simple_pe only --
# 4 cases: adhoc_A, idx1_dt5, pt4, pt20 in the IMRI results). Run this AFTER
# switching the branch away from 'hybrid' (see src/IMRI/README.md).
singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/Fisher_results
    python compute_fishers.py --branch dev_a_pe > compute_fishers_devapa.log 2>&1
    wait
"
