#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=02:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# 13-parameter Fisher at the fits of the arms. Submit with:
#   qsub -N pnpow_fisher -o logs/pbs_fisher.out -e logs/pbs_fisher.err batch_fisher_at_best.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_POWER_DEV

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u fisher_at_best.py --points 0 4 --arms best_ladder > logs/fisher_at_best.log 2>&1
"
