#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=03:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# 14-parameter Fisher at the best LM fit of each point, then the summary table. Submit with:
#   qsub -N rel5_fisher -o logs/pbs_fisher.out -e logs/pbs_fisher.err batch_fisher.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/EMRI_REL5_LM

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u fisher_at_best.py --points 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 > logs/fisher_at_best.log 2>&1
    python -u make_summary.py > logs/make_summary.log 2>&1
"
