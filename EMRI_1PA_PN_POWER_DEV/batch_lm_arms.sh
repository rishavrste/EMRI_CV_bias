#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# LM arms of the power-law PN template (run_lm_arms.py). POINT=4, ARMS=best_full-best_ladder-inj_ladder.
# Submit with:
#   qsub -N pnpow_idx4 -v POINT=4 -o logs/pbs_idx4.out -e logs/pbs_idx4.err batch_lm_arms.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_POWER_DEV
ARMS=${ARMS:-best_full-best_ladder-inj_ladder}
LOG=$D/logs/lm_arms_idx${POINT}_${ARMS}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u run_lm_arms.py --points $POINT --arms $(echo $ARMS | tr - " ") >> $LOG 2>&1
"
