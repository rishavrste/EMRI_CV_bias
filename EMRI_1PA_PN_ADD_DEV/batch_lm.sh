#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# LM ladder of the PN + environmental template (run_lm_arms.py) at one point. Submit with:
#   qsub -N pnenv_idx0 -v POINT=0 -o logs/pbs_lm_idx0.out -e logs/pbs_lm_idx0.err batch_lm.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_ADD_DEV
LOG=$D/logs/lm_idx${POINT}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u run_lm_arms.py --points $POINT --arms pn_ladder >> $LOG 2>&1
"
