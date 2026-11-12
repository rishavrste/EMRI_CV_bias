#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# LM of the PN + frequency template (run_lm_arms.py) at one point: full T, then the T ladder if that fails. Submit with:
#   qsub -N pnfreq_idx4_pn -v POINT=4,SEED=pn_fit -o logs/pbs_lm_idx4_pn_fit.out -e logs/pbs_lm_idx4_pn_fit.err batch_lm.sh
# SEED defaults to physical.
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_freq
SEED=${SEED:-physical}
LOG=$D/logs/lm_idx${POINT}_${SEED}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u run_lm_arms.py --points $POINT --seed $SEED >> $LOG 2>&1
"
