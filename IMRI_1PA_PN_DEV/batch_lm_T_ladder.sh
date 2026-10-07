#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# T-ladder LM (run_lm_T_ladder.py): IMRI grid, 1PA signal, 0PA then 0PA + PN template.
# POINTS="0-1-2-3-4" (hyphen-separated, for qsub -v), SEED=old (1st-gen best fit) or sk (SK_files 0PA
# best fit), T_FRACS=0.9-0.95-1 (default, fractions of T = 1 yr), TEMPLATES=0pa-pn (default).
# The seeds of POINTS are first scored at full T (score_seeds.py).
# Submit with:
#   qsub -P CFP05-CF-131 -N imri_Tl_old_r0 -v POINTS=0-1-2-3-4,SEED=old \
#        -o logs/pbs_Tl_old_r0.out -e logs/pbs_Tl_old_r0.err batch_lm_T_ladder.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/IMRI_1PA_PN_DEV
mkdir -p $D/logs
TEMPLATES=${TEMPLATES:-0pa-pn}
T_FRACS=${T_FRACS:-0.9-0.95-1}
LOG=$D/logs/lm_Tladder_${SEED}_Tfrac${T_FRACS}_${TEMPLATES}_${POINTS}.log
PTS=$(echo $POINTS | tr - " ")

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u score_seeds.py --seed $SEED --points $PTS >> $LOG 2>&1
    python -u run_lm_T_ladder.py --points $PTS --seed $SEED --t-fracs $(echo $T_FRACS | tr - " ") --templates $(echo $TEMPLATES | tr - " ") >> $LOG 2>&1
"
