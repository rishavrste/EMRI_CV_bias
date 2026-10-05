#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# T-ladder LM (run_lm_T_ladder.py): 1PA signal, 0PA then 0PA + PN template, from the old best fits.
# POINTS="0-1-2-3-4" or "adhoc_A" (hyphen-separated, for qsub -v), T_FRACS=0.9-1 (fractions of each
# point's T), TEMPLATES=0pa-pn (default), SCORE_OLD=1 first scores the old fits (overlap_old_fits.py),
# SEED=old (default), row (make_seeds_row.py, with TEMPLATES=pn), rampfine or best (results/best_fits.json).
# Submit with:
#   qsub -N emri_Tl_r0 -v POINTS=0-1-2-3-4,T_FRACS=0.9-1 \
#        -o logs/pbs_Tl_r0.out -e logs/pbs_Tl_r0.err batch_lm_T_ladder.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_DEV
mkdir -p $D/logs
TEMPLATES=${TEMPLATES:-0pa-pn}
SEED=${SEED:-old}
SFX=$([ "$SEED" = old ] || echo _$SEED)
LOG=$D/logs/lm_Tladder_Tfrac${T_FRACS}_${TEMPLATES}_${POINTS}${SFX}.log
SCORE=""
if [ -n "$SCORE_OLD" ]; then SCORE="python -u overlap_old_fits.py --points $(echo $POINTS | tr - " ") >> $LOG 2>&1"; fi

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    $SCORE
    python -u run_lm_T_ladder.py --points $(echo $POINTS | tr - " ") --t-fracs $(echo $T_FRACS | tr - " ") --templates $(echo $TEMPLATES | tr - " ") --seed $SEED >> $LOG 2>&1
"
