#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# T-ladder LM (run_lm_T_ladder.py): 2PA signal, 1PA + (C_p, C_e) template.
# IDX="9" or "0-1-2" (hyphen-separated, for qsub -v), T_STEPS=0.2-0.25, FIX_CHI2=1, DIST_DIV=10,
# SEED=inj|1pabest|devinj|dev1pabest (default inj), THEN_FREE=1 (needs FIX_CHI2),
# CHI2_AT=seed (with FIX_CHI2: hold chi2 at the seed value, not the injected one),
# DEV_START=-8_0 (start at C_p = -8, C_e = 0; underscore-separated).
# Submit with:
#   qsub -N dev_Tl_idx9 -v IDX=9,T_STEPS=0.2-0.25,FIX_CHI2=1,DIST_DIV=10 \
#        -o logs/pbs_idx9.out -e logs/pbs_idx9.err batch_lm_T_ladder.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/1PA_vs_2PA/IMRI_TAIL_dev
mkdir -p $D/logs
FLAG=""
if [ -n "$FIX_CHI2" ]; then FLAG="$FLAG --fix-chi2"; fi
if [ -n "$DIST_DIV" ]; then FLAG="$FLAG --dist-div $DIST_DIV"; fi
SEED=${SEED:-inj}
FLAG="$FLAG --seed $SEED"
if [ -n "$THEN_FREE" ]; then FLAG="$FLAG --then-free"; fi
if [ -n "$CHI2_AT" ]; then FLAG="$FLAG --chi2-at $CHI2_AT"; fi
if [ -n "$DEV_START" ]; then FLAG="$FLAG --dev-start $(echo $DEV_START | tr _ " ")"; fi
SEED_TAG=""
if [ "$SEED" != "inj" ]; then SEED_TAG="_seed$SEED"; fi    # inj runs keep their old log names
LOG=$D/logs/lm_Tladder_${T_STEPS}${FIX_CHI2:+_fixchi2}${CHI2_AT:+at$CHI2_AT}${DIST_DIV:+_distdiv$DIST_DIV}${SEED_TAG}${THEN_FREE:+_thenfree}${DEV_START:+_dev$DEV_START}_idx${IDX}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u run_lm_T_ladder.py --idx $(echo $IDX | tr - " ") ${FLAG} --t-steps $(echo $T_STEPS | tr - " ") >> $LOG 2>&1
"
