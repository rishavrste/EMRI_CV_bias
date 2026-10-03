#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# LM up a ladder of observation times (run_lm_T_ladder.py), one grid point per job.
# GRID (default IMRI_TAIL), TEMPLATE (default 1pa), START=inj|best, T_STEPS=0.2-0.4-...-1 (hyphen-separated, for qsub -v);
# FIX_CHI2=1 holds the secondary spin at its injected value; DIST_DIV=D runs at SNR 20 D.
# Submit with:
#   qsub -N lm_Tl_best_idx3 -v GRID=IMRI,TEMPLATE=0pa,IDX=3,START=best,T_STEPS=0.75-1 \
#        -o logs/pbs_Tladder_best_idx3.out -e logs/pbs_Tladder_best_idx3.err batch_lm_T_ladder.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/1PA_vs_2PA/test_of_LM
mkdir -p $D/logs
FLAG="--grid ${GRID:-IMRI_TAIL} --template ${TEMPLATE:-1pa} --start $START"
if [ -n "$FIX_CHI2" ]; then FLAG="$FLAG --fix-chi2"; fi
if [ -n "$DIST_DIV" ]; then FLAG="$FLAG --dist-div $DIST_DIV"; fi
LOG=$D/logs/lm_${GRID:-IMRI_TAIL}_${TEMPLATE:-1pa}_Tladder_${START}_${T_STEPS}_idx${IDX}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    cd /home/svu/e1583490/packages_to_install/SuperKludge_r && echo \"SuperKludge_r branch: \$(git branch --show-current)\" >> $LOG
    cd $D
    python -u run_lm_T_ladder.py --idx ${IDX} ${FLAG} --t-steps $(echo $T_STEPS | tr - " ") >> $LOG 2>&1
"
