#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# Two-stage LM climb (run_lm_short_T.py), one grid point per job: T_SHORT years (default 0.2)
# from the injection, then the stored 0.25 yr seeded at that final.
# FIX_CHI2=1 holds the secondary spin at its injected value; DIST_DIV=D runs at SNR 20 D;
# TEMPLATE=0pa fits the 0PA template (default 1pa).
# (FIX_PHASES / REG are not supported by run_lm_short_T.py.) Submit with:
#   qsub -N lm_Ts_idx9 -v IDX=9,FIX_CHI2=1,DIST_DIV=10 -o logs/pbs_Tseed_idx9.out \
#        -e logs/pbs_Tseed_idx9.err batch_lm_short_T.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/1PA_vs_2PA/test_of_LM
mkdir -p $D/logs
if [ -n "$FIX_CHI2" ]; then FLAG=--fix-chi2; TAG=fixchi2_; else FLAG=; TAG=; fi
if [ -n "$FIX_PHASES" ]; then FLAG="$FLAG --fix-phases"; TAG=${TAG}fixphases_; fi
if [ -n "$DIST_DIV" ]; then FLAG="$FLAG --dist-div $DIST_DIV"; TAG=${TAG}distdiv${DIST_DIV}_; fi
if [ -n "$REG" ]; then FLAG="$FLAG --regularise"; TAG=${TAG}reg_; fi
if [ -n "$TEMPLATE" ]; then FLAG="$FLAG --template $TEMPLATE"; TAG=${TEMPLATE}_${TAG}; fi
LOG=$D/logs/lm_${TAG}Tseed${T_SHORT:-0.2}_idx${IDX}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    cd /home/svu/e1583490/packages_to_install/SuperKludge_r && echo \"SuperKludge_r branch: \$(git branch --show-current)\" >> $LOG
    cd $D
    python -u run_lm_short_T.py --idx ${IDX} ${FLAG} --t-short ${T_SHORT:-0.2} >> $LOG 2>&1
"
