#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# LM climb from the injection, IMRI_TAIL 2PA signal vs 1PA template, one grid point per job.
# FIX_CHI2=1 holds the secondary spin at its injected value, FIX_PHASES=1 the two phases;
# DIST_DIV=D runs at SNR 20 D, REG=1 steps through the regularised inverse (lm.py). Submit with:
#   qsub -N lm_inj_idx6 -v IDX=6 -o logs/pbs_idx6.out -e logs/pbs_idx6.err batch_lm_from_inj.sh
#   qsub -N lm_inj_fc_idx6 -v IDX=6,FIX_CHI2=1 -o logs/pbs_fixchi2_idx6.out -e logs/pbs_fixchi2_idx6.err batch_lm_from_inj.sh
#   qsub -N lm_inj_fc_dd_idx8 -v IDX=8,FIX_CHI2=1,DIST_DIV=1000 -o logs/pbs_fixchi2_distdiv1000_idx8.out \
#        -e logs/pbs_fixchi2_distdiv1000_idx8.err batch_lm_from_inj.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/1PA_vs_2PA/test_of_LM
mkdir -p $D/logs
if [ -n "$FIX_CHI2" ]; then FLAG=--fix-chi2; TAG=fixchi2_; else FLAG=; TAG=; fi
if [ -n "$FIX_PHASES" ]; then FLAG="$FLAG --fix-phases"; TAG=${TAG}fixphases_; fi
if [ -n "$DIST_DIV" ]; then FLAG="$FLAG --dist-div $DIST_DIV"; TAG=${TAG}distdiv${DIST_DIV}_; fi
if [ -n "$REG" ]; then FLAG="$FLAG --regularise"; TAG=${TAG}reg_; fi
LOG=$D/logs/lm_${TAG}idx${IDX}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    cd /home/svu/e1583490/packages_to_install/SuperKludge_r && echo \"SuperKludge_r branch: \$(git branch --show-current)\" >> $LOG
    cd $D
    python -u run_lm_from_inj.py --idx ${IDX} ${FLAG} >> $LOG 2>&1
"
