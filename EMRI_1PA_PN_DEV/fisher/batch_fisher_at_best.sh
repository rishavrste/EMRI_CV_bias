#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=06:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# Fisher at the best fits (fisher_at_best.py; run best_fits.py first). POINTS="0-1-2" (hyphen-separated,
# default all 25 grid points), TEMPLATES="0pa-pn" (default both), STEPS_FROM="12-17" (optional: fixed SEF
# steps taken from these points' stored Fishers, the step-sensitivity test).
# Submit from fisher/ with:
#   qsub -N emri_fisher -o ../logs/pbs_fisher_at_best.out -e ../logs/pbs_fisher_at_best.err batch_fisher_at_best.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_DEV
mkdir -p $D/logs
POINTS=${POINTS:-$(seq -s - 0 24)}
TEMPLATES=${TEMPLATES:-0pa-pn}
LOG=$D/logs/fisher_at_best_${TEMPLATES}${STEPS_FROM:+_steps${STEPS_FROM}}.log
STEPS_ARG=${STEPS_FROM:+--steps-from $(echo $STEPS_FROM | tr - " ")}

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D/fisher
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u fisher_at_best.py --templates $(echo $TEMPLATES | tr - " ") --points $(echo $POINTS | tr - " ") $STEPS_ARG >> $LOG 2>&1
"
