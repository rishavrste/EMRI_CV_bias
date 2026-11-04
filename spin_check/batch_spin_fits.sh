#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=48:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# Spin check (run_spin_fits.py): injection, 0PA and 0PA + PN T ladder, Fisher at the best fit.
# SPINS="-0.1_0.1" (underscore-separated, for qsub -v), TEMPLATES=0pa-pn (default).
# Optional: SEED=injection, TFRACS=0.4_0.5_..._1.0, TAG=inj_fine (outputs {template}_{TAG}_*).
# Submit with:
#   qsub -N spin_a01 -v SPINS=-0.1_0.1 -o logs/pbs_a01.out -e logs/pbs_a01.err batch_spin_fits.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/spin_check
mkdir -p $D/logs
TEMPLATES=${TEMPLATES:-0pa-pn}
EXTRA="--seed ${SEED:-offset}"
[ -n "$TFRACS" ] && EXTRA="$EXTRA --t-fracs $(echo $TFRACS | tr _ ' ')"
[ -n "$TAG" ] && EXTRA="$EXTRA --tag $TAG"
LOG=$D/logs/spin_fits_${SPINS}_${TEMPLATES}${TAG:+_$TAG}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u run_spin_fits.py --spins $(echo $SPINS | tr _ " ") --templates $(echo $TEMPLATES | tr - " ") $EXTRA >> $LOG 2>&1
"
