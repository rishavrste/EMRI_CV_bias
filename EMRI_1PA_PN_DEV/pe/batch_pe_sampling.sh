#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# PARIS posterior sampling at a template's best fit vs its Fisher (pe_sampling.py). TEMPLATE=0pa or pn,
# POINT=24 (default), RESUME=1 continues from results/pe/{TEMPLATE}_idx{POINT}[_{TAG}]/sampler_state.pkl.
# Optional: PSR (prior half-width in Fisher sigma), STOP_DLOGZ, MIN_ESS, TAG (output/log suffix).
# Submit from pe/ with:
#   qsub -N pe_0pa24 -v TEMPLATE=0pa -o ../logs/pbs_pe_0pa24.out -e ../logs/pbs_pe_0pa24.err batch_pe_sampling.sh
#   qsub -N pe_pn24  -v TEMPLATE=pn  -o ../logs/pbs_pe_pn24.out  -e ../logs/pbs_pe_pn24.err  batch_pe_sampling.sh
#   qsub -N pe_pn22r -v TEMPLATE=pn,POINT=22,RESUME=1,STOP_DLOGZ=1e-3,MIN_ESS=1000 ... batch_pe_sampling.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_DEV
mkdir -p $D/logs
POINT=${POINT:-24}
RESUME_ARG=$([ "$RESUME" = 1 ] && echo --resume)
EXTRA="${PSR:+--psr $PSR} ${STOP_DLOGZ:+--stop-dlogz $STOP_DLOGZ} ${MIN_ESS:+--min-ess $MIN_ESS} ${TAG:+--tag $TAG}"
LOG=$D/logs/pe_${TEMPLATE}_idx${POINT}${TAG:+_$TAG}${RESUME_ARG:+_resume}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D/pe
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u pe_sampling.py --template $TEMPLATE --point $POINT $RESUME_ARG $EXTRA >> $LOG 2>&1
"
