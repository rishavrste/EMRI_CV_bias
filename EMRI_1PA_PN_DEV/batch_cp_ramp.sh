#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# C_p ramp re-climb (run_cp_ramp.py) of PN fits stalled on the C = 0 ridge, then the T ladder.
# POINTS="7" or "7-12" (hyphen-separated, for qsub -v), T_FRACS=0.9-1 (fractions of each point's T).
# Submit with:
#   qsub -N emri_ramp7 -v POINTS=7,T_FRACS=0.9-1 -o logs/pbs_ramp7.out -e logs/pbs_ramp7.err batch_cp_ramp.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_DEV
mkdir -p $D/logs
LOG=$D/logs/cp_ramp_Tfrac${T_FRACS}_${POINTS}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u run_cp_ramp.py --points $(echo $POINTS | tr - " ") --t-fracs $(echo $T_FRACS | tr - " ") >> $LOG 2>&1
"
