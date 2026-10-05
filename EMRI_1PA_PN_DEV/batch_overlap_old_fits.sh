#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=04:00:00
#PBS -l select=1:ngpus=1:mem=64gb
#PBS -k oed

# Overlap of the old 1st-generation best fits in the 2nd-generation AE setup (overlap_old_fits.py).
# Submit with:
#   qsub -N emri_ov_old -o logs/pbs_overlap_old.out -e logs/pbs_overlap_old.err batch_overlap_old_fits.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_DEV
mkdir -p $D/logs
LOG=$D/logs/overlap_old_fits.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u overlap_old_fits.py >> $LOG 2>&1
"
