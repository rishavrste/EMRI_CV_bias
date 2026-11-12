#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# Reseed LM (run_lm.py --coeff-seed-from $SRC) at the EMRI points in $POINTS: injected physical parameters +
# the deviation coefficients of point $SRC full fit, one climb at T_FULL -> results/reseed/. Needs PN_v_3.
# Each point logs to logs/reseed_idx{i}.log. Submit with:
#   qsub -N rel5_rs3 -v POINTS=3,SRC=8 -o logs/pbs_reseed.out -e logs/pbs_reseed.err batch_reseed.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/EMRI_REL5_LM

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    for p in \$(echo $POINTS | tr ':' ' '); do
        LOG=$D/logs/reseed_idx\$p.log
        python -u -c 'import few; print(few.__file__)' > \$LOG 2>&1
        python -u run_lm.py --points \$p --coeff-seed-from $SRC >> \$LOG 2>&1
    done
"
