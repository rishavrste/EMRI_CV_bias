#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=00:30:00
#PBS -l select=1:ngpus=1:mem=64gb
#PBS -k oed

# diagnose_step.py at one seed. Submit with: qsub -N pnpow_diag -v POINT=4 -o logs/pbs_diag.out -e logs/pbs_diag.err batch_diagnose.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_POWER_DEV

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u diagnose_step.py --point $POINT > logs/diagnose_idx$POINT.log 2>&1
"
