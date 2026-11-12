#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=48:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# CPU phase fits of deviation families (phase_fit.py), one process per (point, family), all in parallel.
# The GPU is not used; the job is here for the node's CPUs. Expected coefficients first (expected_values.py).
# Submit:
#   qsub -N check_models -v POINTS="1 2 3 10 19",FAMS="A B" -o logs/pbs_cpu.out -e logs/pbs_cpu.err batch_cpu.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/check_models

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    cd $D
    nproc > logs/pbs_cpu_nproc.txt
    for p in $POINTS; do
        python -u expected_values.py --point \$p > logs/expected_idx\$p.log 2>&1 &
    done
    wait
    for p in $POINTS; do
        for f in $FAMS; do
            python -u phase_fit.py --point \$p --family \$f > logs/phase_fit_idx\${p}_\$f.log 2>&1 &
        done
    done
    wait
"
