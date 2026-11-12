#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# PN-order importance (pn_orders.py) at all 25 EMRI and 25 IMRI grid points, CPU only, NPAR processes at a time.
# Submit:  qsub -N pn_orders -o logs/pbs_pn_orders.out -e logs/pbs_pn_orders.err batch_pn_orders.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/check_models
NPAR=12

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    cd $D
    for g in EMRI IMRI; do for i in \$(seq 0 24); do echo \$g \$i; done; done |
        xargs -P $NPAR -n 2 sh -c 'python -u pn_orders.py --grid \$0 --point \$1 > logs/pn_orders_\$0_idx\$1.log 2>&1'
"
