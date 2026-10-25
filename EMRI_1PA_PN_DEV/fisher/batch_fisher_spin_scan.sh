#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=04:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# PN Fisher at a point's best fit with only the spin swapped (fisher_spin_scan.py). POINT (default 22),
# SPINS hyphen-free, space-separated in quotes (default 0 0.001 -0.001 0.01 -0.01), TEMPLATE (default pn).
# Submit from fisher/ with:
#   qsub -N fisher_spin22 -o ../logs/pbs_fisher_spin_scan.out -e ../logs/pbs_fisher_spin_scan.err batch_fisher_spin_scan.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/EMRI_1PA_PN_DEV
mkdir -p $D/logs
POINT=${POINT:-22}
TEMPLATE=${TEMPLATE:-pn}
LOG=$D/logs/fisher_spin_scan_${TEMPLATE}_idx${POINT}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D/fisher
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u fisher_spin_scan.py --template $TEMPLATE --point $POINT ${SPINS:+--spins $SPINS} >> $LOG 2>&1
"
