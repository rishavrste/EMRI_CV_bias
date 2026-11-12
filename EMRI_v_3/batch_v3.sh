#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=12:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# forms_v3.py: rate fit of every form at all 50 grid tracks, then the dephasing test at EMRI idx0, 4, 7, 19 and
# IMRI idx0, 4, in parallel. CPU only. Needs SuperKludge_r on PN_v_3 and check_models/results/track_*.npz.
#   qsub -N forms_v3 -o logs/pbs_v3.out -e logs/pbs_v3.err batch_v3.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/EMRI_v_3

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    cd $D
    python -u forms_v3.py --rates > logs/rates.log 2>&1 &
    for gp in 'EMRI 0' 'EMRI 4' 'EMRI 7' 'EMRI 19' 'IMRI 0' 'IMRI 4'; do
        set -- \$gp
        python -u forms_v3.py --dephase --grid \$1 --point \$2 > logs/dephase_\$1_idx\$2.log 2>&1 &
    done
    wait
"
