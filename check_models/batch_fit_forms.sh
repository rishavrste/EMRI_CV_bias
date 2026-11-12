#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=12:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# fit_forms.py: rate-level fit of every form at every grid track (needs pn_orders.py's tracks), then the dephasing
# test at a few points, in parallel. CPU only. Submit after the pn_orders job:
#   qsub -W depend=afterok:<pn_orders job> -N fit_forms -o logs/pbs_fit_forms.out -e logs/pbs_fit_forms.err batch_fit_forms.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/check_models
FORMS="B J5 PL6 SVD_1 SVD_2 SVD_3 SVD_4 PL LIN PADE"

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    cd $D
    python -u fit_forms.py --rates > logs/fit_forms_rates.log 2>&1 &
    for gp in 'EMRI 0' 'EMRI 4' 'EMRI 7' 'EMRI 19' 'IMRI 4' 'IMRI 0'; do
        set -- \$gp
        python -u fit_forms.py --dephase --grid \$1 --point \$2 --forms $FORMS > logs/fit_forms_dephase_\$1_idx\$2.log 2>&1 &
    done
    wait
"
