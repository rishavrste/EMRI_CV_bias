#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=12:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# phase_fit_v3.py for 0PA, B and REL5_1 at the grid points in $GRID_POINTS ("EMRI-0:EMRI-4:..."), all in parallel.
# CPU only; needs SuperKludge_r on PN_v_3. Submit:
#   qsub -N pf_v3_a -v GRID_POINTS=EMRI-0:EMRI-4:EMRI-7 -o logs/pbs_pf_a.out -e logs/pbs_pf_a.err batch_phase_fit_v3.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/EMRI_v_3

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    cd $D
    for gp in \$(echo $GRID_POINTS | tr ':' ' '); do
        set -- \$(echo \$gp | tr '-' ' ')
        for form in 0PA B REL5_1; do
            python -u phase_fit_v3.py --grid \$1 --point \$2 --form \$form > logs/phase_fit_\$1_idx\$2_\$form.log 2>&1 &
        done
    done
    wait
"
