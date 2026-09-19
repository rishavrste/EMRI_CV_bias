#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N h13_pn_ecc
#PBS -l walltime=96:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/harder_environmental/pn_s13_ecc.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/harder_environmental/pn_s13_ecc.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
    python cv_de_cv.py --model 0PA+PN --point-idx 13 --tag pn_s13_ecc --fix-params none --full-phase-bounds --seed-from results_cv_de_cv_pn_s13_inj5.json > pn_s13_ecc.log 2>&1
    wait
"
