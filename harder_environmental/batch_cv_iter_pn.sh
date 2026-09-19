#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N iter_pn
#PBS -l walltime=12:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/harder_environmental/iter_pn.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/harder_environmental/iter_pn.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/harder_environmental
    python cv_iterate.py --model 0PA+PN --tag pn --seed-results results_cv_de_cv_0pa.json --seed-stage stage2_de --iters 40 --patience 8 > iter_pn.log 2>&1
    wait
"
