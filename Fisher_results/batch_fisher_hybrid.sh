#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N fisher_hybrid
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/Fisher_results/fisher_hybrid.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/Fisher_results/fisher_hybrid.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/Fisher_results
module load singularity

# Requires SuperKludge_r on the 'hybrid' branch (covers 0PA, PN, simple models
# for every case in results_combined.txt / results_compiled.txt). Resumable:
# already-saved Fisher_*.npy / SNR_*.npy files are skipped, so a walltime kill
# and requeue just picks up where it left off.
singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/Fisher_results
    python compute_fishers.py --branch hybrid > compute_fishers_hybrid.log 2>&1
    wait
"
