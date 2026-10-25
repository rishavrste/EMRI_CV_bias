#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -N fisher_2g_ecc
#PBS -l walltime=12:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/second_gen_TDI/results/fisher_bestfit_2g.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/second_gen_TDI/results/fisher_bestfit_2g.error
#PBS -k oed

cd /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/second_gen_TDI/results
module load singularity

singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/second_gen_TDI/results
    python fisher_bestfit_2g.py > fisher_bestfit_2g.log 2>&1 && python plot_fisher_bestfit.py >> fisher_bestfit_2g.log 2>&1 && python report_fisher_bestfit_2g.py >> fisher_bestfit_2g.log 2>&1 && python export_fishers_2g.py >> fisher_bestfit_2g.log 2>&1
    wait
"
