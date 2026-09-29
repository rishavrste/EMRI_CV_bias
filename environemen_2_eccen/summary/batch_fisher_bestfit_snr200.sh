#!/bin/bash
#PBS -P CFP03-CF-051
#PBS -N fisher_bf200
#PBS -l walltime=06:00:00
#PBS -l select=1:ngpus=1:mem=250gb
#PBS -o /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/summary/fisher_bestfit_snr200.output
#PBS -e /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/summary/fisher_bestfit_snr200.error
#PBS -k oed
cd /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/summary
module load singularity
singularity exec --nv -e --env PARIS_SYSTEM_ID="$PARIS_SYSTEM_ID" \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd /home/svu/e1583490/EMRI_CV_bias/environemen_2_eccen/summary
    python fisher_bestfit.py --snr 200 > fisher_bestfit_snr200.log 2>&1 && python plot_fisher_bestfit.py fisher_bestfit_snr200.json plots_snr200 >> fisher_bestfit_snr200.log 2>&1 && python report_fisher_bestfit.py fisher_bestfit_snr200.json >> fisher_bestfit_snr200.log 2>&1 && python export_fishers.py fisher_bestfit_snr200.json >> fisher_bestfit_snr200.log 2>&1
    wait
"
