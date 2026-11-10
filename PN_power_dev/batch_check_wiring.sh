#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -N pn_power_wiring
#PBS -l walltime=01:00:00
#PBS -l select=1:ngpus=1:mem=64gb
#PBS -k oed
#PBS -o /nfs/home/svu/e1583490/EMRI_CV_bias/PN_power_dev/check_waveform_wiring.output
#PBS -e /nfs/home/svu/e1583490/EMRI_CV_bias/PN_power_dev/check_waveform_wiring.error

# Waveform and SEF wiring of the power-law deviation, idx4, T 0.25 yr.
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/PN_power_dev

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    git -C /home/svu/e1583490/packages_to_install/SuperKludge_r status -sb | head -3 > check_waveform_wiring.log 2>&1
    python -u check_waveform_wiring.py --point 4 --T 0.25 >> check_waveform_wiring.log 2>&1
"
