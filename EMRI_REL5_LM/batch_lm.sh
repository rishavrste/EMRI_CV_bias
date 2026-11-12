#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=72:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# LM of the 0PA + REL5_1 template (run_lm.py) at the EMRI points in $POINTS ("0:1:2", one after another): full T,
# then the T ladder if that ends below the 0PA floor. Needs SuperKludge_r on PN_v_3 and results/seeds.json
# (make_seeds.py). Each point logs to logs/lm_idx{i}.log. Submit with:
#   qsub -N rel5_a -v POINTS=0:1:2:3:4 -o logs/pbs_lm_a.out -e logs/pbs_lm_a.err batch_lm.sh
module load singularity
D=/home/svu/e1583490/EMRI_CV_bias/EMRI_REL5_LM

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    for p in \$(echo $POINTS | tr ':' ' '); do
        LOG=$D/logs/lm_idx\$p.log
        python -u -c 'import few; print(few.__file__)' > \$LOG 2>&1
        python -u run_lm.py --points \$p >> \$LOG 2>&1
    done
"
