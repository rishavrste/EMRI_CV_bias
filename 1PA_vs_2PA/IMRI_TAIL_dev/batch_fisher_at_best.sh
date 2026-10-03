#!/bin/bash
#PBS -P CFP05-CF-131
#PBS -l walltime=06:00:00
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -k oed

# Fisher at the best fits (fisher_at_best.py). IDX="0-1-2" (hyphen-separated, default all 25),
# TEMPLATE="1pa-1pa_dev" (default both).
# Submit with:
#   qsub -N fisher_best -o logs/pbs_fisher_at_best.out -e logs/pbs_fisher_at_best.err batch_fisher_at_best.sh
module load singularity
D=/nfs/home/svu/e1583490/EMRI_CV_bias/1PA_vs_2PA/IMRI_TAIL_dev
mkdir -p $D/logs
IDX=${IDX:-$(seq -s - 0 24)}
TEMPLATE=${TEMPLATE:-1pa-1pa_dev}
LOG=$D/logs/fisher_at_best_${TEMPLATE}.log

singularity exec --nv -e \
/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif \
bash -lc "
    source /home/svu/e1583490/bias_inference_emri/.venv/bin/activate
    cd $D
    python -u -c 'import few; print(few.__file__)' > $LOG 2>&1
    python -u fisher_at_best.py --template $(echo $TEMPLATE | tr - " ") --idx $(echo $IDX | tr - " ") >> $LOG 2>&1
"
