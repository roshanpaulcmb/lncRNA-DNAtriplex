#!/bin/bash
#SBATCH --job asclc_msk_2024_MALAT1
#SBATCH --output ./logs/asclc_msk_2024_MALAT1.stdout
#SBATCH --error ./logs/asclc_msk_2024_MALAT1.stderr
#SBATCH --cpus-per-task=32
#SBATCH -t 60:00:00
#SBATCH --mail-user=rop174@pitt.edu
#SBATCH --mail-type=ALL

mkdir -p ./logs

echo Running on `hostname`
nvidia-smi
module avail cuda
eval "$(conda shell.bash hook)"
conda activate lncRNA

python -u ./src/alignment.py \
    --mutations ./data/asclc_msk_2024/data_mutations.txt \
    --genome ./data/GCF_000001405.25_GRCh37.p13_genomic.fna.gz \
    --annotation ./data/GCF_000001405.25_GRCh37.p13_genomic.gtf.gz \
    --assembly ./data/GCF_000001405.25_GRCh37.p13_assembly_report.txt \
    --lncrna ./data/lncipedia_5_2_hc.fasta \
    --lncrna_ids MALAT1 \
    --n_cores 32 \
    --outpath ./results/asclc_msk_2024_MALAT1 \
    --report_name asclc_msk_2024_MALAT1.csv
