#!/bin/bash
#SBATCH --job acbc_mskcc_2015_HOTAIR
#SBATCH --output ./logs/acbc_mskcc_2015_HOTAIR.stdout
#SBATCH --error ./logs/acbc_mskcc_2015_HOTAIR.stderr
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
    --mutations ./data/acbc_mskcc_2015/data_mutations.txt \
    --genome ./data/GCF_000001405.25_GRCh37.p13_genomic.fna.gz \
    --annotation ./data/GCF_000001405.25_GRCh37.p13_genomic.gtf.gz \
    --assembly ./data/GCF_000001405.25_GRCh37.p13_assembly_report.txt \
    --lncrna ./data/lncipedia_5_2_hc.fasta \
    --lncrna_ids HOTAIR \
    --n_cores 32 \
    --outpath ./results/acbc_mskcc_2015_HOTAIR \
    --report_name acbc_mskcc_2015_HOTAIR.csv
