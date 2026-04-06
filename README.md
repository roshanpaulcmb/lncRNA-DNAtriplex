# lncRNA-DNAtriplex

A computational pipeline for predicting lncRNA–DNA triplex formation at somatic mutation sites in cancer genomes.

## Overview

This pipeline investigates how cancer-associated somatic mutations alter the capacity of long non-coding RNAs (lncRNAs) to form DNA triplexes at gene loci. For each mutated gene identified in a patient cohort, the pipeline extracts both the reference (healthy) and mutation-applied (somatic) DNA sequences, then screens all sequences against a high-confidence lncRNA library for triplex-forming potential. Comparing triplex binding profiles between healthy and mutated sequences can reveal whether a given mutation disrupts or creates lncRNA regulatory binding sites.

## Pipeline

1. **Mutation loading** — Parses one or more cBioPortal MAF files (`data_mutations.txt`) to extract somatic variants (SNPs, indels, DNPs) per gene.
2. **Sequence extraction** — Retrieves reference gene body sequences from an NCBI RefSeq genome FASTA + GTF annotation, then applies each somatic mutation in-silico to generate per-mutation sequences.
3. **lncRNA loading** — Parses a LNCipedia high-confidence FASTA to build a transcript sequence library.
4. **Triplex alignment** — Runs the LongTarget algorithm (parasail-accelerated Smith-Waterman + MLE statistical threshold) to score lncRNA–DNA triplex-forming potential for each lncRNA × gene × condition (healthy / mutated) combination.
5. **Reporting** — Outputs a CSV summary with triplex counts, mean identity, stability, and score per lncRNA–gene–condition triple.

## Acknowledgements

The triplex alignment logic in this pipeline is a Python reimplementation of [LongTarget](https://github.com/LongTarget/LongTarget), a C++ tool for predicting lncRNA/DNA binding motifs and binding sites based on Hoogsteen and reverse Hoogsteen base-pairing rules, distributed under the AGPLv3 license. The core algorithms — sequence transfer rules, Smith-Waterman local alignment, statistical score thresholding, and triplex clustering — are adapted from the original LongTarget codebase. This project is also distributed under AGPLv3 in accordance with those terms.

## Dependencies

- Python ≥ 3.10
- `biopython`, `pyfaidx`, `pandas`, `parasail`
- `bgzip` (for FASTA recompression)

See `environment.yml` for the full conda environment.

## Usage

```bash
python src/alignment.py \
  --mutations    data_mutations.txt \
  --clinical     data_clinical_sample.txt \
  --genome       genomic.fna.gz \
  --annotation   genomic.gtf.gz \
  --assembly     assembly_report.txt \
  --lncrna       lncipedia.fasta \
  --lncrna_ids   LINC01234 HOTAIR \
  --mode         window \
  --outpath      ./results \
  --report_name  my_results.csv \
  -n_cores       8
```

The pipeline supports multiprocessing via the `-n_cores` flag, distributing lncRNA–gene pair computations across multiple CPU cores to significantly reduce runtime on large datasets. See `example.ipynb` for a walkthrough.

## Data Sources

| Data | Source |
|---|---|
| Somatic mutations | [cBioPortal](https://www.cbioportal.org/) |
| Reference genome | [NCBI RefSeq](https://www.ncbi.nlm.nih.gov/refseq/) (GRCh37 / GRCh38) |
| lncRNA sequences | [LNCipedia](https://lncipedia.org/) |
