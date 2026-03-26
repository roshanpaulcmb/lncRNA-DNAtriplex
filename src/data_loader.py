from __future__ import annotations

"""
data_loader.py

Loads:
  1. Mutation data from one or more cBioPortal MAF files (data_mutations.txt)
  2. Chromosome name map from a single NCBI assembly report
  3. Healthy gene sequences from NCBI RefSeq genome FASTA + GTF annotation
  4. lncRNA sequences from a LNCipedia FASTA file

Usage:
  python data_loader.py \
    --mutations  path/to/data_mutations.txt [path/to/more.txt ...] \
    --genome     path/to/GCF_000001405.40_GRCh38.p14_genomic.fna.gz \
    --annotation path/to/GCF_000001405.40_GRCh38.p14_genomic.gtf.gz \
    --assembly   path/to/GCF_000001405.40_GRCh38.p14_assembly_report.txt \
    --lncrna     path/to/lncipedia_5_2_hc.fasta
"""

import argparse
import gzip
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from pyfaidx import Fasta


# ---------------------------------------------------------------------------
# 1. Load mutation data from one or more MAF files
# ---------------------------------------------------------------------------

MAF_COLS = [
    "Hugo_Symbol",
    "Entrez_Gene_Id",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Consequence",
    "Variant_Classification",
    "Variant_Type",
]


def load_mutations(filepaths: list[str]) -> tuple[dict, set]:
    """
    Parse one or more cBioPortal MAF files.

    Returns
    -------
    mutated_dict : dict
        {
          Hugo_Symbol: {
            "entrez_id":  int | None,
            "chromosome": str,
            "mutations": [
                {
                  "start":         int,
                  "end":           int,
                  "consequence":   str,
                  "variant_class": str,
                  "variant_type":  str,
                },
                ...
            ]
          }
        }
    gene_names : set[str]
        Flat set of all Hugo symbols — convenient for downstream filtering.
    """
    frames = []
    for fp in filepaths:
        p = Path(fp)
        if not p.exists():
            print(f"[WARNING] Mutation file not found, skipping: {fp}", file=sys.stderr)
            continue
        df = pd.read_csv(
            p,
            sep="\t",
            comment="#",
            low_memory=False,
            usecols=lambda c: c in MAF_COLS,
        )
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No valid mutation files were loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["Hugo_Symbol"])
    combined["Hugo_Symbol"] = combined["Hugo_Symbol"].str.strip()

    mutated_dict: dict = {}

    for row in combined.itertuples(index=False):
        sym = row.Hugo_Symbol
        if sym not in mutated_dict:
            mutated_dict[sym] = {
                "entrez_id":  _safe_int(getattr(row, "Entrez_Gene_Id", None)),
                "chromosome": str(getattr(row, "Chromosome", "")).strip(),
                "mutations":  [],
            }

        mut_record = {
            "start":         _safe_int(getattr(row, "Start_Position", None)),
            "end":           _safe_int(getattr(row, "End_Position",   None)),
            "consequence":   str(getattr(row, "Consequence",            "")).strip(),
            "variant_class": str(getattr(row, "Variant_Classification", "")).strip(),
            "variant_type":  str(getattr(row, "Variant_Type",           "")).strip(),
        }
        mutated_dict[sym]["mutations"].append(mut_record)

    gene_names = set(mutated_dict.keys())
    print(f"[INFO] Loaded {len(gene_names)} unique mutated genes from "
          f"{len(filepaths)} file(s).")
    return mutated_dict, gene_names


# ---------------------------------------------------------------------------
# 2. Load chromosome name map from a single NCBI assembly report
# ---------------------------------------------------------------------------

def load_chrom_map(assembly_report_path: str) -> dict[str, str]:
    """
    Parse a single NCBI assembly report to map simple chromosome names
    to RefSeq accessions. e.g. '1' -> 'NC_000001.11', 'X' -> 'NC_000023.11'

    One assembly report covers the entire reference genome — this file is
    independent of how many mutation cohorts are loaded.

    Parameters
    ----------
    assembly_report_path : str
        Path to the NCBI assembly report .txt file.

    Returns
    -------
    chrom_map : dict[str, str]
        Keys include both bare ('1') and prefixed ('chr1') forms for each
        chromosome, so callers don't need to normalise before lookup.
    """
    p = Path(assembly_report_path)
    if not p.exists():
        raise FileNotFoundError(f"Assembly report not found: {assembly_report_path}")

    chrom_map: dict[str, str] = {}
    with open(p, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            # Columns: Sequence-Name, Sequence-Role, Assigned-Molecule,
            #          Assigned-Molecule-Location/Type, GenBank-Accn,
            #          Relationship, RefSeq-Accn, ...
            seq_name   = parts[0].strip()   # e.g. '1', 'X', 'MT'
            refseq_acc = parts[6].strip()   # e.g. 'NC_000001.11'
            if refseq_acc and refseq_acc != "na":
                chrom_map[seq_name]         = refseq_acc
                chrom_map[f"chr{seq_name}"] = refseq_acc  # 'chr1' style

    print(f"[INFO] Loaded {len(chrom_map)} chromosome mappings from {p.name}.")
    return chrom_map


# ---------------------------------------------------------------------------
# 3. Load healthy (reference) gene sequences from NCBI RefSeq FASTA + GTF
# ---------------------------------------------------------------------------

def ensure_bgzipped(fasta_path: str) -> str:
    """
    If the FASTA is gzip-compressed (not BGZF), recompress it with bgzip.
    Returns the path to the bgzipped file (same path, rewritten in place).
    """
    p = Path(fasta_path)
    if p.suffix != ".gz":
        return fasta_path

    with open(p, "rb") as f:
        magic = f.read(4)
    is_bgzf = magic[:2] == b"\x1f\x8b" and magic[3:4] == b"\x04"

    if is_bgzf:
        print("[INFO] FASTA is already BGZF compressed.")
        return fasta_path

    print("[INFO] Recompressing FASTA with bgzip (this will take a while for large files)…")
    tmp = p.with_suffix("")  # e.g. file.fna.gz -> file.fna
    subprocess.run(f"gunzip -kf {p} && bgzip -f {tmp}",
                   shell=True, check=True, env=os.environ.copy())
    print(f"[INFO] Recompressed: {p}")
    return str(p)


def load_healthy_sequences(
    fasta_path: str,
    gtf_path: str,
    gene_names: set[str],
    chrom_map: dict[str, str],
) -> dict[str, str]:
    """
    Extract reference gene body sequences for genes in `gene_names`.

    Strategy
    --------
    Parse the GTF for gene-level intervals (gene feature type), then use
    pyfaidx to pull the sequence from the genomic FASTA.

    Returns
    -------
    healthy_dict : {Hugo_Symbol: nucleotide_sequence (str)}

    Notes
    -----
    - The NCBI GTF uses `gene_name` / `gene` attributes for the Hugo symbol.
    - Only the longest interval found per gene is used.
    - Sequences on the minus strand are reverse-complemented automatically.
    """
    fasta_path = ensure_bgzipped(fasta_path)
    print("[INFO] Indexing genome FASTA (this may take a moment the first time)…")
    genome = Fasta(fasta_path, sequence_always_upper=True)

    print("[INFO] Parsing GTF annotation…")
    gene_intervals = _parse_gtf_for_genes(gtf_path, gene_names)

    healthy_dict: dict[str, str] = {}
    missing = []

    for sym, intervals in gene_intervals.items():
        chrom, start, end, strand = max(intervals, key=lambda x: x[2] - x[1])
        refseq_chrom = chrom_map.get(chrom, chrom)
        try:
            seq = str(genome[refseq_chrom][start - 1 : end])
            if strand == "-":
                seq = _reverse_complement(seq)
            healthy_dict[sym] = seq
        except KeyError:
            missing.append(sym)

    if missing:
        print(f"[WARNING] Chromosome not found in FASTA for {len(missing)} genes: "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}", file=sys.stderr)

    not_found = gene_names - set(healthy_dict)
    if not_found:
        print(f"[WARNING] {len(not_found)} genes had no GTF entry: "
              f"{list(not_found)[:10]}{'...' if len(not_found) > 10 else ''}",
              file=sys.stderr)

    print(f"[INFO] Retrieved reference sequences for {len(healthy_dict)} genes.")
    return healthy_dict


def _parse_gtf_for_genes(
    gtf_path: str,
    gene_names: set[str],
) -> dict[str, list[tuple]]:
    """
    Returns {hugo_symbol: [(chrom, start, end, strand), ...]}
    Reads only 'gene' feature rows to keep memory light.
    """
    intervals: dict[str, list] = defaultdict(list)
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom  = parts[0]
            start  = int(parts[3])
            end    = int(parts[4])
            strand = parts[6]
            attrs  = _parse_gtf_attributes(parts[8])
            sym    = attrs.get("gene_name") or attrs.get("gene") or attrs.get("gene_id", "")
            if sym in gene_names:
                intervals[sym].append((chrom, start, end, strand))
    return dict(intervals)


def _parse_gtf_attributes(attr_str: str) -> dict[str, str]:
    """Parse GTF attribute column into a dict."""
    attrs = {}
    for field in attr_str.strip().split(";"):
        field = field.strip()
        if not field:
            continue
        parts = field.split(" ", 1)
        if len(parts) == 2:
            attrs[parts[0]] = parts[1].strip('"')
    return attrs


# ---------------------------------------------------------------------------
# 4. Load lncRNA sequences from LNCipedia FASTA
# ---------------------------------------------------------------------------

def load_lncrnas(fasta_path: str) -> dict[str, str]:
    """
    Parse a LNCipedia FASTA file.

    LNCipedia FASTA headers look like:
      >lnc-GENE-1:1   or   >LINC01234:2   (transcript_id : version)

    Returns
    -------
    lncrna_dict : {transcript_id (str): sequence (str)}
    """
    p = Path(fasta_path)
    if not p.exists():
        raise FileNotFoundError(f"lncRNA FASTA not found: {fasta_path}")

    lncrna_dict: dict[str, str] = {}
    for record in SeqIO.parse(str(p), "fasta"):
        tid = record.id.split()[0]
        lncrna_dict[tid] = str(record.seq).upper()

    print(f"[INFO] Loaded {len(lncrna_dict)} lncRNA transcripts.")
    return lncrna_dict


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]

def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Load mutation, reference genome, and lncRNA data."
    )
    p.add_argument("--mutations",   nargs="+", required=True, metavar="MAF_FILE",
                   help="One or more cBioPortal data_mutations.txt files.")
    p.add_argument("--genome",      required=True, metavar="FASTA",
                   help="NCBI RefSeq genomic FASTA (.fna or .fna.gz).")
    p.add_argument("--annotation",  required=True, metavar="GTF",
                   help="Matching NCBI RefSeq GTF annotation file.")
    p.add_argument("--assembly",    required=True, metavar="ASSEMBLY_REPORT",
                   help="NCBI assembly report .txt file (one per reference genome).")
    p.add_argument("--lncrna",      required=True, metavar="FASTA",
                   help="LNCipedia FASTA file.")
    return p.parse_args()


def main():
    args = parse_args()

    mutated_dict, gene_names = load_mutations(args.mutations)
    chrom_map                = load_chrom_map(args.assembly)
    healthy_dict             = load_healthy_sequences(
                                   fasta_path  = args.genome,
                                   gtf_path    = args.annotation,
                                   gene_names  = gene_names,
                                   chrom_map   = chrom_map,
                               )
    lncrna_dict              = load_lncrnas(args.lncrna)

    print("\n=== Data Loading Summary ===")
    print(f"  Mutated genes:           {len(mutated_dict)}")
    print(f"  Healthy seqs retrieved:  {len(healthy_dict)}")
    print(f"  lncRNA transcripts:      {len(lncrna_dict)}")

    return mutated_dict, healthy_dict, lncrna_dict


if __name__ == "__main__":
    main()