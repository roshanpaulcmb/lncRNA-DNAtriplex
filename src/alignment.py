"""
alignment.py – LongTarget: lncRNA–DNA triplex alignment pipeline

Wires together:
  rules.py      → transfer_string, reverse_seq, complement
  stats.py      → calc_score
  sim.py        → SIM, Triplex, cluster_triplex, print_cluster
  data_loader.py→ load_mutations, load_chrom_map,
                  load_healthy_sequences, load_lncrnas

Usage:
  python alignment.py \
    --genome      GCF_000001405.40_GRCh38.p14_genomic.fna.gz \
    --annotation  GCF_000001405.40_GRCh38.p14_genomic.gtf.gz \
    --assembly    GCF_000001405.40_GRCh38.p14_assembly_report.txt \
    --mutations   ucec_msk_2024/data_mutations.txt \
    --lncrna      lncipedia_5_2_hc.fasta \
    --outpath     ./results

Dictionary schemas (from data_loader.py)
-----------------------------------------
mutated_dict  = {
    Hugo_Symbol: {
        "entrez_id":  int | None,
        "chromosome": str,
        "mutations":  [{"start", "end", "consequence",
                        "variant_class", "variant_type"}, ...]
    }
}
healthy_dict  = { Hugo_Symbol:   sequence_str }
lncrna_dict   = { transcript_id: sequence_str }
"""

import sys
import os
import argparse
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

from rules       import transfer_string, reverse_seq, complement
from stats       import calc_score
from sim         import SIM, Triplex, cluster_triplex, print_cluster
from data_loader import (load_mutations, load_chrom_map,
                         load_healthy_sequences, load_lncrnas)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass  (alignment parameters)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Para:
    rule:           int   = 0          # 0 = all rules
    cut_length:     int   = 5000       # DNA chunk size
    strand:         int   = 0          # 0=both, >0=parallel only, <0=anti only
    overlap_length: int   = 100        # overlap between consecutive DNA chunks
    min_score:      int   = 0
    detail_output:  bool  = False
    nt_min:         int   = 20         # minimum triplex length (nt)
    nt_max:         int   = 100000     # maximum triplex length (nt)
    score_min:      float = 0.0
    min_identity:   float = 60.0       # minimum % identity to keep
    min_stability:  float = 1.0        # minimum mean stability score
    penalty_t:      int   = -1000      # penalty for consecutive A's
    penalty_c:      int   = 0          # penalty for consecutive G's
    c_distance:     int   = 15         # clustering distance
    c_length:       int   = 50         # clustering minimum length


# ─────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────────────────────────

def cut_sequence(seq: str, cut_length: int,
                 overlap_length: int) -> Tuple[List[str], List[int], int]:
    """Split a long DNA sequence into overlapping chunks."""
    seqs_vec:       List[str] = []
    seqs_start_pos: List[int] = []
    pos = 0
    while pos < len(seq):
        seqs_vec.append(seq[pos: pos + cut_length])
        seqs_start_pos.append(pos)
        pos += cut_length - overlap_length
    return seqs_vec, seqs_start_pos, len(seqs_vec)


def same_seq(seq: str) -> bool:
    """Return True if the sequence is a homopolymer (uninformative)."""
    counts = {"A": 0, "C": 0, "G": 0, "T": 0, "U": 0, "N": 0}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
        else:
            print(f"Warning: unknown base '{ch}'")
    n = len(seq)
    return n > 0 and any(v == n for v in counts.values())


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_strand(reverse: int, strand: int) -> str:
    if   reverse == 0 and strand ==  1: return "ParaPlus"
    elif reverse == 1 and strand ==  1: return "ParaMinus"
    elif reverse == 1 and strand == -1: return "AntiMinus"
    elif reverse == 0 and strand == -1: return "AntiPlus"
    return ""


def comp_key(t: Triplex) -> int:
    return t.motif


# ─────────────────────────────────────────────────────────────────────────────
# LongTarget core
# ─────────────────────────────────────────────────────────────────────────────

def long_target(para: Para, rna_sequence: str,
                dna_sequence: str) -> List[Triplex]:
    """
    Main alignment loop.

    For every DNA chunk and every applicable triplex rule:
      1. Transfer (encode) the DNA chunk via the rule            ← rules.py
      2. Compute the statistical significance threshold          ← stats.py
      3. Run SIM local alignment and collect triplex candidates  ← sim.py

    Returns the filtered list of Triplex objects.
    """
    chunks, start_positions, _ = cut_sequence(
        dna_sequence, para.cut_length, para.overlap_length)

    triplex_list: List[Triplex] = []

    for i, dna_chunk in enumerate(chunks):
        dna_start = start_positions[i]
        print(f"  DNA chunk @ {dna_start} ({len(dna_chunk)} bp) ...")

        if same_seq(dna_chunk):
            print("  Skipped (homopolymer)")
            continue

        if para.strand >= 0:
            for j in (range(1, 7) if para.rule == 0 else [para.rule]):
                if para.rule != 0 and not (0 < para.rule < 7):
                    break
                seq2 = transfer_string(dna_chunk, 0, 1, j)
                min_sc = calc_score(rna_sequence, seq2, dna_start, j)
                SIM(rna_sequence, seq2, dna_chunk, dna_start, min_sc,
                    5, -4, -12, -4, triplex_list, 0, 1, j,
                    para.nt_min, para.nt_max, para.penalty_t, para.penalty_c)

                seq2 = reverse_seq(transfer_string(dna_chunk, 1, 1, j))
                min_sc = calc_score(rna_sequence, seq2, dna_start, j)
                SIM(rna_sequence, seq2, dna_chunk, dna_start, min_sc,
                    5, -4, -12, -4, triplex_list, 1, 1, j,
                    para.nt_min, para.nt_max, para.penalty_t, para.penalty_c)

        if para.strand <= 0:
            for j in (range(1, 19) if para.rule == 0 else [para.rule]):
                seq2 = transfer_string(dna_chunk, 0, -1, j)
                min_sc = calc_score(rna_sequence, seq2, dna_start, j)
                SIM(rna_sequence, seq2, dna_chunk, dna_start, min_sc,
                    5, -4, -12, -4, triplex_list, 0, -1, j,
                    para.nt_min, para.nt_max, para.penalty_t, para.penalty_c)

                seq2 = reverse_seq(transfer_string(dna_chunk, 1, -1, j))
                min_sc = calc_score(rna_sequence, seq2, dna_start, j)
                SIM(rna_sequence, seq2, dna_chunk, dna_start, min_sc,
                    5, -4, -12, -4, triplex_list, 1, -1, j,
                    para.nt_min, para.nt_max, para.penalty_t, para.penalty_c)

    filtered = [
        t for t in triplex_list
        if (t.score     >= para.score_min    and
            t.identity  >= para.min_identity and
            t.tri_score >= para.min_stability)
    ]
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def report(
    results:      Dict[str, Dict[str, Dict[str, List[Triplex]]]],
    mutated_dict: Dict,
    outpath:      str,
    report_name:  str = None,
) -> None:
    """
    Flatten the nested results dict and write a single CSV summary.

    Output columns
    --------------
    lnc_id, gene, condition, variant_class, chromosome,
    n_triplexes, mean_identity, mean_stability, mean_score,
    min_nt, max_nt
    """
    os.makedirs(outpath, exist_ok=True)
    if report_name is None:
        report_name = f"triplex_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = os.path.join(outpath, report_name)

    fieldnames = [
        "lnc_id", "gene", "condition", "variant_class", "chromosome",
        "n_triplexes", "mean_identity", "mean_stability", "mean_score",
        "min_nt", "max_nt",
    ]

    rows = []
    for lnc_id, gene_dict in results.items():
        for gene, condition_dict in gene_dict.items():
            gene_info  = mutated_dict.get(gene, {})
            chromosome = gene_info.get("chromosome", "")
            mutations  = {
                f"mut{i}_{m.get('variant_class', 'unknown')}": m.get("variant_class", "")
                for i, m in enumerate(gene_info.get("mutations", []))
            }

            for condition, triplexes in condition_dict.items():
                n = len(triplexes)
                if n > 0:
                    mean_identity  = sum(t.identity  for t in triplexes) / n
                    mean_stability = sum(t.tri_score for t in triplexes) / n
                    mean_score     = sum(t.score     for t in triplexes) / n
                    min_nt         = min(t.nt        for t in triplexes)
                    max_nt         = max(t.nt        for t in triplexes)
                else:
                    mean_identity = mean_stability = mean_score = 0.0
                    min_nt = max_nt = 0

                rows.append({
                    "lnc_id":         lnc_id,
                    "gene":           gene,
                    "condition":      condition,
                    "variant_class":  mutations.get(condition, "healthy" if condition == "healthy" else ""),
                    "chromosome":     chromosome,
                    "n_triplexes":    n,
                    "mean_identity":  round(mean_identity,  4),
                    "mean_stability": round(mean_stability, 4),
                    "mean_score":     round(mean_score,     4),
                    "min_nt":         min_nt,
                    "max_nt":         max_nt,
                })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Report written → {out_path}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str]) -> Tuple[Para, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="LongTarget lncRNA triplex pipeline")

    # ── data loading ──────────────────────────────────────────────────────
    parser.add_argument("--mutations",   nargs="+", required=True, metavar="MAF_FILE",
                        help="One or more cBioPortal data_mutations.txt files.")
    parser.add_argument("--genome",      required=True, metavar="FASTA",
                        help="NCBI RefSeq genomic FASTA (.fna or .fna.gz).")
    parser.add_argument("--annotation",  required=True, metavar="GTF",
                        help="Matching NCBI RefSeq GTF annotation file.")
    parser.add_argument("--assembly",    required=True, metavar="ASSEMBLY_REPORT",
                        help="NCBI assembly report .txt file (one per reference genome).")
    parser.add_argument("--lncrna",      required=True, metavar="FASTA",
                        help="LNCipedia FASTA file.")
    parser.add_argument("--outpath",     default="./results",
                        help="Output directory (default: ./results).")

    # ── alignment parameters ──────────────────────────────────────────────
    parser.add_argument("-r",  dest="rule",           type=int,   default=0)
    parser.add_argument("-c",  dest="cut_length",     type=int,   default=5000)
    parser.add_argument("-o",  dest="overlap_length", type=int,   default=100)
    parser.add_argument("-t",  dest="strand",         type=int,   default=0)
    parser.add_argument("-m",  dest="min_score",      type=int,   default=0)
    parser.add_argument("-i",  dest="min_identity",   type=float, default=60.0)
    parser.add_argument("-S",  dest="min_stability",  type=float, default=1.0)
    parser.add_argument("-ni", dest="nt_min",         type=int,   default=20)
    parser.add_argument("-na", dest="nt_max",         type=int,   default=100000)
    parser.add_argument("-pt", dest="penalty_t",      type=int,   default=-1000)
    parser.add_argument("-pc", dest="penalty_c",      type=int,   default=0)
    parser.add_argument("-ds", dest="c_distance",     type=int,   default=15)
    parser.add_argument("-lg", dest="c_length",       type=int,   default=50)
    parser.add_argument("-d",  dest="detail_output",  action="store_true")

    args = parser.parse_args(argv[1:])

    para = Para(
        rule           = args.rule,
        cut_length     = args.cut_length,
        overlap_length = args.overlap_length,
        strand         = args.strand,
        min_score      = args.min_score,
        min_identity   = args.min_identity,
        min_stability  = args.min_stability,
        nt_min         = args.nt_min,
        nt_max         = args.nt_max,
        penalty_t      = args.penalty_t,
        penalty_c      = args.penalty_c,
        c_distance     = args.c_distance,
        c_length       = args.c_length,
        detail_output  = args.detail_output,
    )
    return para, args


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] = None) -> int:
    if argv is None:
        argv = sys.argv

    para, args = parse_args(argv)

    # ── ensure output directory exists ───────────────────────────────────
    os.makedirs(args.outpath, exist_ok=True)
    print(f"Output directory: {args.outpath}")

    # ── load dictionaries ─────────────────────────────────────────────────
    print("Loading mutated gene data ...")
    mutated_dict, gene_names = load_mutations(args.mutations)

    print("Building chromosome map ...")
    chrom_map = load_chrom_map(args.assembly)

    print("Loading healthy reference sequences ...")
    healthy_dict = load_healthy_sequences(
        fasta_path = args.genome,
        gtf_path   = args.annotation,
        gene_names = gene_names,
        chrom_map  = chrom_map,
    )

    print("Loading lncRNA sequences ...")
    lncrna_dict = load_lncrnas(args.lncrna)

    print(f"\n=== Data Summary ===")
    print(f"  Mutated genes:    {len(mutated_dict)}")
    print(f"  Healthy seqs:     {len(healthy_dict)}")
    print(f"  lncRNA entries:   {len(lncrna_dict)}")

    # ── prepare gene/lncRNA lists ─────────────────────────────────────────
    shared_genes = sorted(g for g in gene_names if g in healthy_dict)
    lncrna_ids   = list(lncrna_dict.keys())

    TEST_LNCRNA  = 100
    test_lncrnas = lncrna_ids[:TEST_LNCRNA]

    print(f"\n  Genes available for analysis: {len(shared_genes)}")
    print(f"  lncRNAs to test:              {len(test_lncrnas)}")
    print()

    # ── main loop ─────────────────────────────────────────────────────────
    # results : {lnc_id: {gene: {"healthy": [Triplex], mut_label: [Triplex], ...}}}
    loop_start = datetime.now()
    print(f"Loop started at: {loop_start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results:     Dict[str, Dict[str, Dict[str, List[Triplex]]]] = {}
    total_pairs = 0
    total_hits  = 0

    for lnc_idx, lnc_id in enumerate(test_lncrnas):
        lnc_seq = lncrna_dict[lnc_id]
        print(f"[{lnc_idx + 1}/{len(test_lncrnas)}] lncRNA: {lnc_id} "
              f"({len(lnc_seq)} nt)")
        results[lnc_id] = {}

        for gene in shared_genes:
            healthy_seq = healthy_dict[gene]
            gene_info   = mutated_dict[gene]
            results[lnc_id][gene] = {}

            # ── healthy run ───────────────────────────────────────────────
            healthy_triplexes = long_target(para, lnc_seq, healthy_seq)
            results[lnc_id][gene]["healthy"] = healthy_triplexes
            total_hits += len(healthy_triplexes)

            # ── mutated runs ──────────────────────────────────────────────
            for mut_idx, mut in enumerate(gene_info.get("mutations", [])):
                mut_label = f"mut{mut_idx}_{mut.get('variant_class', 'unknown')}"
                mutated_triplexes = long_target(para, lnc_seq, healthy_seq)
                results[lnc_id][gene][mut_label] = mutated_triplexes
                total_hits += len(mutated_triplexes)

            total_pairs += 1

    loop_end = datetime.now()
    elapsed  = loop_end - loop_start

    print(f"\nLoop ended at:   {loop_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time:    {elapsed}")
    print(f"Total (lncRNA × gene) pairs processed: {total_pairs}")
    print(f"Total triplex hits across all runs:     {total_hits}")

    # ── write report ──────────────────────────────────────────────────────
    report(results, mutated_dict, args.outpath)

    print("\nFinished normally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())