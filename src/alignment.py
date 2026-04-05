"""
alignment.py – LongTarget: lncRNA–DNA triplex alignment pipeline

Wires together:
  rules.py      → transfer_string, reverse_seq, complement
  stats.py      → calc_score
  sim.py        → SIM, Triplex, cluster_triplex, print_cluster
  data_loader.py→ load_mutations, load_chrom_map,
                  load_sequences, load_lncrnas

Usage:
  python alignment.py \
    --genome      genomic.fna.gz \
    --annotation  genomic.gtf.gz \
    --assembly    assembly_report.txt \
    --mutations   data_mutations.txt \
    --lncrna      lncipedia_5_2_hc.fasta \
    --outpath     ./results \
    --lncrna_ids  FENDRR LINC01234

  # Or pass a file with one ID per line:
  python alignment.py ... --lncrna_ids_file my_lncrnas.txt
"""

import sys
import os
import gc
import argparse
import csv
from datetime import datetime
from dataclasses import dataclass
from multiprocessing import Pool
from pympler import asizeof
from typing import List, Dict, Tuple

from rules       import transfer_string, reverse_seq, complement
from stats       import calc_score
from sim         import SIM, Triplex, cluster_triplex, print_cluster
from data_loader import (load_mutations, load_chrom_map,
                         load_sequences, load_lncrnas)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass  (alignment parameters)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Para:
    rule:           int   = 0
    cut_length:     int   = 5000
    strand:         int   = 0
    overlap_length: int   = 100
    min_score:      int   = 0
    detail_output:  bool  = False
    nt_min:         int   = 20
    nt_max:         int   = 100000
    score_min:      float = 0.0
    min_identity:   float = 60.0
    min_stability:  float = 1.0
    penalty_t:      int   = -1000
    penalty_c:      int   = 0
    c_distance:     int   = 15
    c_length:       int   = 50


# ─────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────────────────────────

def cut_sequence(seq: str, cut_length: int,
                 overlap_length: int) -> Tuple[List[str], List[int], int]:
    seqs_vec:       List[str] = []
    seqs_start_pos: List[int] = []
    pos = 0
    while pos < len(seq):
        seqs_vec.append(seq[pos: pos + cut_length])
        seqs_start_pos.append(pos)
        pos += cut_length - overlap_length
    return seqs_vec, seqs_start_pos, len(seqs_vec)


def same_seq(seq: str) -> bool:
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
# lncRNA selection
# ─────────────────────────────────────────────────────────────────────────────

def resolve_lncrna_ids(
    lncrna_seqs_dict: Dict[str, Dict[int, str]],
    ids:              List[str] | None,
    ids_file:         str | None,
) -> List[str]:
    """
    Returns a flat list of "base_id:version" strings.

    Resolution rules:
      - "BASE_ID:N"  → use version N if it exists, else fall back to lowest.
      - "BASE_ID"    → use the lowest available version for that base_id.
      - base_id not in dict → warn and skip.
    """
    def lowest_version(base_id: str) -> int:
        return min(lncrna_seqs_dict[base_id].keys())

    def resolve_one(requested: str) -> str | None:
        if ":" in requested:
            base_id, ver_str = requested.rsplit(":", 1)
            try:
                version = int(ver_str)
            except ValueError:
                print(f"[WARNING] Could not parse version in '{requested}', skipping.",
                      file=sys.stderr)
                return None
            if base_id not in lncrna_seqs_dict:
                return None
            if version not in lncrna_seqs_dict[base_id]:
                low = lowest_version(base_id)
                print(f"[WARNING] '{requested}' version {version} not found; "
                      f"defaulting to lowest version {low}.", file=sys.stderr)
                return f"{base_id}:{low}"
            return f"{base_id}:{version}"
        else:
            if requested not in lncrna_seqs_dict:
                return None
            low = lowest_version(requested)
            print(f"[INFO] No version specified for '{requested}'; "
                  f"using lowest version {low}.")
            return f"{requested}:{low}"

    if not ids and not ids_file:
        print("[ERROR] One of --lncrna_ids or --lncrna_ids_file must be supplied.",
              file=sys.stderr)
        sys.exit(1)

    if ids:
        requested_list = ids
        source         = "--lncrna_ids"
    else:
        try:
            with open(ids_file) as fh:
                requested_list = [ln.strip() for ln in fh if ln.strip()]
        except FileNotFoundError:
            print(f"[ERROR] --lncrna_ids_file not found: {ids_file}", file=sys.stderr)
            sys.exit(1)
        source = f"--lncrna_ids_file ({ids_file})"

    valid   = []
    missing = []
    for r in requested_list:
        resolved = resolve_one(r)
        if resolved:
            valid.append(resolved)
        else:
            missing.append(r)

    if missing:
        print(f"[WARNING] {len(missing)} ID(s) from {source} not found: "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}", file=sys.stderr)
    if not valid:
        print("[ERROR] No valid lncRNA IDs remain after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] {len(valid)} lncRNA(s) selected via {source}.")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# LongTarget core
# ─────────────────────────────────────────────────────────────────────────────

def long_target(para: Para, rna_sequence: str,
                dna_sequence: str) -> List[Triplex]:
    chunks, start_positions, _ = cut_sequence(
        dna_sequence, para.cut_length, para.overlap_length)

    triplex_list: List[Triplex] = []

    for i, dna_chunk in enumerate(chunks):
        dna_start = start_positions[i]
        # print(f"  DNA chunk @ {dna_start} ({len(dna_chunk)} bp) ...")

        if same_seq(dna_chunk):
            # print("  Skipped (homopolymer)")
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
# Multiprocessing worker  (module-level for pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pair(args_tuple) -> Tuple[str, str, Dict[str, List[Triplex]]]:
    """
    Worker for one (lnc_id, gene) pair.
    lnc_id is a "base_id:version" string.
    Returns (lnc_id, gene, condition_dict).
    """
    para, lnc_id, lnc_seq, gene, healthy_seq, gene_muts, mut_seqs = args_tuple

    condition_dict: Dict[str, List[Triplex]] = {}

    print(f"  [{lnc_id}] {gene} healthy ...", flush=True)
    condition_dict["healthy"] = long_target(para, lnc_seq, healthy_seq)

    for mut_idx, (mut, mutated_seq) in enumerate(zip(gene_muts, mut_seqs)):
        if mutated_seq is None:
            continue
        mut_label = f"mut{mut_idx}_{mut.get('variant_class', 'unknown')}"
        print(f"  [{lnc_id}] {gene} {mut_label} ...", flush=True)
        condition_dict[mut_label] = long_target(para, lnc_seq, mutated_seq)

    return lnc_id, gene, condition_dict


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def report(
    results:        Dict[str, Dict[str, Dict[str, List[Triplex]]]],
    mutations_dict: Dict,
    outpath:        str,
    report_name:    str = None,
) -> None:
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
            gene_info  = mutations_dict.get(gene, {})
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
    parser = argparse.ArgumentParser(description="LongTarget lncRNA triplex pipeline")

    # ── data loading ──────────────────────────────────────────────────────
    parser.add_argument("--mutations",  nargs="+", required=True, metavar="MAF_FILE")
    parser.add_argument("--genome",     required=True, metavar="FASTA")
    parser.add_argument("--annotation", required=True, metavar="GTF")
    parser.add_argument("--assembly",   required=True, metavar="ASSEMBLY_REPORT")
    parser.add_argument("--lncrna",     required=True, metavar="FASTA")

    # ── lncRNA selection ──────────────────────────────────────────────────
    lnc_group = parser.add_mutually_exclusive_group()
    lnc_group.add_argument("--lncrna_ids",      nargs="+", metavar="ID",  default=None)
    lnc_group.add_argument("--lncrna_ids_file", metavar="FILE",           default=None)

    # ── runtime ───────────────────────────────────────────────────────────
    parser.add_argument("--n_cores", type=int, default=1, metavar="N",
                        help="Number of worker processes (default: 1).")

    # ── reporting ─────────────────────────────────────────────────────────
    parser.add_argument("--outpath",    default="./results")
    parser.add_argument("--report_name", type=str, default=None)

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

    os.makedirs(args.outpath, exist_ok=True)
    print(f"Output directory: {args.outpath}")

    # ── load data ─────────────────────────────────────────────────────────
    print("Loading mutated gene data ...")
    mutations_dict, gene_names = load_mutations(args.mutations)

    print("Building chromosome map ...")
    chrom_map = load_chrom_map(args.assembly)

    print("Loading gene sequences ...")
    healthy_seqs_dict, mutated_seqs_dict = load_sequences(
        fasta_path     = args.genome,
        gtf_path       = args.annotation,
        gene_names     = gene_names,
        chrom_map      = chrom_map,
        mutations_dict = mutations_dict,
    )
    del chrom_map, gene_names
    gc.collect()

    print("Loading lncRNA sequences ...")
    lncrna_seqs_dict = load_lncrnas(args.lncrna)

    # Restrict sequence dicts to only genes present in mutations_dict
    shared_gene_set   = set(mutations_dict.keys()) & set(healthy_seqs_dict.keys())
    healthy_seqs_dict = {g: healthy_seqs_dict[g] for g in shared_gene_set}
    mutated_seqs_dict = {g: mutated_seqs_dict[g] for g in shared_gene_set
                         if g in mutated_seqs_dict}
    gc.collect()

    # Resolve lncRNA IDs → flat list of "base_id:version" strings
    test_lncrnas = resolve_lncrna_ids(
        lncrna_seqs_dict,
        ids      = args.lncrna_ids,
        ids_file = args.lncrna_ids_file,
    )

    # Keep only the lncRNAs we will actually use: {lnc_id: sequence}
    filtered_lncrna_seqs_dict = {}
    for lnc_id in test_lncrnas:
        base_id, ver_str = lnc_id.rsplit(":", 1)
        filtered_lncrna_seqs_dict[lnc_id] = lncrna_seqs_dict[base_id][int(ver_str)]
    del lncrna_seqs_dict
    gc.collect()

    print(f"\n=== Data Summary ===")
    print(f"  Genes with mutations:     {len(mutations_dict)}")
    print(f"  Gene sequences retrieved: {len(healthy_seqs_dict)}")
    print(f"  lncRNA transcripts:       {len(filtered_lncrna_seqs_dict)}")

    shared_genes = sorted(shared_gene_set)

    print(f"\n  Genes available for analysis: {len(shared_genes)}")
    print(f"  lncRNAs to test:              {len(test_lncrnas)}")
    print(f"  Workers:                      {args.n_cores}")
    print(f"\n=== Memory Summary ===")
    print(f"  healthy_seqs_dict: {asizeof.asizeof(healthy_seqs_dict) / 1e6:.2f} MB")
    print(f"  mutated_seqs_dict: {asizeof.asizeof(mutated_seqs_dict) / 1e6:.2f} MB")

    # ── build task list ───────────────────────────────────────────────────
    tasks = [
        (
            para,
            lnc_id,
            filtered_lncrna_seqs_dict[lnc_id],
            gene,
            healthy_seqs_dict[gene],
            mutations_dict[gene]["mutations"],
            mutated_seqs_dict.get(gene, []),
        )
        for lnc_id in test_lncrnas
        for gene   in shared_genes
    ]

    # ── run ───────────────────────────────────────────────────────────────
    loop_start = datetime.now()
    print(f"\nLoop started at: {loop_start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    with Pool(processes=args.n_cores) as pool:
        task_results = pool.map(_run_pair, tasks)

    # Re-nest results into {lnc_id: {gene: {condition: [Triplex]}}}
    results:     Dict[str, Dict[str, Dict[str, List[Triplex]]]] = {}
    total_pairs = 0
    total_hits  = 0

    for lnc_id, gene, condition_dict in task_results:
        results.setdefault(lnc_id, {})[gene] = condition_dict
        total_hits  += sum(len(v) for v in condition_dict.values())
        total_pairs += 1

    loop_end = datetime.now()
    elapsed  = loop_end - loop_start

    print(f"\nLoop ended at:   {loop_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time:    {elapsed}")
    print(f"Total (lncRNA × gene) pairs processed: {total_pairs}")
    print(f"Total triplex hits across all runs:     {total_hits}")

    # ── write report ──────────────────────────────────────────────────────
    report(results, mutations_dict, args.outpath, args.report_name)

    print("\nFinished normally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())