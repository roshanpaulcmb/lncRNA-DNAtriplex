#!/usr/bin/env python3
"""
run_longtarget.py
=================
Run the Python LongTarget pipeline against every entry in a multi-FASTA
DNA file (e.g. oncogene_2kb_upstream.fa) using a single lncRNA query.

All four modules (rules.py, stats.py, sim.py, and this script) must live
in the same directory.

Usage
-----
    # Minimal — all defaults:
    python run_longtarget.py \\
        --dna  oncogene_2kb_upstream.fa \\
        --rna  my_lncRNA.fa

    # With options:
    python run_longtarget.py \\
        --dna  oncogene_2kb_upstream.fa \\
        --rna  my_lncRNA.fa             \\
        --out  results/                 \\
        --jobs 8                        \\
        --rule 0                        \\
        --nt-min 20                     \\
        --identity 60                   \\
        --stability 1.0

Arguments
---------
  --dna         Multi-FASTA DNA file  (required)
  --rna         Single-entry lncRNA FASTA  (required)
  --out         Output directory  (default: longtarget_results/)
  --jobs        Parallel worker processes  (default: 1)
  --rule        Triplex rule: 0=all, 1-6=parallel, 1-18=antiparallel  (default: 0)
  --strand      0=both, 1=parallel only, -1=antiparallel only  (default: 0)
  --nt-min      Minimum triplex length in bp  (default: 20)
  --nt-max      Maximum triplex length in bp  (default: 100000)
  --identity    Minimum identity %  (default: 60.0)
  --stability   Minimum mean stability score  (default: 1.0)
  --penalty-t   Penalty for consecutive A pairs  (default: -1000)
  --penalty-c   Penalty for consecutive G pairs  (default: 0)
  --cut-length  DNA chunk size for long sequences  (default: 5000)
  --overlap     Overlap between consecutive chunks  (default: 100)
  --c-distance  Cluster distance parameter  (default: 15)
  --c-length    Cluster length threshold  (default: 50)

Output files (all written to --out directory)
---------------------------------------------
  merged_TFOsorted.tsv          All triplex hits across every gene (TSV)
  summary.tsv                   One row per gene: hit count + best scores
  per_gene/<GENE>-TFOsorted     Raw per-gene output (same format as C++ LongTarget)
  per_gene/<GENE>-TFOclass*     bedGraph cluster files per gene

Dependencies
------------
    pip install parasail
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# ── Import sibling modules ────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from rules import transfer_string, reverse_seq, complement
from stats import calc_score
from sim   import (
    Triplex, TmpClass,
    SIM, cluster_triplex, print_cluster,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
    level   = logging.INFO,
    stream  = sys.stdout,
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parameters dataclass  (mirrors C++ struct para)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Params:
    rule:         int   = 0
    strand:       int   = 0
    nt_min:       int   = 20
    nt_max:       int   = 100_000
    score_min:    float = 0.0
    min_identity: float = 60.0
    min_stability:float = 1.0
    penalty_t:    int   = -1000
    penalty_c:    int   = 0
    cut_length:   int   = 5_000
    overlap:      int   = 100
    c_distance:   int   = 15
    c_length:     int   = 50


# ─────────────────────────────────────────────────────────────────────────────
# FASTA helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_fasta(path: str) -> List[Tuple[str, str]]:
    """
    Parse a FASTA file.  Returns a list of (header, sequence) tuples.
    The header is everything after '>' on the definition line.
    """
    entries = []
    header  = None
    seqbuf: List[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seqbuf).upper()))
                header = line[1:].strip()
                seqbuf = []
            elif line:
                seqbuf.append(line)

    if header is not None:
        entries.append((header, "".join(seqbuf).upper()))

    return entries


def gene_symbol(header: str) -> str:
    """Extract the first whitespace-delimited token from a FASTA header."""
    return header.split()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Sequence utilities  (mirrors longtarget.cpp helpers)
# ─────────────────────────────────────────────────────────────────────────────

def cut_sequence(seq: str, cut_length: int, overlap: int) -> Tuple[List[str], List[int]]:
    """Split seq into overlapping chunks. Returns (chunks, start_positions)."""
    chunks: List[str] = []
    starts: List[int] = []
    pos = 0
    while pos < len(seq):
        chunks.append(seq[pos: pos + cut_length])
        starts.append(pos)
        pos += cut_length - overlap
    return chunks, starts


def same_seq(seq: str) -> bool:
    """Return True if seq consists entirely of one nucleotide (uninformative)."""
    if not seq:
        return True
    return len(set(seq)) == 1


def get_strand_label(reverse: int, strand: int) -> str:
    if   reverse == 0 and strand ==  1: return "ParaPlus"
    elif reverse == 1 and strand ==  1: return "ParaMinus"
    elif reverse == 1 and strand == -1: return "AntiMinus"
    elif reverse == 0 and strand == -1: return "AntiPlus"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Core: LongTarget for one (lncRNA, DNA) pair  (mirrors C++ LongTarget())
# ─────────────────────────────────────────────────────────────────────────────

def run_longtarget_one(rna_seq: str, dna_seq: str,
                       params: Params) -> List[Triplex]:
    """
    Run LongTarget on a single lncRNA / DNA pair.
    Returns the list of triplex hits that pass the score/identity/stability
    thresholds.
    """
    chunks, starts = cut_sequence(dna_seq, params.cut_length, params.overlap)
    triplex_list: List[Triplex] = []

    for seq1, dna_start_pos in zip(chunks, starts):
        if same_seq(seq1):
            continue

        # ── Parallel (positive-strand) rules ─────────────────────────────────
        if params.strand >= 0:
            rules = range(1, 7) if params.rule == 0 else \
                    (range(params.rule, params.rule + 1) if 0 < params.rule < 7 else [])
            for rule_num in rules:
                for rev in (0, 1):
                    seq2 = transfer_string(seq1, rev, 1, rule_num)
                    if rev == 1:
                        seq2 = reverse_seq(seq2)
                    min_score = calc_score(rna_seq, seq2, dna_start_pos, params.rule)
                    SIM(rna_seq, seq2, seq1, dna_start_pos, min_score,
                        5, -4, -12, -4,
                        triplex_list, rev, 1, rule_num,
                        params.nt_min, params.nt_max,
                        params.penalty_t, params.penalty_c)

        # ── Antiparallel (negative-strand) rules ──────────────────────────────
        if params.strand <= 0:
            if params.rule == 0:
                rules = range(1, 19)
            elif params.rule > 0:
                rules = range(params.rule, params.rule + 1)
            else:
                rules = []
            for rule_num in rules:
                for rev in (0, 1):
                    seq2 = transfer_string(seq1, rev, -1, rule_num)
                    if rev == 1:
                        seq2 = reverse_seq(seq2)
                    min_score = calc_score(rna_seq, seq2, dna_start_pos, params.rule)
                    SIM(rna_seq, seq2, seq1, dna_start_pos, min_score,
                        5, -4, -12, -4,
                        triplex_list, rev, -1, rule_num,
                        params.nt_min, params.nt_max,
                        params.penalty_t, params.penalty_c)

    # Filter by quality thresholds
    return [
        t for t in triplex_list
        if (t.score    >= params.score_min    and
            t.identity >= params.min_identity and
            t.tri_score >= params.min_stability)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Output: write per-gene TFOsorted file  (mirrors C++ printResult())
# ─────────────────────────────────────────────────────────────────────────────

HEADER = (
    "QueryStart\tQueryEnd\tStartInSeq\tEndInSeq\tDirection\t"
    "StartInGenome\tEndInGenome\tMeanStability\tMeanIdentity(%)\t"
    "Strand\tRule\tScore\tNt(bp)\tClass\tMidPoint\tCenter\tTFO_sequence"
)


def write_gene_result(gene: str, hits: List[Triplex],
                      dna_seq: str, start_genome: int,
                      chro_tag: str, rna_name: str,
                      params: Params, out_dir: Path) -> str:
    """
    Write the TFOsorted file for one gene and the companion bedGraph files.
    Returns the path to the TFOsorted file.
    """
    out_path = str(out_dir / f"{gene}-TFOsorted")
    c_dd     = str(params.c_distance)
    c_len    = str(params.c_length)

    class1  = [{} for _ in range(6)]
    class1a = [{} for _ in range(6)]
    class1b = [{} for _ in range(6)]

    cluster_triplex(params.c_distance, params.c_length,
                    hits, class1, class1a, class1b, class_level=5)

    hits_sorted = sorted(hits, key=lambda t: t.motif)

    with open(out_path, "w") as fh:
        fh.write(HEADER + "\n")
        for t in hits_sorted:
            if t.motif == 0:
                continue
            tfo = t.stri_align.replace("-", "")
            if t.starj < t.endj:
                direction  = "R"
                gen_start  = t.starj + start_genome - 1
                gen_end    = t.endj  + start_genome - 1
            else:
                direction  = "L"
                gen_start  = t.endj  + start_genome - 1
                gen_end    = t.starj + start_genome - 1
            fh.write(
                f"{t.stari}\t{t.endi}\t{t.starj}\t{t.endj}\t{direction}\t"
                f"{gen_start}\t{gen_end}\t{t.tri_score:.4f}\t{t.identity:.2f}\t"
                f"{get_strand_label(t.reverse, t.strand)}\t{t.rule}\t"
                f"{t.score:.4f}\t{t.nt}\t{t.motif}\t{t.middle}\t{t.center}\t"
                f"{tfo}\n"
            )

    w_tmp: List[TmpClass] = []
    for level in range(1, 3):
        print_cluster(level, class1, start_genome - 1,
                      chro_tag, len(dna_seq), rna_name,
                      params.c_distance, params.c_length,
                      out_path, c_dd, c_len, w_tmp)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Worker function  (called by multiprocessing.Pool)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> dict:
    """
    Process one gene entry.  Returns a result dict for the summary.
    This function runs in a subprocess so imports are re-executed there.
    """
    (gene, dna_seq, dna_header,
     rna_name, rna_seq,
     params, per_gene_dir) = args

    t0 = time.perf_counter()
    log.info("  [worker] %s  (%d bp DNA)", gene, len(dna_seq))

    # Parse chrom and start_genome from the FASTA header if present
    # Header format: GENE  genome=hg38  chrN:START-END(STRAND)  upstream=2000bp_of_TSS
    import re
    chro_tag    = gene
    start_genome = 1
    m = re.search(r'(chr\S+):(\d+)-(\d+)', dna_header)
    if m:
        chro_tag     = m.group(1)
        start_genome = int(m.group(2)) + 1   # convert 0-based to 1-based

    try:
        hits = run_longtarget_one(rna_seq, dna_seq, params)
    except Exception as exc:
        log.error("  [worker] %s failed: %s", gene, exc)
        return {"gene": gene, "status": "error", "error": str(exc),
                "n_hits": 0, "tfo_path": None}

    tfo_path = write_gene_result(
        gene, hits, dna_seq, start_genome,
        chro_tag, rna_name, params, per_gene_dir
    )

    elapsed = time.perf_counter() - t0
    log.info("  [worker] %s  → %d hits  (%.1f s)", gene, len(hits), elapsed)

    best = max(hits, key=lambda t: t.score) if hits else None
    return {
        "gene":       gene,
        "status":     "ok",
        "n_hits":     len(hits),
        "tfo_path":   tfo_path,
        "best_score":     round(best.score,     4) if best else None,
        "best_identity":  round(best.identity,  2) if best else None,
        "best_stability": round(best.tri_score, 4) if best else None,
        "best_nt":        best.nt                  if best else None,
        "best_tfo":       best.stri_align.replace("-","") if best else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Merge per-gene TFOsorted files into one TSV
# ─────────────────────────────────────────────────────────────────────────────

def merge_results(results: List[dict], out_dir: Path) -> Tuple[Path, Path]:
    merged_path  = out_dir / "merged_TFOsorted.tsv"
    summary_path = out_dir / "summary.tsv"

    with open(merged_path, "w") as mf:
        mf.write("Gene\t" + HEADER + "\n")
        for r in results:
            if not r.get("tfo_path") or not Path(r["tfo_path"]).exists():
                continue
            gene = r["gene"]
            with open(r["tfo_path"]) as tf:
                next(tf)   # skip header
                for line in tf:
                    line = line.rstrip()
                    if line:
                        mf.write(f"{gene}\t{line}\n")

    with open(summary_path, "w", newline="") as sf:
        w = csv.writer(sf, delimiter="\t")
        w.writerow(["Gene", "Hits", "BestScore", "BestIdentity(%)",
                    "BestStability", "BestNt(bp)", "BestTFO", "Status"])
        for r in sorted(results, key=lambda x: x.get("n_hits", 0), reverse=True):
            w.writerow([
                r["gene"],
                r.get("n_hits", 0),
                r.get("best_score",     "NA"),
                r.get("best_identity",  "NA"),
                r.get("best_stability", "NA"),
                r.get("best_nt",        "NA"),
                r.get("best_tfo",       "NA"),
                r.get("status", "error"),
            ])

    return merged_path, summary_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dna",         required=True,
                    help="Multi-FASTA DNA file (e.g. oncogene_2kb_upstream.fa)")
    ap.add_argument("--rna",         required=True,
                    help="Single-entry lncRNA FASTA")
    ap.add_argument("--out",         default="longtarget_results",
                    help="Output directory (default: longtarget_results/)")
    ap.add_argument("--jobs",        type=int, default=1,
                    help="Parallel worker processes (default: 1)")
    ap.add_argument("--rule",        type=int, default=0)
    ap.add_argument("--strand",      type=int, default=0,
                    choices=[0, 1, -1])
    ap.add_argument("--nt-min",      type=int,   default=20)
    ap.add_argument("--nt-max",      type=int,   default=100_000)
    ap.add_argument("--identity",    type=float, default=60.0)
    ap.add_argument("--stability",   type=float, default=1.0)
    ap.add_argument("--penalty-t",   type=int,   default=-1000)
    ap.add_argument("--penalty-c",   type=int,   default=0)
    ap.add_argument("--cut-length",  type=int,   default=5_000)
    ap.add_argument("--overlap",     type=int,   default=100)
    ap.add_argument("--c-distance",  type=int,   default=15)
    ap.add_argument("--c-length",    type=int,   default=50)
    args = ap.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    for path, label in [(args.dna, "--dna"), (args.rna, "--rna")]:
        if not Path(path).exists():
            ap.error(f"{label} file not found: {path}")

    # ── Read sequences ────────────────────────────────────────────────────────
    rna_entries = read_fasta(args.rna)
    if not rna_entries:
        ap.error(f"No sequences found in {args.rna}")
    rna_header, rna_seq = rna_entries[0]
    rna_name = gene_symbol(rna_header)

    dna_entries = read_fasta(args.dna)
    if not dna_entries:
        ap.error(f"No sequences found in {args.dna}")

    log.info("lncRNA  : %s  (%d bp)", rna_name, len(rna_seq))
    log.info("Targets : %d genes from %s", len(dna_entries), args.dna)

    # ── Create output dirs ────────────────────────────────────────────────────
    out_dir      = Path(args.out)
    per_gene_dir = out_dir / "per_gene"
    per_gene_dir.mkdir(parents=True, exist_ok=True)

    # ── Build params ──────────────────────────────────────────────────────────
    params = Params(
        rule          = args.rule,
        strand        = args.strand,
        nt_min        = args.nt_min,
        nt_max        = args.nt_max,
        min_identity  = args.identity,
        min_stability = args.stability,
        penalty_t     = args.penalty_t,
        penalty_c     = args.penalty_c,
        cut_length    = args.cut_length,
        overlap       = args.overlap,
        c_distance    = args.c_distance,
        c_length      = args.c_length,
    )

    log.info("Parameters: rule=%d  strand=%d  nt=%d-%d  identity>=%.0f%%  "
             "stability>=%.1f  jobs=%d",
             params.rule, params.strand, params.nt_min, params.nt_max,
             params.min_identity, params.min_stability, args.jobs)

    # ── Build job list ────────────────────────────────────────────────────────
    jobs = [
        (gene_symbol(hdr), seq, hdr,
         rna_name, rna_seq,
         params, per_gene_dir)
        for hdr, seq in dna_entries
    ]

    # ── Run ───────────────────────────────────────────────────────────────────
    t_start = time.perf_counter()

    if args.jobs == 1:
        log.info("Running single-process (use --jobs N to parallelise)")
        results = [_worker(j) for j in jobs]
    else:
        log.info("Spawning %d worker processes", args.jobs)
        with multiprocessing.Pool(processes=args.jobs) as pool:
            results = pool.map(_worker, jobs)

    elapsed = time.perf_counter() - t_start

    # ── Merge & summarise ─────────────────────────────────────────────────────
    merged_path, summary_path = merge_results(results, out_dir)

    total_hits = sum(r.get("n_hits", 0) for r in results)
    ok_count   = sum(1 for r in results if r.get("status") == "ok")
    err_count  = len(results) - ok_count

    print()
    print("=" * 62)
    print(f"  Done!  {ok_count}/{len(results)} genes processed")
    if err_count:
        print(f"  Errors: {err_count} genes (check log above)")
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    print(f"  Runtime: {h:02d}h {m:02d}m {s:02d}s")
    print(f"  Total triplex hits : {total_hits:,}")
    print(f"  Merged TSV  → {merged_path}")
    print(f"  Summary     → {summary_path}")
    print(f"  Per-gene    → {per_gene_dir}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
