"""
main.py – LongTarget: lncRNA–DNA triplex alignment pipeline

Wires together:
  rules.py  → transfer_string, reverse_seq, complement
  stats.py  → calc_score
  sim.py    → SIM, Triplex, cluster_triplex, print_cluster

Usage:
  python main.py -f1 DNAseq.fa -f2 RNAseq.fa -r 0 -O ./results

Input file formats
------------------
DNA FASTA header:  >species|chrTag|start-end
                   e.g.  >hg38|chr1|100000-200000
RNA FASTA header:  >lncRNA_name
                   e.g.  >HOTAIR
"""

import sys
import os
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ── import from the three supporting modules ──────────────────────────────────
from rules import transfer_string, reverse_seq, complement
from stats import calc_score
from sim   import SIM, Triplex, cluster_triplex, print_cluster


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Para:
    file1path:      str   = "./"       # DNA FASTA file
    file2path:      str   = "./"       # RNA FASTA file
    outpath:        str   = "./"       # output directory
    rule:           int   = 0          # 0 = all rules
    cut_length:     int   = 5000       # chunk size for DNA cutting
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


@dataclass
class LgInfo:
    lnc_name:    str = ""
    lnc_seq:     str = ""
    species:     str = ""
    dna_chro_tag:str = ""
    file_name:   str = ""
    dna_seq:     str = ""
    start_genome:int = 0
    result_dir:  str = ""


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_rna(rna_file_name: str) -> Tuple[str, str]:
    """
    Read a single-sequence FASTA file for the lncRNA.
    Returns (lnc_name, rna_sequence).
    """
    with open(rna_file_name, "r") as f:
        first_line = f.readline()
        lnc_name   = first_line.lstrip(">").strip()
        print(f"RNA name: {lnc_name}")
        rna_seq = "".join(line.strip() for line in f)
    return lnc_name, rna_seq


def read_dna(dna_file_name: str) -> Tuple[str, str, str, str]:
    """
    Read a single-sequence FASTA file for the DNA target region.
    Expected header format: >species|chrTag|start-end
    Returns (dna_sequence, species, chro_tag, start_genome_str).
    """
    with open(dna_file_name, "r") as f:
        first_line = f.readline().lstrip(">").strip()
        parts      = first_line.split("|")
        species      = parts[0] if len(parts) > 0 else ""
        chro_tag     = parts[1] if len(parts) > 1 else ""
        start_genome = parts[2].split("-")[0] if len(parts) > 2 else "0"
        print(f"Species: {species}  Chr: {chro_tag}  Start: {start_genome}")
        dna_seq = "".join(line.strip() for line in f)
    return dna_seq, species, chro_tag, start_genome


# ─────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────────────────────────

def cut_sequence(seq: str, cut_length: int,
                 overlap_length: int) -> Tuple[List[str], List[int], int]:
    """
    Split a long DNA sequence into overlapping chunks.
    Returns (chunks, start_positions, number_of_chunks).
    """
    seqs_vec:       List[str] = []
    seqs_start_pos: List[int] = []
    pos = 0
    while pos < len(seq):
        seqs_vec.append(seq[pos: pos + cut_length])
        seqs_start_pos.append(pos)
        pos += cut_length - overlap_length
    return seqs_vec, seqs_start_pos, len(seqs_vec)


def same_seq(seq: str) -> bool:
    """
    Return True if the sequence is composed entirely of one nucleotide type.
    Such sequences are uninformative and are skipped.
    """
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
    """Convert (reverse, strand) integers to a human-readable label."""
    if   reverse == 0 and strand ==  1: return "ParaPlus"
    elif reverse == 1 and strand ==  1: return "ParaMinus"
    elif reverse == 1 and strand == -1: return "AntiMinus"
    elif reverse == 0 and strand == -1: return "AntiPlus"
    return ""


def comp_key(t: Triplex) -> int:
    """Sort key: ascending motif class number."""
    return t.motif


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def show_help() -> None:
    print("""
LongTarget – lncRNA triplex target site predictor
--------------------------------------------------
Required:
  -f1  <path>   DNA sequence FASTA file
  -f2  <path>   RNA (lncRNA) sequence FASTA file
  -r   <int>    Triplex rule (0 = all rules)

Optional:
  -O   <path>   Output directory            (default: ./)
  -c   <int>    DNA chunk size              (default: 5000)
  -o   <int>    Chunk overlap length        (default: 100)
  -t   <int>    Strand  0=both 1=para -1=anti (default: 0)
  -m   <int>    Minimum raw score          (default: 0)
  -i   <float>  Minimum % identity         (default: 60.0)
  -S   <float>  Minimum mean stability     (default: 1.0)
  -ni  <int>    Minimum triplex length     (default: 20)
  -na  <int>    Maximum triplex length     (default: 100000)
  -pt  <int>    Penalty for AA runs        (default: -1000)
  -pc  <int>    Penalty for GG runs        (default: 0)
  -ds  <int>    Clustering distance        (default: 15)
  -lg  <int>    Clustering min length      (default: 50)
  -d            Enable detailed output
  -h            Show this help

Example:
  python main.py -f1 DNAseq.fa -f2 RNAseq.fa -r 0 -O ./results
""")
    sys.exit(1)


def init_env(argv: List[str]) -> Para:
    """Parse command-line arguments into a Para config object."""
    para = Para()

    if len(argv) <= 1:
        show_help()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-f1", "--f1",   dest="file1path")
    parser.add_argument("-f2", "--f2",   dest="file2path")
    parser.add_argument("-r",            dest="rule",           type=int)
    parser.add_argument("-O",            dest="outpath")
    parser.add_argument("-c",            dest="cut_length",     type=int)
    parser.add_argument("-m",            dest="min_score",      type=int)
    parser.add_argument("-t",            dest="strand",         type=int)
    parser.add_argument("-d",            dest="detail_output",  action="store_true")
    parser.add_argument("-i",            dest="min_identity",   type=float)
    parser.add_argument("-S",            dest="min_stability",  type=float)
    parser.add_argument("-ni", "--ni",   dest="nt_min",         type=int)
    parser.add_argument("-na", "--na",   dest="nt_max",         type=int)
    parser.add_argument("-pc", "--pc",   dest="penalty_c",      type=int)
    parser.add_argument("-pt", "--pt",   dest="penalty_t",      type=int)
    parser.add_argument("-o",            dest="overlap_length", type=int)
    parser.add_argument("-h",            dest="show_help",      action="store_true")
    parser.add_argument("-ds", "--ds",   dest="c_distance",     type=int)
    parser.add_argument("-lg", "--lg",   dest="c_length",       type=int)

    args, _ = parser.parse_known_args(argv[1:])

    if args.show_help:
        show_help()

    if args.file1path      is not None: para.file1path      = args.file1path
    if args.file2path      is not None: para.file2path      = args.file2path
    if args.rule           is not None: para.rule           = args.rule
    if args.outpath        is not None: para.outpath        = args.outpath
    if args.cut_length     is not None: para.cut_length     = args.cut_length
    if args.min_score      is not None: para.min_score      = args.min_score
    if args.strand         is not None: para.strand         = args.strand
    if args.detail_output:              para.detail_output  = True
    if args.min_identity   is not None: para.min_identity   = args.min_identity
    if args.min_stability  is not None: para.min_stability  = args.min_stability
    if args.nt_min         is not None: para.nt_min         = args.nt_min
    if args.nt_max         is not None: para.nt_max         = args.nt_max
    if args.penalty_c      is not None: para.penalty_c      = args.penalty_c
    if args.penalty_t      is not None: para.penalty_t      = args.penalty_t
    if args.overlap_length is not None: para.overlap_length = args.overlap_length
    if args.c_distance     is not None: para.c_distance     = args.c_distance
    if args.c_length       is not None: para.c_length       = args.c_length

    return para


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
    t0 = time.time()

    for i, dna_chunk in enumerate(chunks):
        dna_start = start_positions[i]
        print(f"Processing DNA chunk at position {dna_start} "
              f"({len(dna_chunk)} bp) ...")

        if same_seq(dna_chunk):
            print("  Skipped (homopolymer / single-base sequence)")
            continue

        # ── parallel rules (Para = +1) ────────────────────────────────────
        if para.strand >= 0:
            rules_para = range(1, 7)  # rules 1–6
            for j in (rules_para if para.rule == 0 else [para.rule]):
                if para.rule != 0 and not (0 < para.rule < 7):
                    break

                # forward strand
                seq2 = transfer_string(dna_chunk, 0, 1, j)
                min_sc = calc_score(rna_sequence, seq2, dna_start, j)
                SIM(rna_sequence, seq2, dna_chunk, dna_start, min_sc,
                    5, -4, -12, -4, triplex_list, 0, 1, j,
                    para.nt_min, para.nt_max, para.penalty_t, para.penalty_c)

                # reverse strand
                seq2 = reverse_seq(transfer_string(dna_chunk, 1, 1, j))
                min_sc = calc_score(rna_sequence, seq2, dna_start, j)
                SIM(rna_sequence, seq2, dna_chunk, dna_start, min_sc,
                    5, -4, -12, -4, triplex_list, 1, 1, j,
                    para.nt_min, para.nt_max, para.penalty_t, para.penalty_c)

        # ── anti-parallel rules (Para = -1) ───────────────────────────────
        if para.strand <= 0:
            rules_anti = range(1, 19)  # rules 1–18
            for j in (rules_anti if para.rule == 0 else [para.rule]):

                # forward strand
                seq2 = transfer_string(dna_chunk, 0, -1, j)
                min_sc = calc_score(rna_sequence, seq2, dna_start, j)
                SIM(rna_sequence, seq2, dna_chunk, dna_start, min_sc,
                    5, -4, -12, -4, triplex_list, 0, -1, j,
                    para.nt_min, para.nt_max, para.penalty_t, para.penalty_c)

                # reverse strand
                seq2 = reverse_seq(transfer_string(dna_chunk, 1, -1, j))
                min_sc = calc_score(rna_sequence, seq2, dna_start, j)
                SIM(rna_sequence, seq2, dna_chunk, dna_start, min_sc,
                    5, -4, -12, -4, triplex_list, 1, -1, j,
                    para.nt_min, para.nt_max, para.penalty_t, para.penalty_c)

    elapsed = time.time() - t0
    print(f"Alignment done in {elapsed:.1f}s — "
          f"{len(triplex_list)} raw candidates found")

    # ── filter by quality thresholds ──────────────────────────────────────
    filtered = [
        t for t in triplex_list
        if (t.score     >= para.score_min    and
            t.identity  >= para.min_identity and
            t.tri_score >= para.min_stability)
    ]
    print(f"{len(filtered)} candidates pass quality filters")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Result output
# ─────────────────────────────────────────────────────────────────────────────

def print_result(species: str, para: Para, lnc_name: str, dna_file: str,
                 sort_triplex_list: List[Triplex], chro_tag: str,
                 dna_sequence: str, start_genome: int,
                 c_tmp_dd: str, c_tmp_length: str,
                 result_dir: str) -> None:
    """
    Write the main results TSV and UCSC bedGraph cluster tracks.
    """
    os.makedirs(result_dir, exist_ok=True)

    out_file_path = os.path.join(
        result_dir, f"{species}-{lnc_name}-{dna_file}-TFOsorted")

    header = (
        "QueryStart\tQueryEnd\tStartInSeq\tEndInSeq\tDirection\t"
        "StartInGenome\tEndInGenome\tMeanStability\tMeanIdentity(%)\t"
        "Strand\tRule\tScore\tNt(bp)\tClass\tMidPoint\tCenter\tTFO sequence"
    )

    # cluster coverage maps (6 levels, indices 0–5; index 0 unused)
    class1  = [{} for _ in range(6)]
    class1a = [{} for _ in range(6)]
    class1b = [{} for _ in range(6)]
    class_level = 5

    cluster_triplex(para.c_distance, para.c_length,
                    sort_triplex_list, class1, class1a, class1b, class_level)

    sort_triplex_list.sort(key=comp_key)

    with open(out_file_path, "w") as out_file:
        out_file.write(header + "\n")
        for atr in sort_triplex_list:
            if atr.motif == 0:
                continue

            tfo_seq = atr.stri_align.replace("-", "")

            if atr.starj < atr.endj:
                direction = "R"
                gen_start = atr.starj + start_genome - 1
                gen_end   = atr.endj  + start_genome - 1
            else:
                direction = "L"
                gen_start = atr.endj  + start_genome - 1
                gen_end   = atr.starj + start_genome - 1

            out_file.write(
                f"{atr.stari}\t{atr.endi}\t{atr.starj}\t{atr.endj}\t"
                f"{direction}\t{gen_start}\t{gen_end}\t"
                f"{atr.tri_score:.4f}\t{atr.identity:.2f}\t"
                f"{get_strand(atr.reverse, atr.strand)}\t"
                f"{atr.rule}\t{atr.score:.4f}\t{atr.nt}\t{atr.motif}\t"
                f"{atr.middle}\t{atr.center}\t{tfo_seq}\n"
            )

    print(f"Results written to: {out_file_path}")

    # write UCSC bedGraph cluster tracks for levels 1 and 2
    w_tmp_class: list = []
    for pr_loop in range(1, 3):
        print_cluster(pr_loop, class1, start_genome - 1, chro_tag,
                      len(dna_sequence), lnc_name,
                      para.c_distance, para.c_length,
                      out_file_path, c_tmp_dd, c_tmp_length, w_tmp_class)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: List[str] = None) -> int:
    if argv is None:
        argv = sys.argv

    para = init_env(argv)

    c_tmp_dd     = str(para.c_distance)
    c_tmp_length = str(para.c_length)

    # read inputs
    dna_seq, species, dna_chro_tag, start_genome_str = read_dna(para.file1path)
    start_genome = int(start_genome_str) if start_genome_str.isdigit() else 0

    lnc_name, lnc_seq = read_rna(para.file2path)

    # strip .fa extension for output naming
    dna_file = (para.file1path[:-3]
                if para.file1path.endswith(".fa")
                else para.file1path)
    dna_file = os.path.basename(dna_file)

    lg_info = LgInfo(
        lnc_name=lnc_name,     lnc_seq=lnc_seq,
        species=species,       dna_chro_tag=dna_chro_tag,
        file_name=dna_file,    dna_seq=dna_seq,
        start_genome=start_genome,
        result_dir=para.outpath,
    )

    # run alignment
    results = long_target(para, lg_info.lnc_seq, lg_info.dna_seq)

    # write output
    print_result(
        lg_info.species, para, lg_info.lnc_name, lg_info.file_name,
        results, lg_info.dna_chro_tag, lg_info.dna_seq,
        lg_info.start_genome, c_tmp_dd, c_tmp_length, lg_info.result_dir,
    )

    print("Finished normally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())