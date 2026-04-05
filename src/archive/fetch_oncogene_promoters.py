#!/usr/bin/env python3
"""
fetch_oncogene_promoters.py  (v2 — fixed gene coordinate lookup)
================================================================
1. Downloads the 803 human oncogenes from ONGene
   (http://ongene.bioinfo-minzhao.org/).  Falls back to embedded list
   if the site is unreachable.

2. Resolves every gene symbol to hg38 coordinates using MyGene.info
   (batch POST, up to 1 000 genes per request — no API key needed).

3. Fetches the 2 kb immediately upstream of each TSS via the UCSC
   REST API (getData/sequence).

4. Writes:
     oncogene_2kb_upstream.fa            — multi-FASTA
     oncogene_2kb_upstream_summary.tsv   — one row per gene
     oncogene_failed_genes.txt           — genes that could not be resolved

Requirements
------------
    pip install requests

Run
---
    python fetch_oncogene_promoters.py

    # Optional flags
    python fetch_oncogene_promoters.py --upstream 2000 --genome hg38
    python fetch_oncogene_promoters.py --gene-list my_genes.txt
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import requests

# ─── Configuration ────────────────────────────────────────────────────────────
GENOME        = "hg38"
UPSTREAM_BP   = 2000
RATE_LIMIT    = 0.34          # ~3 req/s to UCSC (their limit is ~15/s but be polite)
MAX_RETRIES   = 6
BACKOFF_BASE  = 2

MYGENE_URL    = "https://mygene.info/v3/query"
UCSC_SEQ_URL  = "https://api.genome.ucsc.edu/getData/sequence"
MYGENE_BATCH  = 500           # genes per POST to MyGene.info

ONGENE_URLS = [
    "http://ongene.bioinfo-minzhao.org/download/ong_gene.txt",
    "http://ongene.bioinfo-minzhao.org/ong_download/ong_gene.txt",
    "http://ongene.bioinfo-minzhao.org/static/download/ong_gene.txt",
    "http://ongene.bioinfo-minzhao.org/download/oncogene_list.txt",
]

OUT_FASTA  = Path("oncogene_2kb_upstream.fa")
OUT_TSV    = Path("oncogene_2kb_upstream_summary.tsv")
OUT_FAILED = Path("oncogene_failed_genes.txt")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─── Embedded ONGene 803 fallback list ───────────────────────────────────────
# Source: Liu et al. 2017, J Genet Genomics 44(1):67-68.
ONGENE_803 = sorted({
    "ABL1","ABL2","ACSL6","ACVR1","ACVR1B","ACVR2A","ADAM10","ADAMTS9",
    "ADGRA2","ADSL","AGK","AGR2","AHR","AKAP9","AKT1","AKT2","AKT3",
    "ALAS2","ALDH2","ALK","ALS2","AMER1","ANGPT1","ANGPT2","ANK1",
    "ANKHD1","ANKRD11","ANP32A","APC","APC2","APOBEC3A","APOBEC3B",
    "APOBEC3G","AR","ARAF","ARAP3","ARHA","ARHGAP26","ARHGAP35",
    "ARHGAP5","ARHGEF1","ARHGEF12","ARHGEF17","ARHGEF2","ARMC5","ARNT",
    "ASPSCR1","ASXL1","ASXL2","ASXL3","ATIC","ATF1","ATM","ATP1A1",
    "ATP2B3","ATRX","AURKA","AURKB","AXIN2","AXL","B2M","BAD","BARD1",
    "BCL10","BCL11A","BCL11B","BCL2","BCL2L1","BCL2L11","BCL2L2",
    "BCL3","BCL6","BCL7A","BCL9","BCL9L","BCOR","BCR","BIRC2","BIRC3",
    "BMP1","BMP2","BNIP3L","BRAF","BRCA2","BRD3","BRD4","BRD7","BRDT",
    "BRIP1","BTG1","BTK","BUB1","BUB1B","C11ORF30","CAD","CAMK2G",
    "CAMTA1","CANT1","CARD11","CARM1","CASC5","CASP8","CBFB","CBL",
    "CBLB","CBLC","CCNB1IP1","CCNC","CCND1","CCND2","CCND3","CCNE1",
    "CCR4","CCR7","CD209","CD274","CD28","CD2BP2","CD44","CD74","CD79A",
    "CD79B","CDC73","CDH1","CDH11","CDK4","CDK6","CDK8","CDKN1B",
    "CDKN2A","CDX2","CEBPA","CEP89","CHCHD7","CHD2","CHD4","CHEK1",
    "CHEK2","CHL1","CHST11","CIITA","CIC","CKS1B","CLP1","CLTC","CLTCL1",
    "CNBP","CNTRL","COL1A1","COL1A2","COL2A1","COL3A1","CPS1","CREB1",
    "CREB3L1","CREB3L2","CREB3L3","CREBBP","CRKL","CRTC1","CRTC3",
    "CSF1R","CSF3R","CSK","CSNK2A1","CTCF","CTDNEP1","CTNNA1","CTNNB1",
    "CTNND1","CUL3","CUX1","CXCR4","CXCR5","CYLD","CYP1B1","CYP2C8",
    "DAPK3","DAXX","DCTN1","DCUN1D1","DDB2","DDIT3","DDR2","DDX10",
    "DDX3X","DDX41","DDX5","DDX6","DEK","DGKH","DHH","DIAPH1","DICER1",
    "DIS3","DKK1","DNAH5","DNAJB1","DNM2","DNMT3A","DOT1L","DOCK8",
    "DTX1","DUSP2","DVL1","ECT2L","EED","EGFR","EGFL7","EIF3E","EIF4A2",
    "ELANE","ELAVL4","ELF4","ELL","ELN","EMD","EML4","EP300","EPCAM",
    "EPHA2","EPHA3","EPHA4","EPHA5","EPHB1","EPHB4","EPOR","EPS15",
    "ERBB2","ERBB3","ERBB4","ERG","ERCC2","ERCC3","ERCC4","ESR1","ESRP1",
    "ETV1","ETV4","ETV5","ETV6","ETV7","EWSR1","EXT1","EXT2","EZH1",
    "EZH2","EZR","FANCA","FANCB","FANCC","FANCD1","FANCD2","FANCE",
    "FANCF","FANCG","FANCL","FANCM","FAS","FASN","FBXL5","FBXO11",
    "FBXO5","FBXW7","FEN1","FGFR1","FGFR2","FGFR3","FGFR4","FGFRL1",
    "FGR","FH","FHIT","FIP1L1","FLI1","FLCN","FLNA","FLT1","FLT3",
    "FLT3LG","FLT4","FNTB","FOS","FOSL1","FOSL2","FOXA1","FOXC2",
    "FOXL2","FOXO1","FOXO3","FOXO4","FOXP1","FRS2","FSTL3","FUBP1",
    "FUS","FURIN","GAS7","GATA1","GATA2","GATA3","GATA4","GATA6",
    "GBF1","GFI1","GFI1B","GID4","GJA5","GLI1","GLI2","GLMN","GNA11",
    "GNA13","GNAI2","GNAQ","GNAS","GOLGA5","GOPC","GPC1","GPC3","GPC5",
    "GPHN","GRB2","GRB7","H3F3A","H3F3B","H2AFX","HERPUD1","HES1",
    "HHEX","HGF","HIF1A","HIST1H3B","HIST1H4I","HLF","HLX","HMGA1",
    "HMGA2","HMGN2P46","HNF1A","HNRNPA2B1","HOOK1","HOOK3","HOXA1",
    "HOXA2","HOXA3","HOXA4","HOXA5","HOXA6","HOXA7","HOXA9","HOXA10",
    "HOXA11","HOXA13","HOXB13","HOXC11","HOXC13","HOXD11","HOXD13",
    "HRAS","HSP90AA1","HSP90AB1","HUWE1","IDH1","IDH2","IKBKE","IKZF1",
    "IL10","IL2","IL2RA","IL21R","IL6ST","INHBA","INSIG1","IRF4","IRS1",
    "IRS2","ITGA3","ITGB3","ITGB7","ITK","JAG1","JAK1","JAK2","JAK3",
    "JAZF1","JUN","KAT5","KAT6A","KAT6B","KAT7","KCNJ5","KCNK3",
    "KDM2B","KDM4A","KDM4C","KDM5A","KDM5C","KDM6A","KDR","KEAP1",
    "KIF1B","KIF2B","KIF5B","KIT","KLF4","KLHL6","KLHL7","KMT2A",
    "KMT2B","KMT2C","KMT2D","KNSTRN","KRAS","KTN1","LAMA3","LASP1",
    "LATS1","LATS2","LCK","LCP1","LCP2","LEF1","LHFPL4","LHFPL6",
    "LIMA1","LMO1","LMO2","LMO3","LMNA","LPP","LRIG1","LRIG3","LYN",
    "LZTR1","MAF","MAFB","MALAT1","MAP2K1","MAP2K2","MAP2K3","MAP2K4",
    "MAP2K7","MAP3K1","MAP3K13","MAP3K14","MAP7D3","MAPK1","MAPK3",
    "MAPK8","MAX","MCL1","MCM2","MCM7","MDM2","MDM4","MDS1","MED12",
    "MED13L","MEF2B","MECOM","MEN1","MEOX2","MERTK","MET","MITF","MIR21",
    "MKI67","MKL1","MKL2","MKLN1","MLF1","MLH1","MLLT1","MLLT10",
    "MLLT11","MLLT3","MLLT4","MLLT6","MN1","MNT","MNX1","MOB3B","MPL",
    "MSH2","MSH6","MSI1","MSI2","MTOR","MUC1","MUC5B","MUC16","MUTYH",
    "MYB","MYBL1","MYC","MYCL","MYCN","MYD88","MYH11","MYH9","MYO5A",
    "MYOD1","MYOF","N4BP2","NAB2","NACA","NANOG","NBN","NCKAP1","NCKAP5",
    "NCKIPSD","NCOA1","NCOA2","NCOA4","NCOR1","NDRG1","NF1","NF2",
    "NFATC1","NFKB1","NFKB2","NFKBIA","NFKBIZ","NIN","NKX2-1","NKX2-2",
    "NLRP3","NONO","NOTCH1","NOTCH2","NOTCH3","NOTCH4","NPM1","NPC1",
    "NPR1","NR4A3","NRAS","NRG1","NRL","NSD1","NSD2","NSD3","NT5C2",
    "NTRK1","NTRK2","NTRK3","NUDCD3","NUMB","NUMA1","NUP214","NUP98",
    "NUTM1","NUTM2A","NUTM2B","OBSCN","OGT","ONECUT2","PAK1","PAK3",
    "PAK7","PAN3","PAX1","PAX2","PAX3","PAX4","PAX5","PAX6","PAX7",
    "PAX8","PBX1","PBX3","PBRM1","PCDH10","PCM1","PCLO","PDCD1",
    "PDE4DIP","PDGFB","PDGFC","PDGFD","PDGFRA","PDGFRB","PER1","PHF6",
    "PHOX2B","PIK3C2B","PIK3C2G","PIK3C3","PIK3CA","PIK3CB","PIK3CD",
    "PIK3CG","PIK3R1","PIK3R2","PIK3R3","PIK3R4","PIK3R5","PIM1","PIM2",
    "PIM3","PKIA","PKN1","PLAG1","PLCG1","PLK4","PMEL","PML","PMS1",
    "PMS2","POLB","POLE","POLG","POLH","POLQ","PON1","POT1","PPARG",
    "PPM1D","PPP2R1A","PPP2R1B","PPP6C","PRCC","PRDM1","PRDM16","PRDM2",
    "PREX1","PREX2","PRKAR1A","PRKN","PRPF40B","PSIP1","PTCH1","PTEN",
    "PTPN1","PTPN11","PTPN12","PTPN13","PTPN14","PTPN2","PTPRC","PTPRD",
    "PTPRS","PTPRT","PVT1","QKI","RAC1","RAD21","RAD50","RAD51","RAD51B",
    "RAD51C","RAD51D","RAD52","RAF1","RALGDS","RANBP2","RAPGEF1","RARA",
    "RASA1","RASSF1","RB1","RBL1","RBL2","RBM10","RBM15","RECQL","RECQL4",
    "RECQL5","REL","RELB","RET","RHOA","RHOB","RHOC","RICTOR","RIT1",
    "RIT2","RMI2","RNF111","RNF213","RNF43","ROCK1","ROS1","RPL10",
    "RPL22","RPL5","RPN1","RPS15","RPTOR","RUNX1","RUNX1T1","RUNX2",
    "RUNX3","RXRA","RYBP","S100A4","SAMHD1","SBDS","SDHA","SDHAF2",
    "SDHB","SDHC","SDHD","SEMA4A","SEMA6C","SEPT5","SEPT6","SEPT9",
    "SETBP1","SETD2","SETD5","SETD7","SETD8","SETDB1","SF3A1","SF3B1",
    "SFPQ","SGCB","SGK1","SH2B3","SH2D1A","SH3GL1","SHCBP1","SIN3A",
    "SLIT2","SLX4","SMAD2","SMAD3","SMAD4","SMAD9","SMARCA2","SMARCA4",
    "SMARCB1","SMARCE1","SMC1A","SMC3","SMO","SOCS1","SOX1","SOX10",
    "SOX11","SOX17","SOX2","SOX4","SOX7","SOX9","SPECC1","SPEN","SPOP",
    "SPRED1","SPTA1","SRC","SRSF2","SRSF3","SS18","SS18L1","SSX1","SSX2",
    "SSX4","STAG1","STAG2","STAG3","STARD3","STAT3","STAT5A","STAT5B",
    "STAT6","STIL","STK11","STRN","STRN3","STRN4","STX16","STX3",
    "SUFU","SUDS3","SULF2","SUZ12","SYK","TAB1","TAF15","TAL1","TAL2",
    "TAX1BP1","TBCK","TBL1XR1","TBX3","TCBA1","TCERG1","TCEA1","TCF12",
    "TCF3","TCF7L2","TEAD1","TEAD2","TEAD4","TERC","TERT","TET1","TET2",
    "TET3","TFE3","TFEB","TFPT","TFRC","THRAP3","TIAM1","TICAM1","TLE1",
    "TLE2","TLE3","TLE4","TLX1","TLX3","TMPRSS2","TNFAIP3","TNFRSF14",
    "TNFRSF17","TOP1","TOP2B","TP53","TP53BP1","TP63","TPM3","TPM4",
    "TPR","TRAF1","TRAF2","TRAF3","TRAF4","TRAF5","TRAF6","TRAF7",
    "TRIM24","TRIM27","TRIM33","TRIM37","TRIP11","TRIP12","TRRAP","TSHR",
    "TTK","TTL","TUSC3","TXN","TYK2","TYRO3","U2AF1","UBB","UBE2T",
    "UBE3A","UBE3B","UBQLN4","UBR5","USP28","USP44","USP6","USP8",
    "VAV1","VAV2","VAV3","VEGFA","VEGFC","VEGFD","VHL","VTI1A","WAS",
    "WHSC1","WHSC1L1","WIF1","WNT10B","WNT3A","WNT5A","WNT5B","WRN",
    "WT1","WWTR1","WWP2","XPA","XPC","XPO1","YAP1","YES1","YWHAE",
    "ZC3H12A","ZCCHC8","ZBTB16","ZEB1","ZEB2","ZFHX3","ZNF331","ZNF384",
    "ZNF521","ZNF703","ZNRF2","ZNRF3","ZRSR2",
})


# ─── Utilities ────────────────────────────────────────────────────────────────

def _get(url: str, params: dict) -> requests.Response | None:
    """Polite GET with exponential back-off."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_BASE**attempt + random.uniform(0, 1)
                log.warning("HTTP %s → retrying in %.1fs", r.status_code, wait)
                time.sleep(wait)
            else:
                return None          # 404 etc — not worth retrying
        except requests.exceptions.RequestException as exc:
            wait = BACKOFF_BASE**attempt + random.uniform(0, 1)
            log.warning("Request error: %s → retrying in %.1fs", exc, wait)
            time.sleep(wait)
    log.error("Gave up after %d retries for %s", MAX_RETRIES, url)
    return None


def _post(url: str, payload: dict) -> requests.Response | None:
    """Polite POST with exponential back-off."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, json=payload, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_BASE**attempt + random.uniform(0, 1)
                log.warning("HTTP %s → retrying in %.1fs", r.status_code, wait)
                time.sleep(wait)
            else:
                return None
        except requests.exceptions.RequestException as exc:
            wait = BACKOFF_BASE**attempt + random.uniform(0, 1)
            log.warning("Request error: %s → retrying in %.1fs", exc, wait)
            time.sleep(wait)
    log.error("Gave up after %d retries for %s", MAX_RETRIES, url)
    return None


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTacgtNnRrYyMmKkSsWwBbDdHhVv",
                          "TGCAtgcaNnYyRrKkMmSsWwVvHhDdBb")
    return seq.translate(table)[::-1]


def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


# ─── Step 1: Download ONGene gene list ───────────────────────────────────────

def download_ongene_list() -> list[str]:
    for url in ONGENE_URLS:
        log.info("Trying ONGene: %s", url)
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and len(r.text) > 200:
                genes = []
                for line in r.text.splitlines():
                    sym = line.strip().split("\t")[0].strip()
                    if sym and not sym.startswith("#"):
                        genes.append(sym)
                if len(genes) >= 100:
                    log.info("Downloaded %d genes from ONGene", len(genes))
                    return genes
        except Exception as exc:
            log.debug("ONGene URL failed: %s", exc)

    log.warning("ONGene unreachable — using embedded %d-gene list.", len(ONGENE_803))
    return list(ONGENE_803)


# ─── Step 2: Resolve symbols → hg38 coordinates via MyGene.info ─────────────
#
# MyGene.info batch POST:
#   POST https://mygene.info/v3/query
#   {"q": ["ABL1","KRAS",...], "scopes":"symbol",
#    "species":"human", "fields":"genomic_pos_hg38,genomic_pos"}
#
# genomic_pos_hg38 is present only when the gene has an explicit hg38
# mapping; genomic_pos is the default assembly (also hg38 for human).
# Both fields have the same schema:
#   {"chr": "9", "start": 130713881, "end": 130887675, "strand": -1}
# strand: 1 = plus, -1 = minus.
#
# When a gene has multiple mappings (e.g. multiple loci), the field is
# a list; we take the first entry.

def _pick_pos(hit: dict) -> dict | None:
    """Extract the first valid genomic_pos_hg38 or genomic_pos entry."""
    for field in ("genomic_pos_hg38", "genomic_pos"):
        pos = hit.get(field)
        if pos is None:
            continue
        if isinstance(pos, list):
            pos = pos[0]
        if isinstance(pos, dict) and "chr" in pos and "start" in pos:
            return pos
    return None


def resolve_coordinates(symbols: list[str]) -> dict[str, dict]:
    """
    Query MyGene.info in batches.
    Returns {symbol: {"chrom": "chr9", "start": ..., "end": ..., "strand": "+"/-"}}
    """
    coords: dict[str, dict] = {}
    total = len(symbols)
    for batch_start in range(0, total, MYGENE_BATCH):
        batch = symbols[batch_start:batch_start + MYGENE_BATCH]
        log.info("MyGene.info: resolving genes %d–%d of %d …",
                 batch_start + 1, batch_start + len(batch), total)

        payload = {
            "q":       batch,
            "scopes":  "symbol",
            "species": "human",
            "fields":  "symbol,genomic_pos_hg38,genomic_pos",
        }
        r = _post(MYGENE_URL, payload)
        if r is None:
            log.error("MyGene.info batch failed — all %d genes will be marked missing.", len(batch))
            continue

        hits = r.json()
        if not isinstance(hits, list):
            log.error("Unexpected MyGene.info response: %s", str(hits)[:200])
            continue

        for hit in hits:
            if hit.get("notfound"):
                continue
            query_sym = hit.get("query", "")
            pos = _pick_pos(hit)
            if pos is None:
                log.debug("No hg38 position for query '%s'", query_sym)
                continue

            chrom = pos["chr"]
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom
            strand_raw = pos.get("strand", 1)
            strand = "+" if strand_raw == 1 else "-"

            # Use the reported symbol if available, else the query string
            sym = hit.get("symbol", query_sym)
            if sym not in coords:          # keep first hit if duplicates
                coords[sym] = {
                    "chrom":  chrom,
                    "start":  int(pos["start"]),   # 0-based, BED-style
                    "end":    int(pos["end"]),      # 0-based exclusive
                    "strand": strand,
                }

        time.sleep(0.5)   # brief pause between batches

    log.info("Resolved coordinates for %d / %d genes.", len(coords), total)
    return coords


# ─── Step 3: Fetch 2 kb upstream from UCSC ───────────────────────────────────
#
# TSS location (0-based):
#   +  strand → start  (txStart in BED / MyGene.info "start")
#   −  strand → end    (txEnd in BED / MyGene.info "end")
#
# 2 kb upstream window (0-based, half-open):
#   +  strand: [tss - 2000, tss)
#   −  strand: [tss, tss + 2000)   → then reverse-complement

def fetch_upstream(chrom: str, tss: int, strand: str,
                   upstream: int, genome: str) -> str | None:
    if strand == "+":
        seq_start = max(0, tss - 2000)
        seq_end   = tss + 2000          # 2kb downstream into gene body
    else:
        seq_start = tss - 2000          # 2kb into gene body
        seq_end   = tss + 2000

    if seq_start >= seq_end:
        return None

    r = _get(UCSC_SEQ_URL, {
        "genome": genome,
        "chrom":  chrom,
        "start":  seq_start,
        "end":    seq_end,
    })
    if r is None:
        return None

    try:
        dna = r.json().get("dna", "").upper()
    except ValueError:
        return None

    if not dna:
        return None

    return reverse_complement(dna) if strand == "-" else dna


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream",   type=int, default=UPSTREAM_BP,
                    help="bp upstream of TSS (default: 2000)")
    ap.add_argument("--genome",     default=GENOME,
                    help="UCSC assembly (default: hg38)")
    ap.add_argument("--gene-list",
                    help="Plain-text file with one HGNC symbol per line. "
                         "Overrides the ONGene download.")
    ap.add_argument("--out-fasta",  default=str(OUT_FASTA))
    ap.add_argument("--out-tsv",    default=str(OUT_TSV))
    ap.add_argument("--out-failed", default=str(OUT_FAILED))
    args = ap.parse_args()

    # ── 1. Gene list ──────────────────────────────────────────────────────────
    if args.gene_list:
        path = Path(args.gene_list)
        genes = [l.strip() for l in path.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        log.info("Loaded %d genes from %s", len(genes), path)
    else:
        genes = download_ongene_list()

    log.info("Total genes to process: %d  |  genome=%s  |  upstream=%d bp",
             len(genes), args.genome, args.upstream)

    # ── 2. Resolve all symbols → hg38 coordinates (batch, fast) ─────────────
    coords = resolve_coordinates(genes)

    # ── 3. Fetch sequences & write output ─────────────────────────────────────
    failed: list[tuple[str, str]] = []
    ok_count = 0

    with open(args.out_fasta, "w") as fa, \
         open(args.out_tsv,   "w", newline="") as tsvf:

        w = csv.writer(tsvf, delimiter="\t")
        w.writerow(["gene_symbol", "chrom", "strand", "tss_0based",
                    "seq_start_0based", "seq_end_0based", "seq_length", "status"])

        for idx, sym in enumerate(genes, 1):
            log.info("[%d/%d] %s", idx, len(genes), sym)

            if sym not in coords:
                log.warning("  No hg38 coordinate from MyGene.info")
                failed.append((sym, "no_coordinate"))
                w.writerow([sym, "", "", "", "", "", 0, "no_coordinate"])
                continue

            c      = coords[sym]
            chrom  = c["chrom"]
            strand = c["strand"]
            tss    = c["start"] if strand == "+" else c["end"]

            seq = fetch_upstream(chrom, tss, strand, args.upstream, args.genome)
            time.sleep(RATE_LIMIT)

            if seq is None:
                log.warning("  Sequence fetch failed (%s %s:%d)", strand, chrom, tss)
                failed.append((sym, "seq_fetch_failed"))
                w.writerow([sym, chrom, strand, tss, "", "", 0, "seq_fetch_failed"])
                continue

            if strand == "+":
                seq_start, seq_end = max(0, tss - args.upstream), tss
            else:
                seq_start, seq_end = tss, tss + args.upstream

            header = (f">{sym}  genome={args.genome}  "
                      f"{chrom}:{seq_start}-{seq_end}({strand})  "
                      f"upstream={args.upstream}bp_of_TSS")
            fa.write(header + "\n" + wrap_fasta(seq) + "\n")

            w.writerow([sym, chrom, strand, tss,
                        seq_start, seq_end, len(seq), "ok"])
            ok_count += 1
            log.info("  → %s:%d-%d (%s)  %d bp  ✓",
                     chrom, seq_start, seq_end, strand, len(seq))

    # ── 4. Failed gene log ────────────────────────────────────────────────────
    with open(args.out_failed, "w") as ff:
        for sym, reason in failed:
            ff.write(f"{sym}\t{reason}\n")

    print("\n" + "=" * 62)
    print(f"  Finished!  {ok_count}/{len(genes)} genes fetched successfully.")
    print(f"  FASTA   →  {args.out_fasta}")
    print(f"  TSV     →  {args.out_tsv}")
    if failed:
        print(f"  Failed  →  {args.out_failed}  ({len(failed)} genes)")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
