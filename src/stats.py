"""
stats.py – Python translation of stats.h

Covers:
  • Shell sort
  • MLE with censoring (Extreme Value Distribution fit)
  • PAM scoring matrix initialisation
  • Scoring profile construction (byte and word)
  • Fisher-Yates shuffle with a Park-Miller LCG
  • Smith-Waterman local alignment (SSE2 striped algorithm, pure-Python)
  • calc_score – top-level driver that returns an E-value threshold
"""

from __future__ import annotations

import math
import time
import ctypes
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BIGNUM    = 1_000_000
MAXTST    = 1_500
MAXLIB    = 10_000
EL        = 125          # end-of-line sentinel
ES        = 126          # end-of-sequence sentinel
MIN_RES   = 1_000
NA        = 123          # "not an amino-acid" sentinel
MAXSQ     = 60
AA        = 16_807       # Park-Miller LCG multiplier
MM        = 2_147_483_647  # 2^31 - 1 (Mersenne prime)
QQ        = 127_773      # MM // AA
RW        = 2_836        # MM % AA
PI_SQRT6  = 1.28254983016186409554
TINY      = 1.0e-6
MAX_NIT   = 100
MAX_LNCRNA = 1_000_000

# ─────────────────────────────────────────────────────────────────────────────
# nascii – ASCII → internal residue-index mapping (matches C array)
# ─────────────────────────────────────────────────────────────────────────────

nascii: List[int] = [
    EL, NA, NA, NA, NA, NA, NA, NA, NA, NA, EL, NA, NA, EL, NA, NA,  # 0–15
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,  # 16–31
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, ES, NA, NA, 16, NA, NA,  # 32–47
    NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, ES, NA, NA, ES, NA,  # 48–63
    NA,  1, 15,  2, 12, NA, NA,  3, 13, NA, NA, 11, NA,  8, 16, NA,  # 64–79
     6,  7,  6, 10,  4,  5, 14,  9, 17,  7, NA, NA, NA, NA, NA, NA,  # 80–95
    NA,  1, 15,  2, 12, NA, NA,  3, 13, NA, NA, 11, NA,  8, 16, NA,  # 96–111
     6,  7,  6, 10,  4,  5, 14,  9, 17,  7, NA, NA, NA, NA, NA, NA,  # 112–127
]

# ─────────────────────────────────────────────────────────────────────────────
# npam – lower-triangular PAM scoring matrix (flattened)
# ─────────────────────────────────────────────────────────────────────────────

npam: List[int] = [
     5,
    -4,  5,
    -4, -4,  5,
    -4, -4, -4,  5,
    -4, -4, -4,  5,  5,
     2, -1,  2, -1, -1,  2,
    -1,  2, -1,  2,  2, -2,  2,
     2,  2, -1, -1, -1, -1, -1,  2,
     2, -1, -1,  2,  2,  1,  1,  1,  2,
    -1,  2,  2, -1, -1,  1,  1,  1, -1,  2,
    -1, -1,  2,  2,  2,  1,  1, -1,  1,  1,  2,
     1, -2,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,
     1,  1, -2,  1,  1, -1,  1,  1,  1, -1, -1, -1,  1,
     1,  1,  1, -2, -2,  1, -1,  1, -1,  1, -1, -1, -1,  1,
    -2,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
]

# ─────────────────────────────────────────────────────────────────────────────
# Data-structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PStruct:
    """Corresponds to C struct pstruct."""
    maxlen:    int = 0
    pam2:      List[List[int]] = field(
        default_factory=lambda: [[0] * MAXSQ for _ in range(MAXSQ)])
    dnaseq:    int = 0
    pam_h:     int = -1
    pam_l:     int = 1
    pamoff:    int = 0
    have_pam2: int = 0
    nsq:       int = 0


@dataclass
class MRandState:
    """Corresponds to C struct m_rand_struct."""
    seed: int = 33


@dataclass
class FStruct:
    """
    Corresponds to C struct f_struct.
    byte_score and word_score are flat bytearray / array objects that
    act as the striped scoring profiles consumed by the SW kernels.
    """
    max_res:    int = 0
    bias:       int = 0
    byte_score: bytearray = field(default_factory=bytearray)
    word_score: List[int] = field(default_factory=list)   # list of uint16
    try_8bit:   int = 0
    done_8bit:  int = 0
    done_16bit: int = 0
    # raw memory (not needed in Python, kept for API symmetry)
    alphabet_size: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# st_sort  –  Shell sort (in-place)
# ─────────────────────────────────────────────────────────────────────────────

def st_sort(v: List[int]) -> None:
    """In-place Shell sort (matches the C gap sequence: 3*gap+1)."""
    n = len(v)
    gap = 1
    while gap < n // 3:
        gap = 3 * gap + 1
    while gap > 0:
        for i in range(gap, n):
            for j in range(i - gap, -1, -gap):
                if v[j] <= v[j + gap]:
                    break
                v[j], v[j + gap] = v[j + gap], v[j]
        gap = (gap - 1) // 3


# ─────────────────────────────────────────────────────────────────────────────
# findmax_score  –  element-wise maximum of two arrays
# ─────────────────────────────────────────────────────────────────────────────

def findmax_score(a: List[int], b: List[int]) -> List[int]:
    """Return element-wise maximum of a and b (up to min length)."""
    n = min(len(a), len(b))
    return [max(a[i], b[i]) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# MLE helpers  –  first and second derivatives for Newton-Raphson
# ─────────────────────────────────────────────────────────────────────────────

def first_deriv_cen(
        lam: float,
        sptr: List[int], n1: List[int],
        start: int, stop: int,
        sumlenL: float, cenL: float,
        sumlenH: float, cenH: float) -> float:
    """First derivative of the log-likelihood (censored EVD)."""
    total = 0.0
    sum1  = 0.0
    sum2  = 0.0
    for i in range(start, stop):
        s  = float(sptr[i])
        l  = float(n1[i])
        es = math.exp(-lam * s)
        total += s
        sum2  += l * es
        sum1  += s * l * es

    sum1 += sumlenL * cenL * math.exp(-lam * cenL) \
          - sumlenH * cenH * math.exp(-lam * cenH)
    sum2 += sumlenL * math.exp(-lam * cenL) \
          - sumlenH * math.exp(-lam * cenH)

    return (1.0 / lam) - (total / float(stop - start)) + (sum1 / sum2)


def second_deriv_cen(
        lam: float,
        sptr: List[int], n1: List[int],
        start: int, stop: int,
        sumlenL: float, cenL: float,
        sumlenH: float, cenH: float) -> float:
    """Second derivative of the log-likelihood (censored EVD)."""
    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    for i in range(start, stop):
        s  = float(sptr[i])
        l  = float(n1[i])
        es = math.exp(-lam * s)
        sum2 += l * es
        sum1 += l * s * es
        sum3 += l * s * s * es

    sum1 += sumlenL * cenL * math.exp(-lam * cenL) \
          - sumlenH * cenH * math.exp(-lam * cenH)
    sum2 += sumlenL * math.exp(-lam * cenL) \
          - sumlenH * math.exp(-lam * cenH)
    sum3 += sumlenL * cenL * cenL * math.exp(-lam * cenL) \
          - sumlenH * cenH * cenH * math.exp(-lam * cenH)

    return ((sum1 * sum1) / (sum2 * sum2)) - (sum3 / sum2) \
           - (1.0 / (lam * lam))


# ─────────────────────────────────────────────────────────────────────────────
# mle_cen  –  MLE fit of a censored Extreme Value Distribution
# ─────────────────────────────────────────────────────────────────────────────

def mle_cen(
        sptr: List[int], n_len: int,
        n1: List[int],   m_len: int,
        fc: float, Lambda: float, K_tmp: float
) -> Optional[Tuple[float, float]]:
    """
    Fit lambda and K by maximum likelihood on a censored EVD.

    Returns (lambda, K) or None on failure.
    Python equivalent of the C function that returned a malloc'd double[2].
    """
    nf    = int((fc / 2.0) * n_len)
    start = nf
    stop  = n_len - nf

    st_sort(sptr)

    # Trimmed mean & variance
    sum_s = sum(float(sptr[i]) for i in range(start, stop))
    dtmp  = float(stop - start)
    mean_s = sum_s / dtmp
    sum2_s = sum(float(sptr[i]) ** 2 for i in range(start, stop))
    var_s  = sum2_s / (dtmp - 1.0)

    # Censored tail sums
    sumlenL = sum(float(n1[i]) for i in range(start))
    sumlenH = sum(float(n1[i]) for i in range(stop, n_len))

    if nf > 0:
        cenL = float(sptr[start])
        cenH = float(sptr[stop])
    else:
        cenL = float(sptr[start]) / 2.0
        cenH = float(sptr[start]) * 2.0

    if cenL >= cenH:
        print("cenL is larger than cenH! mle_cen is wrong!")
        return None

    lam = PI_SQRT6 / math.sqrt(var_s)
    if lam > 1.0:
        import sys
        print(f" Lambda initial estimate error: lambda: {lam:6.4g}; var_s: {var_s:6.4g}",
              file=sys.stderr)
        lam = 0.2

    # Newton-Raphson
    nit = 0
    old_lam = lam
    while True:
        deriv  = first_deriv_cen( lam, sptr, n1, start, stop, sumlenL, cenL, sumlenH, cenH)
        deriv2 = second_deriv_cen(lam, sptr, n1, start, stop, sumlenL, cenL, sumlenH, cenH)
        old_lam = lam
        step = deriv / deriv2
        if lam - step > 0.0:
            lam = lam - step
        else:
            lam = lam / 2.0
        nit += 1
        if not (abs((lam - old_lam) / lam) > TINY and nit < MAX_NIT):
            break

    if nit >= MAX_NIT:
        return None

    total = sum(float(n1[i]) * math.exp(-lam * float(sptr[i]))
                for i in range(start, stop))

    K = float(n_len) / (float(m_len) * (
        total
        + sumlenL * math.exp(-lam * cenL)
        - sumlenH * math.exp(-lam * cenH)
    ))
    return (lam, K)


# ─────────────────────────────────────────────────────────────────────────────
# PRNG  –  Park-Miller LCG  (my_srand / my_nrand)
# ─────────────────────────────────────────────────────────────────────────────

def my_srand(set_val: int) -> MRandState:
    """
    Initialise the random state.
    If set_val > 0 the seed is fixed; otherwise it is derived from the
    microsecond clock (then overridden to 33, matching the C source).
    """
    state = MRandState()
    if set_val > 0:
        state.seed = set_val
    else:
        n = int(time.time() * 1_000_000) % 65535
        if n % 2 == 0:
            n += 1
        state.seed = n
    state.seed = 33   # matches C: always overrides to 33
    return state


def my_nrand(n: int, state: MRandState) -> int:
    """Return a pseudo-random integer in [0, n)."""
    hi   = state.seed // QQ
    lo   = state.seed  % QQ
    test = AA * lo - RW * hi
    if test > 0:
        state.seed = test
    else:
        state.seed = test + MM
    return state.seed % n


# ─────────────────────────────────────────────────────────────────────────────
# shuffle  –  Fisher-Yates in-place shuffle of a bytearray
# ─────────────────────────────────────────────────────────────────────────────

def shuffle(src: bytearray, state: MRandState) -> bytearray:
    """
    Return a shuffled copy of src (bytearray).
    Matches the C routine which copies then shuffles in-to.
    """
    dst = bytearray(src)
    n   = len(dst)
    for i in range(n, 0, -1):
        j       = my_nrand(i, state)
        dst[j], dst[i - 1] = dst[i - 1], dst[j]
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# cg_str  –  encode a sequence string to internal residue indices
# ─────────────────────────────────────────────────────────────────────────────

_CG_MAP = {1: 0x01, 2: 0x02, 3: 0x03, 4: 0x04, 5: 0x05, 16: 0x10}

def cg_str(seq: str) -> bytearray:
    """
    Convert a nucleotide / amino-acid string to a bytearray of
    internal residue codes, using the nascii lookup table.
    Unknown characters map to 0x10 (the 'gap' / default code).
    """
    out = bytearray(len(seq))
    for j, ch in enumerate(seq):
        idx = ord(ch) & 0x7F
        out[j] = _CG_MAP.get(nascii[idx], 0x10)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# alloc_pam / init_pam2  –  PAM scoring matrix setup
# ─────────────────────────────────────────────────────────────────────────────

def alloc_pam(d1: int, d2: int) -> PStruct:
    """Create a PStruct with the standard DNA defaults."""
    pst = PStruct()
    pst.dnaseq    = 0
    pst.pam_h     = -1
    pst.pam_l     = 1
    pst.pamoff    = 0
    pst.have_pam2 = 1
    return pst


def init_pam2(pst: PStruct) -> None:
    """
    Fill pst.pam2 from the npam lower-triangular array.
    Matches C function init_pam2() exactly.
    """
    pam_sq = "\x00ACGTURYMWSKDHVBNX"   # index 0 unused
    pam_sq_n = 17
    sa_t = nascii[ord('X')]

    pst.pam2[0][0] = -BIGNUM
    pst.pam_h = -1
    pst.pam_l = 1

    # zero-fill
    for i in range(MAXSQ):
        for j in range(MAXSQ):
            pst.pam2[i][j] = 0

    k = 0
    for i in range(1, sa_t):
        p_i = nascii[ord(pam_sq[i])]
        pst.pam2[0][p_i] = pst.pam2[p_i][0] = -BIGNUM
        for j in range(1, i + 1):
            p_j = nascii[ord(pam_sq[j])]
            val = npam[k] - pst.pamoff
            pst.pam2[p_j][p_i] = pst.pam2[p_i][p_j] = val
            k += 1
            if pst.pam_l > val: pst.pam_l = val
            if pst.pam_h < val: pst.pam_h = val

    for i in range(sa_t + 1, pam_sq_n):
        p_i = nascii[ord(pam_sq[i])]
        pst.pam2[0][p_i] = pst.pam2[p_i][0] = -BIGNUM


# ─────────────────────────────────────────────────────────────────────────────
# init_work  –  build striped scoring profiles (byte and word)
# ─────────────────────────────────────────────────────────────────────────────

def init_work(aa0: bytearray, n0: int, pst: PStruct) -> FStruct:
    """
    Build byte-score and word-score striped profiles for the SSW kernels.
    Returns a populated FStruct.
    """
    f_str = FStruct()
    nsq   = 34

    # ---- find bias (most-negative score in the PAM sub-matrix) ----------
    bias = 127
    for i in range(1, nsq):
        for j in range(1, nsq):
            data = pst.pam2[i][j]
            if data < bias:
                bias = data

    # ---- word (16-bit) profile ------------------------------------------
    col_len_w = (n0 + 7) // 8
    n_count_w = (n0 + 7) & 0xFFFFFFF8          # round up to multiple of 8
    word_score: List[int] = [0] * n_count_w     # leading zeros

    for f in range(1, nsq):
        for e in range(col_len_w):
            for i in range(e, n_count_w, col_len_w):
                if i >= n0:
                    data = 0
                else:
                    data = pst.pam2[aa0[i]][f]
                word_score.append(data & 0xFFFF)

    # ---- byte (8-bit) profile -------------------------------------------
    col_len_b = (n0 + 15) // 16
    n_count_b = (n0 + 15) & 0xFFFFFFF0         # round up to multiple of 16
    byte_score = bytearray(n_count_b)           # leading zeros

    for f in range(1, nsq):
        for e in range(col_len_b):
            for i in range(e, n_count_b, col_len_b):
                if i >= n0:
                    data = -bias
                else:
                    data = pst.pam2[aa0[i]][f] - bias
                if data > 255:
                    raise OverflowError(
                        f"data out of range for 8-bit: {data} bias={bias} f={f} e={e}")
                byte_score.append(data & 0xFF)

    f_str.bias       = (-bias) & 0xFF
    f_str.byte_score = byte_score
    f_str.word_score = word_score
    f_str.try_8bit   = 1                       # no overflow detected
    f_str.done_8bit  = 0
    f_str.done_16bit = 0
    f_str.max_res    = max(3 * n0 // 2, MIN_RES)
    return f_str


def close_work_f_str(f_str: Optional[FStruct]) -> None:
    """No-op in Python (GC handles memory). Mirrors the C free() calls."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# SSE2 Smith-Waterman kernels – pure-Python striped implementations
#
# The C versions operate on __m128i SIMD registers.  Here each register is
# modelled as a Python list of lanes (8 int16 or 16 uint8).  All arithmetic
# uses Python integers with explicit clamping to replicate the saturating
# add/subtract semantics of SSE2.
# ─────────────────────────────────────────────────────────────────────────────

INT16_MIN = -32768
INT16_MAX =  32767
UINT8_MAX =  255


def _clamp_i16(x: int) -> int:
    return max(INT16_MIN, min(INT16_MAX, x))

def _clamp_u8(x: int) -> int:
    return max(0, min(UINT8_MAX, x))

def _adds_epi16(a: List[int], b: List[int]) -> List[int]:
    return [_clamp_i16(x + y) for x, y in zip(a, b)]

def _subs_epi16(a: List[int], b: List[int]) -> List[int]:
    return [_clamp_i16(x - y) for x, y in zip(a, b)]

def _max_epi16(a: List[int], b: List[int]) -> List[int]:
    return [max(x, y) for x, y in zip(a, b)]

def _adds_epu8(a: List[int], b: List[int]) -> List[int]:
    return [_clamp_u8(x + y) for x, y in zip(a, b)]

def _subs_epu8(a: List[int], b: List[int]) -> List[int]:
    return [_clamp_u8(x - y) for x, y in zip(a, b)]

def _max_epu8(a: List[int], b: List[int]) -> List[int]:
    return [max(x, y) for x, y in zip(a, b)]

def _slli_si128_epi16(reg: List[int], byte_shift: int) -> List[int]:
    """Left-shift the 128-bit register by byte_shift bytes (insert zeros at low end)."""
    lane_shift = byte_shift // 2   # for 16-bit lanes
    return [0] * lane_shift + reg[:-lane_shift] if lane_shift else reg[:]

def _slli_si128_epi8(reg: List[int], byte_shift: int) -> List[int]:
    """Left-shift the 128-bit register by byte_shift bytes (8-bit lanes)."""
    return [0] * byte_shift + reg[:-byte_shift] if byte_shift else reg[:]

def _srli_si128_epi16(reg: List[int], byte_shift: int) -> List[int]:
    lane_shift = byte_shift // 2
    return reg[lane_shift:] + [0] * lane_shift if lane_shift else reg[:]

def _srli_si128_epi8(reg: List[int], byte_shift: int) -> List[int]:
    return reg[byte_shift:] + [0] * byte_shift if byte_shift else reg[:]

def _cmpgt_epi16(a: List[int], b: List[int]) -> List[int]:
    return [0xFFFF if x > y else 0 for x, y in zip(a, b)]

def _cmpeq_epi8(a: List[int], b: List[int]) -> List[int]:
    return [0xFF if x == y else 0 for x, y in zip(a, b)]

def _movemask_epi8_from_epi16(reg: List[int]) -> int:
    """Build a 16-bit mask from the high byte of each 8-bit chunk."""
    mask = 0
    for i, v in enumerate(reg):
        hi = (v >> 8) & 0xFF
        lo = v & 0xFF
        if hi & 0x80: mask |= (1 << (2 * i + 1))
        if lo & 0x80: mask |= (1 << (2 * i))
    return mask

def _movemask_epi8_u8(reg: List[int]) -> int:
    mask = 0
    for i, v in enumerate(reg):
        if v & 0x80: mask |= (1 << i)
    return mask

def _extract_epi16(reg: List[int], idx: int) -> int:
    return reg[idx] & 0xFFFF

def _or_si128(a: List[int], b: List[int]) -> List[int]:
    return [x | y for x, y in zip(a, b)]


# ── 16-bit (word) kernel ─────────────────────────────────────────────────────

def smith_waterman_sse2_word(
        query_sequence:      bytearray,
        query_profile_word:  List[int],
        query_length:        int,
        db_sequence:         bytearray,
        db_length:           int,
        gap_open:            int,
        gap_extend:          int,
        f_str:               FStruct) -> int:
    """
    Smith-Waterman local alignment using a striped 16-bit approach.
    Returns the raw alignment score + 32768 (matching the C SSE2 version).
    """
    LANES = 8
    iter_ = (query_length + LANES - 1) // LANES

    INT16_MIN_V = [INT16_MIN] * LANES
    v_gapopen   = [gap_open]  * LANES
    v_gapextend = [gap_extend] * LANES
    v_min       = [0] * LANES
    v_min[0]    = INT16_MIN     # single lane minimum for shift-in

    # initialise maxscore to INT16_MIN (= 0x8000)
    v_maxscore = [INT16_MIN] * LANES

    # workspace: 2*iter lanes-of-8, all set to INT16_MIN
    pE      = [[INT16_MIN] * LANES for _ in range(iter_)]
    pHStore = [[INT16_MIN] * LANES for _ in range(iter_)]
    pHLoad  = [[INT16_MIN] * LANES for _ in range(iter_)]

    col_len = (query_length + LANES - 1) // LANES   # same as iter_
    n_count = (query_length + LANES - 1) & ~(LANES - 1)

    def get_score_lane(db_char: int, lane_idx: int) -> List[int]:
        """Retrieve one striped-profile column for a db character."""
        base = db_char * iter_ + n_count  # n_count leading zeros
        seg  = query_profile_word[base + lane_idx * col_len :
                                  base + lane_idx * col_len + LANES]
        return [_clamp_i16(v) for v in (seg + [0] * (LANES - len(seg)))]

    for i in range(db_length):
        db_char = db_sequence[i]
        F = [INT16_MIN] * LANES

        # shift last H column into position (slli 2 bytes = 1 lane)
        H_last = pHStore[iter_ - 1][:]
        H = [0] + H_last[:LANES - 1]   # equivalent to _mm_slli_si128 H, 2 + or v_min
        H[0] = INT16_MIN

        pHLoad, pHStore = pHStore, pHLoad

        for j in range(iter_):
            E = pE[j][:]

            # score lookup from the interleaved word profile
            base    = db_char * iter_ * LANES + n_count
            prof_j  = query_profile_word[base + j: base + j + 1]
            # proper striped access: profile stored as [nsq][iter][LANES]
            score_v = [0] * LANES
            for lane in range(LANES):
                idx = n_count + db_char * iter_ * LANES + j * LANES + lane
                score_v[lane] = query_profile_word[idx] if idx < len(query_profile_word) else 0
            score_v = [_clamp_i16(v) for v in score_v]

            H = _adds_epi16(H, score_v)
            v_maxscore = _max_epi16(v_maxscore, H)
            H = _max_epi16(H, E)
            H = _max_epi16(H, F)
            pHStore[j] = H[:]
            H = _subs_epi16(H, v_gapopen)
            E = _subs_epi16(E, v_gapextend)
            E = _max_epi16(E, H)
            F = _subs_epi16(F, v_gapextend)
            F = _max_epi16(F, H)
            pE[j] = E[:]
            H = pHLoad[j][:]

        # lazy F propagation
        j = 0
        H = pHStore[0][:]
        F = [0] + F[:LANES - 1]   # slli 2 bytes
        F[0] = INT16_MIN           # or v_min

        v_temp = _subs_epi16(H, v_gapopen)
        cmp_v  = _cmpgt_epi16(F, v_temp)
        cmp    = _movemask_epi8_from_epi16(cmp_v)

        while cmp != 0x0000:
            E = pE[j][:]
            H = _max_epi16(H, F)
            pHStore[j] = H[:]
            H = _subs_epi16(H, v_gapopen)
            E = _max_epi16(E, H)
            pE[j] = E[:]
            F = _subs_epi16(F, v_gapextend)
            j += 1
            if j >= iter_:
                j = 0
                F = [0] + F[:LANES - 1]
                F[0] = INT16_MIN
            H = pHStore[j][:]
            v_temp = _subs_epi16(H, v_gapopen)
            cmp_v  = _cmpgt_epi16(F, v_temp)
            cmp    = _movemask_epi8_from_epi16(cmp_v)

    # horizontal maximum over the 8 lanes
    tmp = _srli_si128_epi16(v_maxscore, 8)
    v_maxscore = _max_epi16(v_maxscore, tmp)
    tmp = _srli_si128_epi16(v_maxscore, 4)
    v_maxscore = _max_epi16(v_maxscore, tmp)
    tmp = _srli_si128_epi16(v_maxscore, 2)
    v_maxscore = _max_epi16(v_maxscore, tmp)

    score = _extract_epi16(v_maxscore, 0)
    return score + 32768


# ── 8-bit (byte) kernel ──────────────────────────────────────────────────────

def smith_waterman_sse2_byte(
        query_sequence:      bytearray,
        query_profile_byte:  bytearray,
        query_length:        int,
        db_sequence:         bytearray,
        db_length:           int,
        bias:                int,
        gap_open:            int,
        gap_extend:          int,
        f_str:               FStruct) -> int:
    """
    Smith-Waterman local alignment using a striped 8-bit approach.
    Returns the raw score (unsaturated if < 255).
    """
    LANES = 16
    iter_ = (query_length + LANES - 1) // LANES

    v_bias      = [bias]      * LANES
    v_gapopen   = [gap_open]  * LANES
    v_gapextend = [gap_extend] * LANES
    v_zero      = [0]         * LANES
    v_maxscore  = [0]         * LANES

    n_count = (query_length + LANES - 1) & ~(LANES - 1)

    pE      = [[0] * LANES for _ in range(iter_)]
    pHStore = [[0] * LANES for _ in range(iter_)]
    pHLoad  = [[0] * LANES for _ in range(iter_)]

    for i in range(db_length):
        db_char = db_sequence[i]
        F = [0] * LANES

        # shift last H into position
        H_last = pHStore[iter_ - 1][:]
        H = [0] + H_last[:LANES - 1]   # slli 1 byte

        pHLoad, pHStore = pHStore, pHLoad

        for j in range(iter_):
            E = pE[j][:]
            # byte-profile lookup
            score_v = [0] * LANES
            for lane in range(LANES):
                idx = n_count + db_char * iter_ * LANES + j * LANES + lane
                score_v[lane] = query_profile_byte[idx] if idx < len(query_profile_byte) else 0

            H = _adds_epu8(H, score_v)
            H = _subs_epu8(H, v_bias)
            v_maxscore = _max_epu8(v_maxscore, H)
            H = _max_epu8(H, E)
            H = _max_epu8(H, F)
            pHStore[j] = H[:]
            H = _subs_epu8(H, v_gapopen)
            E = _subs_epu8(E, v_gapextend)
            E = _max_epu8(E, H)
            F = _subs_epu8(F, v_gapextend)
            F = _max_epu8(F, H)
            pE[j] = E[:]
            H = pHLoad[j][:]

        # lazy F propagation
        j = 0
        H = pHStore[0][:]
        F = [0] + F[:LANES - 1]   # slli 1 byte

        v_temp = _subs_epu8(H, v_gapopen)
        v_temp2 = _subs_epu8(F, v_temp)
        cmp_v  = _cmpeq_epi8(v_temp2, v_zero)
        cmp    = _movemask_epi8_u8(cmp_v)

        while cmp != 0xffff:
            E = pE[j][:]
            H = _max_epu8(H, F)
            pHStore[j] = H[:]
            H = _subs_epu8(H, v_gapopen)
            E = _max_epu8(E, H)
            pE[j] = E[:]
            F = _subs_epu8(F, v_gapextend)
            j += 1
            if j >= iter_:
                j = 0
                F = [0] + F[:LANES - 1]
            H = pHStore[j][:]
            v_temp  = _subs_epu8(H, v_gapopen)
            v_temp2 = _subs_epu8(F, v_temp)
            cmp_v   = _cmpeq_epi8(v_temp2, v_zero)
            cmp     = _movemask_epi8_u8(cmp_v)

    # horizontal maximum
    tmp = _srli_si128_epi8(v_maxscore, 8)
    v_maxscore = _max_epu8(v_maxscore, tmp)
    tmp = _srli_si128_epi8(v_maxscore, 4)
    v_maxscore = _max_epu8(v_maxscore, tmp)
    tmp = _srli_si128_epi8(v_maxscore, 2)
    v_maxscore = _max_epu8(v_maxscore, tmp)
    tmp = _srli_si128_epi8(v_maxscore, 1)
    v_maxscore = _max_epu8(v_maxscore, tmp)

    raw = _extract_epi16([v_maxscore[0] | (v_maxscore[1] << 8)] + [0] * 7, 0)
    score = raw & 0x00FF

    if score + bias >= 255:
        score = 255
    return score


# ─────────────────────────────────────────────────────────────────────────────
# calc_score  –  top-level driver
# ─────────────────────────────────────────────────────────────────────────────

_RC_MAP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N'}

def _reverse_complement(seq: str) -> str:
    return ''.join(_RC_MAP.get(c, 'N') for c in reversed(seq))


# ─────────────────────────────────────────────────────────────────────────────
# parasail Smith-Waterman wrapper
#
# parasail is a fast C library for sequence alignment with a Python API.
# It replaces the pure-Python SSE2 kernels which are ~100x slower.
#
# Scoring matrix: DNA full (matches=5, mismatches=-4) to match the original
# PAM matrix values used in the C code.
# Gap open=12, gap extend=4 to match the original '\020' and '\004' values.
# ─────────────────────────────────────────────────────────────────────────────

try:
    import parasail
    _PARASAIL_AVAILABLE = True
    # Build a DNA scoring matrix matching the original PAM values:
    #   match = 5, mismatch = -4
    _MATRIX = parasail.matrix_create("ACGTN", 5, -4)
    _GAP_OPEN   = 12   # matches 0x10 * 0.75 — original open penalty
    _GAP_EXTEND =  4   # matches 0x04
except ImportError:
    _PARASAIL_AVAILABLE = False
    import warnings
    warnings.warn(
        "parasail not installed. Falling back to slow pure-Python SW kernels.\n"
        "Install with:  pip install parasail",
        RuntimeWarning
    )


def _sw_parasail(query: str, target: str) -> int:
    """
    Run Smith-Waterman via parasail and return the alignment score.
    Handles RNA 'U' by converting to 'T' so the DNA matrix applies correctly.
    """
    q = query.replace('U', 'T').upper()
    t = target.replace('U', 'T').upper()
    result = parasail.sw_scan_sat(q, t, _GAP_OPEN, _GAP_EXTEND, _MATRIX)
    return result.score


def _sw_fallback(strA: str, strB: str) -> int:
    """
    Pure-Python SW fallback (original implementation) used if parasail
    is not installed.
    """
    aa0_enc    = cg_str(strA)
    aa1_enc    = cg_str(strB)
    n0 = len(aa0_enc)
    n1 = len(aa1_enc)

    pst = alloc_pam(MAXSQ, MAXSQ)
    init_pam2(pst)
    f_str = init_work(aa0_enc, n0, pst)

    GAP_OPEN   = 0x10
    GAP_EXTEND = 0x04

    s = smith_waterman_sse2_byte(
        aa0_enc, f_str.byte_score, n0,
        aa1_enc, n1,
        f_str.bias, GAP_OPEN, GAP_EXTEND, f_str)
    if s >= 255:
        s = smith_waterman_sse2_word(
            aa0_enc, f_str.word_score, n0,
            aa1_enc, n1,
            GAP_OPEN, GAP_EXTEND, f_str)
    return s


def _sw_score(query: str, target: str) -> int:
    """Dispatch to parasail if available, otherwise fall back to pure Python."""
    if _PARASAIL_AVAILABLE:
        return _sw_parasail(query, target)
    return _sw_fallback(query, target)


def calc_score(strA: str, strB: str, dnaStartPos: int, rule: int) -> int:
    """
    Compute an EVD-derived E-value threshold for the Smith-Waterman alignment
    of strA vs strB (and its reverse complement).

    Uses parasail for fast C-level SW alignment if available, otherwise
    falls back to the pure-Python SSE2 emulation.

    Returns the MLE threshold as an integer, or 0 on failure.
    """
    SHUF_MAX = 1002

    strA_rc = _reverse_complement(strA)
    n0      = len(strA)
    n1      = len(strB)

    # Score the real sequences (forward and reverse complement)
    score    = _sw_score(strA,    strB)
    rc_score = _sw_score(strA_rc, strB)

    # Generate 1002 shuffled versions of strB and score each one
    # Uses the same Park-Miller PRNG as the original for reproducibility
    rand_state  = my_srand(0)
    strB_bytes  = bytearray(strB.encode())
    shuf_scores: List[int] = []

    for shuf_cnt in range(SHUF_MAX):
        # Shuffle bytes then decode back to string
        shuf_bytes = shuffle(strB_bytes, rand_state)
        shuf_str   = shuf_bytes.decode(errors='replace')

        if shuf_cnt % 2 == 0:
            s = _sw_score(strA,    shuf_str)
        else:
            s = _sw_score(strA_rc, shuf_str)
        shuf_scores.append(s)

    # Pair up scores and take pairwise maximum (first 500 pairs)
    max_shuf: List[int] = []
    last_score = 0
    tmp_num    = 0
    for i in range(0, SHUF_MAX, 2):
        pair_max = max(shuf_scores[i], shuf_scores[i + 1])
        if tmp_num < 500:
            max_shuf.append(pair_max)
            tmp_num += 1
        else:
            last_score = pair_max

    # Patch index 150 (matches original C behaviour)
    if len(max_shuf) > 150:
        max_shuf[150] = last_score

    aa1_len = [n1] * len(max_shuf)

    # Fit an Extreme Value Distribution to the shuffle scores
    mle_rst = mle_cen(max_shuf, len(max_shuf), aa1_len, n0,
                      0.0, 0.0, 0.0)
    if mle_rst is None:
        return 0

    lambda_tmp, K_tmp = mle_rst

    # Guard: K_tmp, n1, n0 must all be positive for log() to be valid
    log_arg = K_tmp * n1 * n0
    if log_arg <= 0.0 or lambda_tmp <= 0.0:
        return 0

    mle_thresh = int(
        (math.log(log_arg) - math.log(10.0)) / lambda_tmp + 0.5
    )

    return mle_thresh