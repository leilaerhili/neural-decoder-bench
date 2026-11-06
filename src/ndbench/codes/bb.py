"""Bivariate bicycle (BB) codes (Bravyi et al., Nature 627, 2024).

A BB code is defined by two polynomials A, B in x, y on an l x m torus, where
x = S_l (x) I_m and y = I_l (x) S_m, and S_k is the k x k cyclic shift. Then

    H_X = [A | B],    H_Z = [B^T | A^T],

with n = 2 l m data qubits. Each torus cell (i, j) holds one L data qubit, one
R data qubit, one X check and one Z check.

Index convention: cell (i, j) <-> flat index i * m + j.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

# A monomial x^a y^b is stored as the exponent pair (a, b).
Monomial = tuple[int, int]

# Qubit types within a cell.
L_DATA, R_DATA, X_CHECK, Z_CHECK = 0, 1, 2, 3
QUBIT_TYPES = ("L", "R", "X", "Z")


def _shift(k: int, power: int) -> np.ndarray:
    """k x k cyclic shift matrix raised to `power`: row r has a 1 at column r + power."""
    return np.roll(np.eye(k, dtype=np.uint8), power % k, axis=1)


def monomial_matrix(mono: Monomial, l: int, m: int) -> np.ndarray:
    a, b = mono
    return np.kron(_shift(l, a), _shift(m, b)).astype(np.uint8)


def gf2_rank(mat: np.ndarray) -> int:
    """Rank over GF(2) by Gaussian elimination."""
    a = (np.array(mat, dtype=np.uint8) & 1).copy()
    rows, cols = a.shape
    rank = 0
    for c in range(cols):
        pivot = np.nonzero(a[rank:, c])[0]
        if pivot.size == 0:
            continue
        p = rank + pivot[0]
        if p != rank:
            a[[rank, p]] = a[[p, rank]]
        hits = np.nonzero(a[:, c])[0]
        hits = hits[hits != rank]
        a[hits] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


@dataclass(frozen=True)
class CnotEdge:
    """One CNOT of the syndrome cycle, as a Tanner-graph edge.

    `slot` is the index 0..11 within the check's cell (see README).
    """

    check_type: int  # X_CHECK or Z_CHECK
    check_cell: tuple[int, int]
    data_type: int  # L_DATA or R_DATA
    data_cell: tuple[int, int]
    term: str  # e.g. "A1", "B3"
    slot: int


@dataclass(frozen=True)
class BBCode:
    l: int
    m: int
    a_terms: tuple[Monomial, Monomial, Monomial]
    b_terms: tuple[Monomial, Monomial, Monomial]
    name: str = field(default="", compare=False)

    # ---- matrices -------------------------------------------------------
    @cached_property
    def A(self) -> np.ndarray:
        return sum(monomial_matrix(t, self.l, self.m) for t in self.a_terms) % 2

    @cached_property
    def B(self) -> np.ndarray:
        return sum(monomial_matrix(t, self.l, self.m) for t in self.b_terms) % 2

    @cached_property
    def hx(self) -> np.ndarray:
        return np.hstack([self.A, self.B]).astype(np.uint8)

    @cached_property
    def hz(self) -> np.ndarray:
        return np.hstack([self.B.T, self.A.T]).astype(np.uint8)

    # ---- parameters -----------------------------------------------------
    @property
    def n_cells(self) -> int:
        return self.l * self.m

    @property
    def n(self) -> int:
        return 2 * self.n_cells

    @cached_property
    def k(self) -> int:
        return self.n - gf2_rank(self.hx) - gf2_rank(self.hz)

    # ---- geometry -------------------------------------------------------
    def cell(self, flat: int) -> tuple[int, int]:
        return divmod(flat, self.m)

    def flat(self, i: int, j: int) -> int:
        return (i % self.l) * self.m + (j % self.m)

    @cached_property
    def cnot_edges(self) -> tuple[CnotEdge, ...]:
        """All 12 * l * m CNOTs, ordered by (check cell, slot).

        X check at cell c, term x^a y^b in A: row c of S^(a,b) has its 1 at
        column c + (a, b), so it touches L data at c + (a, b). Terms of B touch
        R data. Z check rows come from B^T and A^T, so its partners sit at
        c - (a, b).
        """
        edges = []
        for i in range(self.l):
            for j in range(self.m):
                slot = 0
                for ctype, sign, groups in (
                    (X_CHECK, +1, (("A", self.a_terms, L_DATA), ("B", self.b_terms, R_DATA))),
                    (Z_CHECK, -1, (("B", self.b_terms, L_DATA), ("A", self.a_terms, R_DATA))),
                ):
                    for letter, terms, dtype in groups:
                        for t, (a, b) in enumerate(terms, start=1):
                            di, dj = (i + sign * a) % self.l, (j + sign * b) % self.m
                            edges.append(
                                CnotEdge(ctype, (i, j), dtype, (di, dj), f"{letter}{t}", slot)
                            )
                            slot += 1
        return tuple(edges)


# x^3 + y + y^2 and y^3 + x + x^2 (Bravyi et al. 2024, Table 3).
_A = ((3, 0), (0, 1), (0, 2))
_B = ((0, 3), (1, 0), (2, 0))

BB72 = BBCode(6, 6, _A, _B, name="[[72,12,6]]")
GROSS144 = BBCode(12, 6, _A, _B, name="[[144,12,12]]")
