#!/usr/bin/env python3
"""Check mode TSV, real-field symmetry, optional VTU projection/MPI agreement."""
import argparse
import cmath
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def check(path, shell=None):
    text = Path(path).read_text()
    rows = [list(map(float, s.split())) for s in text.splitlines()
            if s and not s.startswith('#')]
    nt, np, limit = map(int, re.search(r'n_theta=(\d+) n_phi=(\d+) lmax=(\d+)', text).groups())
    assert len(rows) == (limit + 1)**2
    assert all(len(r) == 8 and all(map(math.isfinite, r)) for r in rows)
    assert [(r[2], r[3]) for r in rows] == [(l, m) for l in range(limit+1) for m in range(-l, l+1)]
    assert all(r[:2] == rows[0][:2] for r in rows)
    modes = {(int(r[2]), int(r[3])): [complex(r[4], r[5]), complex(r[6], r[7])] for r in rows}
    for (l, m), values in modes.items():
        for v, neg in zip(values, modes[l, -m]):
            assert abs(neg - (-1)**m * v.conjugate()) < 1e-11 * max(1, abs(v))
    if shell:
        root = ET.parse(shell).getroot()
        data = {a.attrib['Name']: list(map(float, a.text.split()))
                for a in root.findall('./UnstructuredGrid/Piece/PointData/DataArray')}
        assert len(data['theta']) == 2 + (nt-2)*np
        # Independently form weights from the integrated Chebyshev cosine series.
        n = nt - 1
        def weight(i):
            total = 1.0
            for k in range(2, n+1, 2):
                total += (1 if k == n else 2) * math.cos(k*i*math.pi/n)/(1-k*k)
            return total * (1 if i in (0, n) else 2)/n
        weights = [2*math.pi*weight(0)]
        for i in range(1, n):
            weights.extend([2*math.pi*weight(i)/np]*np)
        weights.append(2*math.pi*weight(n))
        # Closed forms, independent of GSL, test output field wiring and phase.
        basis = {(0, 0): lambda t, p: 1/math.sqrt(4*math.pi),
                 (2, 1): lambda t, p: -math.sqrt(15/(8*math.pi))*math.sin(t)*math.cos(t)*cmath.exp(1j*p)}
        worst = 0
        for key, y in basis.items():
            if key not in modes:
                continue
            for v, field in enumerate(('U_CHI', 'U_PHI')):
                value = sum(w*f*complex(y(t, p)).conjugate() for w, f, t, p in
                            zip(weights, data[field], data['theta'], data['phi']))
                error = abs(value-modes[key][v])
                worst = max(worst, error)
                assert error < 1e-11 * max(1, abs(value))
        print(f'VTU/TSV projection max absolute difference: {worst:.6g}')
    print(f'{path}: PASS ({len(rows)} modes per field)')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modes')
    parser.add_argument('--shell')
    parser.add_argument('--compare')
    args = parser.parse_args()
    rows = check(args.modes, args.shell)
    if args.compare:
        other = check(args.compare)
        assert len(rows) == len(other)
        error = max(abs(a-b) for r, s in zip(rows, other) for a, b in zip(r, s))
        assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
                   for r, s in zip(rows, other) for a, b in zip(r, s))
        print(f'Run agreement: PASS (max absolute difference {error:.6g})')
