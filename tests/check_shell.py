#!/usr/bin/env python3
"""Validate shell geometry, closed connectivity, fields and optional MPI agreement."""
import argparse
from collections import Counter
import math
import xml.etree.ElementTree as ET


def check(path):
    root = ET.parse(path).getroot()
    piece = root.find('./UnstructuredGrid/Piece')
    n, nc = int(piece.attrib['NumberOfPoints']), int(piece.attrib['NumberOfCells'])
    def arrays(tag):
        return {a.attrib['Name']: [float(x) for x in a.text.split()]
                for a in piece.find(tag)}
    xyz = arrays('Points')['Points']
    data, cells = arrays('PointData'), arrays('Cells')
    assert len(xyz) == 3*n
    assert {'U_CHI', 'U_PHI', 'radius', 'theta', 'phi'} <= data.keys()
    assert all(len(a) == n and all(map(math.isfinite, a)) for a in data.values())
    points = [tuple(xyz[i:i+3]) for i in range(0, len(xyz), 3)]
    assert len(set(points)) == n, 'duplicate points / poles'
    # Infer center from the two unique poles.
    center = tuple((points[0][d] + points[-1][d])/2 for d in range(3))
    for i, p in enumerate(points):
        radius = math.sqrt(sum((p[d]-center[d])**2 for d in range(3)))
        assert math.isclose(radius, data['radius'][i], rel_tol=1e-12)
    conn = list(map(int, cells['connectivity']))
    offsets, types = list(map(int, cells['offsets'])), list(map(int, cells['types']))
    assert len(offsets) == len(types) == nc and offsets[-1] == len(conn)
    edges, used = Counter(), set()
    start = 0
    for end, kind in zip(offsets, types):
        ids = conn[start:end]
        assert len(ids) == {5: 3, 9: 4}[kind] and len(set(ids)) == len(ids)
        assert all(0 <= i < n for i in ids)
        used.update(ids)
        for a, b in zip(ids, ids[1:] + ids[:1]):
            edges[tuple(sorted((a,b)))] += 1
        a, b, c = [points[i] for i in ids[:3]]
        u, v = [b[d]-a[d] for d in range(3)], [c[d]-a[d] for d in range(3)]
        normal = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        assert sum(normal[d]*(a[d]-center[d]) for d in range(3)) > 0, 'degenerate/inward cell'
        start = end
    assert len(used) == n and set(edges.values()) == {2}
    assert n - len(edges) + nc == 2, 'not a closed spherical topology'
    print(f'{path}: PASS ({n} points, {nc} cells, closed outward surface)')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    results = [check(path) for path in args.files]
    if args.compare:
        for other in results[1:]:
            for key, values in results[0].items():
                assert len(values) == len(other[key])
                assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
                           for a, b in zip(values, other[key])), key
        print('Serial/MPI agreement: PASS')
