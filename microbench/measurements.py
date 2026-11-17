import math
import numpy as np
from .imaging import InputError, finite


def calibration(value, size):
    unit = str(value.get('unit', 'um'))
    if unit not in ('um', 'mm', 'nm', 'px'):
        raise InputError('单位必须为um/mm/nm/px')
    refs = value.get('references', [])
    if not isinstance(refs, list) or len(refs) > 20:
        raise InputError('最多20条标尺参考')
    lengths = []
    checked = []
    for ref in refs:
        a, b = ref['start'], ref['end']
        points = []
        for p in (a, b):
            points.append([finite(p[0], 0, size[0] - 1, 'x'), finite(p[1], 0, size[1] - 1, 'y')])
        distance = math.dist(*points)
        if distance < 1:
            raise InputError('标尺参考必须至少跨1像素')
        length = finite(ref['length'], 1e-9, 1e12, '物理长度')
        lengths.append((distance, length))
        checked.append({'start': points[0], 'end': points[1], 'length': length})
    if lengths:
        pixel_size = sum(p * length for p, length in lengths) / sum(p * p for p, length in lengths)
    else:
        pixel_size = finite(value.get('pixel_size', 1), 1e-9, 1e9, '每像素长度')
    return {'unit': unit, 'pixel_size': pixel_size, 'references': checked,
            'rms': float(np.sqrt(np.mean([(p * pixel_size - length) ** 2 for p, length in lengths]))) if lengths else 0}


def object_rows(run):
    scale = run['calibration']['pixel_size']
    excluded = set(run['excluded'])
    rows = []
    for obj in run['objects']:
        row = dict(obj)
        row['included'] = obj['id'] not in excluded
        row['area'] = obj['area_px'] * scale ** 2
        row['perimeter'] = obj['perimeter_px'] * scale
        row['equivalent_diameter'] = obj['equivalent_diameter_px'] * scale
        row['major'] = obj['major_px'] * scale
        row['minor'] = obj['minor_px'] * scale
        row['x'] = obj['cx'] * scale
        row['y'] = obj['cy'] * scale
        row['unit'] = run['calibration']['unit']
        rows.append(row)
    return rows


def summary(run):
    rows = [r for r in object_rows(run) if r['included']]
    area = run['diagnostics']['roi_pixels'] * run['calibration']['pixel_size'] ** 2
    sizes = np.array([r['equivalent_diameter'] for r in rows])
    hist, edges = np.histogram(sizes, bins=min(20, max(1, len(rows)))) if len(rows) else ([], [])
    return {'count': len(rows), 'excluded_count': len(run['excluded']),
            'sample_area': area, 'number_density': len(rows) / area if area else None,
            'area_fraction': sum(r['area'] for r in rows) / area if area else None,
            'mean_diameter': float(sizes.mean()) if len(rows) else None,
            'median_diameter': float(np.median(sizes)) if len(rows) else None,
            'unit': run['calibration']['unit'],
            'histogram': {'counts': list(map(int, hist)), 'edges': list(map(float, edges))}}


def annotation(value, size, scale):
    kind = value.get('kind')
    pts = value.get('points', [])
    if kind not in ('distance', 'angle', 'polygon'):
        raise InputError('支持distance/angle/polygon')
    expected = {'distance': 2, 'angle': 3}
    if kind in expected and len(pts) != expected[kind]:
        raise InputError('标注顶点数不匹配')
    if kind == 'polygon' and not 3 <= len(pts) <= 100:
        raise InputError('面积标注需要3–100个点')
    points = [[finite(p[0], 0, size[0] - 1, 'x'), finite(p[1], 0, size[1] - 1, 'y')] for p in pts]
    if kind == 'distance':
        result = math.dist(*points) * scale
    elif kind == 'polygon':
        array = np.array(points)
        result = .5 * abs(np.dot(array[:, 0], np.roll(array[:, 1], 1)) - np.dot(array[:, 1], np.roll(array[:, 0], 1))) * scale ** 2
    else:
        a = np.subtract(points[0], points[1])
        b = np.subtract(points[2], points[1])
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            raise InputError('角度顶点不能重合')
        result = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / norm, -1, 1))))
    return {'kind': kind, 'points': points, 'value': result, 'label': str(value.get('label', ''))[:120]}
