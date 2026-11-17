import copy
import time
import uuid
import cv2
import numpy as np
from scipy import signal
from PIL import Image, ImageDraw
from .imaging import InputError, finite, roi_mask, intensity

SCATTER_BINS = 64
PREVIEW_LIMIT = 1200
MIN_ROI_PIXELS = 25
MIN_POSITIVE_PIXELS = 20
MIN_CANDIDATE_SHIFTS = 12
MIN_CONTROL_PIXELS = 100
WORK_BUDGET = 150_000_000


def _roi_points(size, value):
    points = value.get('roi', [])
    if not isinstance(points, list) or not points:
        return []
    if not isinstance(points[0], (list, tuple)):
        raise InputError('共同ROI必须为点列表')
    roi_mask(size, points)
    return copy.deepcopy(points)


def specification(value, field_a, field_b):
    if field_a['id'] == field_b['id']:
        raise InputError('请选择两个不同视野')
    if list(field_a['size']) != list(field_b['size']):
        raise InputError('两幅视野尺寸必须相同（本工具不做配准或缩放）')
    if not value.get('aligned'):
        raise InputError('请确认两幅图已在同一位置对齐；本工具不做配准')
    thresholds = value.get('thresholds') or {}
    control = value.get('control') or {}
    minimum = int(finite(control.get('min_shift', 8), 1, 96, '最小错位'))
    maximum = int(finite(control.get('max_shift', 32), 1, 96, '最大错位'))
    if maximum < minimum:
        raise InputError('最大错位必须不小于最小错位')
    trials = int(finite(control.get('trials', 199), 19, 999, '对照次数'))
    seed = int(finite(control.get('seed', 20261117), 0, 2 ** 31 - 1, '随机种子'))
    return {'aligned': True, 'roi': _roi_points(tuple(field_a['size']), value),
            'thresholds': {'a': finite(thresholds.get('a', .4), 0, 1, 'A通道阳性阈值'),
                           'b': finite(thresholds.get('b', .4), 0, 1, 'B通道阳性阈值')},
            'control': {'min_shift': minimum, 'max_shift': maximum,
                        'trials': trials, 'seed': seed}}


def _sum(array):
    return float(np.sum(array, dtype=np.float64))


def _correlation(a, b):
    if len(a) < 3:
        return None
    left = a.astype(np.float64)
    right = b.astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(_sum(left * left) * _sum(right * right)))
    return _sum(left * right) / denominator if denominator > 0 else None


def _metrics(a, b, threshold_a, threshold_b):
    """Per-pixel co-localization metrics used for the observation and every trial."""
    positive_a = a >= threshold_a
    positive_b = b >= threshold_b
    in_a = _sum(np.where(positive_a, a, 0))
    in_b = _sum(np.where(positive_b, b, 0))
    if in_a <= 0 or in_b <= 0:
        return None
    overlap = positive_a & positive_b
    count_a = int(np.count_nonzero(positive_a))
    count_b = int(np.count_nonzero(positive_b))
    return {'m_ab': _sum(np.where(overlap, a, 0)) / in_a,
            'm_ba': _sum(np.where(overlap, b, 0)) / in_b,
            'pearson': _correlation(a, b),
            'positive_a': count_a, 'positive_b': count_b, 'overlap': int(np.count_nonzero(overlap))}


def _spots(mask):
    count, _, stats_array, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats_array[1:, cv2.CC_STAT_AREA]
    return {'count': int(len(areas)),
            'median_equivalent_diameter_px': float(np.median(np.sqrt(4 * areas / np.pi))) if len(areas) else None,
            'max_area_px': int(areas.max()) if len(areas) else 0}


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def _candidates(roi, visible, minimum, maximum):
    height, width = roi.shape
    min_x, max_x, min_y, max_y = _bbox(roi)
    dx_low, dx_high = max(-min_x, -maximum), min(width - 1 - max_x, maximum)
    dy_low, dy_high = max(-min_y, -maximum), min(height - 1 - max_y, maximum)
    if dx_low > dx_high or dy_low > dy_high:
        return []
    blocked = ~visible
    overlap_count = None
    if blocked.any():
        overlap_count = signal.fftconvolve(blocked.astype(np.float32), roi[::-1, ::-1].astype(np.float32), mode='full')
    shifts = []
    floor = minimum * minimum
    ceiling = maximum * maximum
    for dy in range(dy_low, dy_high + 1):
        for dx in range(dx_low, dx_high + 1):
            distance = dx * dx + dy * dy
            if distance < floor or distance > ceiling:
                continue
            if overlap_count is not None and overlap_count[height - 1 + dy, width - 1 + dx] > .5:
                continue
            shifts.append([dx, dy])
    return shifts


def _distribution(values, observed):
    if observed is None:
        return None
    clean = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(clean):
        return None
    low = float(min(clean.min(), observed))
    high = float(max(clean.max(), observed))
    if high - low < 1e-9:
        low, high = low - .5, high + .5
    counts, edges = np.histogram(clean, bins=20, range=(low, high))
    return {'observed': float(observed), 'trials': int(len(clean)),
            'mean': float(clean.mean()), 'sd': float(clean.std()),
            'min': float(clean.min()), 'max': float(clean.max()),
            'percentile': float((clean < observed).sum() / len(clean) * 100),
            'p_high': float((1 + (clean >= observed).sum()) / (1 + len(clean))),
            'histogram': {'counts': [int(v) for v in counts], 'edges': [float(v) for v in edges]}}


def _control(ia, ib, visible_b, visible_a, roi, thresholds, config, observed, polygon_pixels):
    participation = int(roi.sum())
    base = {'configured': config, 'available': False, 'reason': None, 'candidates': 0,
            'polygon_pixels': int(polygon_pixels), 'participation_pixels': participation,
            'participation_fraction': participation / polygon_pixels if polygon_pixels else 0.0}
    if roi.sum() < MIN_CONTROL_PIXELS:
        base['reason'] = '共同ROI仅%d个像素（少于%d），错位对照不可用。' % (roi.sum(), MIN_CONTROL_PIXELS)
        return base
    if observed is None or observed['positive_a'] < MIN_POSITIVE_PIXELS or observed['positive_b'] < MIN_POSITIVE_PIXELS:
        base['reason'] = '至少一个通道的阳性像素少于%d，强度比例不稳定，错位对照不可用。' % MIN_POSITIVE_PIXELS
        return base
    work = int(roi.sum()) * config['trials']
    if work > WORK_BUDGET:
        base['reason'] = ('共同ROI %d像素 × %d次对照超出计算预算%d像素次，错位对照不可用；'
                          '可缩小共同ROI或减少对照次数。' % (roi.sum(), config['trials'], WORK_BUDGET))
        return base
    shifts = _candidates(roi, visible_a & visible_b, config['min_shift'], config['max_shift'])
    base['candidates'] = len(shifts)
    if not shifts:
        base['reason'] = ('在±%d像素内没有能让共同ROI完整留在两幅图像可见范围内的位移；'
                          '为避免越界和环绕，错位对照不可用。' % config['max_shift'])
        return base
    if len(shifts) < MIN_CANDIDATE_SHIFTS:
        base['reason'] = '可用不越界位移仅%d个（少于%d），位移分布过窄，错位对照不可用。' % (len(shifts), MIN_CANDIDATE_SHIFTS)
        return base
    rng = np.random.default_rng(config['seed'])
    trials = config['trials']
    picked = rng.choice(len(shifts), size=trials, replace=len(shifts) < trials)
    ys, xs = np.nonzero(roi)
    a = ia[ys, xs].astype(np.float32)
    values = {'m_ab': [], 'm_ba': [], 'pearson': []}
    used = []
    for index in picked:
        dx, dy = shifts[int(index)]
        b = ib[ys + dy, xs + dx].astype(np.float32)
        trial = _metrics(a, b, thresholds['a'], thresholds['b'])
        if trial is None or trial['positive_a'] < MIN_POSITIVE_PIXELS or trial['positive_b'] < MIN_POSITIVE_PIXELS:
            used.append([dx, dy, 0])
            for key in values:
                values[key].append(None)
            continue
        used.append([dx, dy, 1])
        for key in values:
            values[key].append(trial[key])
    valid = sum(1 for row in used if row[2])
    base.update(trials_requested=trials, trials_valid=valid, invalid_trials=trials - valid,
                distinct_shifts=len({tuple(row[:2]) for row in used}), shifts=used, values=values)
    required = max(MIN_CANDIDATE_SHIFTS, int(trials * .8))
    if valid < required:
        base['reason'] = ('%d次错位中仅%d次两通道仍有足够阳性像素（至少需要%d次），'
                          '共同ROI与阳性信号不足，错位对照不可用。' % (trials, valid, required))
        return base
    base['available'] = True
    base['metrics'] = {key: _distribution(values[key], observed[key]) for key in values}
    base['metrics'] = {key: value for key, value in base['metrics'].items() if value is not None}
    base['tested_for'] = 'A→B强度比例、B→A强度比例与Pearson r各自单侧高于对照分布（p_high）；不对多个指标做多重比较校正。'
    return base


def _overlay(ia, ib, roi):
    height, width = roi.shape
    green = np.clip(ia, 0, 1) * 255
    magenta = np.clip(ib, 0, 1) * 255
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[..., 0] = magenta
    data[..., 1] = green
    data[..., 2] = magenta
    data[~roi] = (data[~roi].astype(np.float32) * .32).astype(np.uint8)
    boundary = roi & ~cv2.erode(roi.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    data[boundary] = (255, 214, 70)
    return Image.fromarray(data, 'RGB')


def display(image, limit=PREVIEW_LIMIT):
    """Downscale a preview render for transport; saved renders keep source pixels."""
    largest = max(image.size)
    if largest <= limit:
        return image
    scale = limit / largest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


def _participation(polygon, roi):
    height, width = roi.shape
    data = np.zeros((height, width, 3), dtype=np.uint8)
    data[polygon & ~roi] = (58, 64, 68)
    data[roi] = (55, 140, 120)
    boundary = roi & ~cv2.erode(roi.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    data[boundary] = (255, 214, 70)
    return Image.fromarray(data, 'RGB')


def analyze(image_a, image_b, spec):
    if image_a.size != image_b.size:
        raise InputError('两幅视野尺寸必须相同')
    ia = intensity(image_a)
    ib = intensity(image_b)
    visible_a = np.asarray(image_a)[..., 3] > 0
    visible_b = np.asarray(image_b)[..., 3] > 0
    polygon = roi_mask(image_a.size, spec['roi'])
    roi = polygon & visible_a & visible_b
    pixels = int(roi.sum())
    if pixels < MIN_ROI_PIXELS:
        raise InputError('共同ROI可见像素仅%d个（至少%d）' % (pixels, MIN_ROI_PIXELS))
    ys, xs = np.nonzero(roi)
    a = ia[ys, xs].astype(np.float32)
    b = ib[ys, xs].astype(np.float32)
    observed = _metrics(a, b, spec['thresholds']['a'], spec['thresholds']['b'])
    counts, _, _ = np.histogram2d(a, b, bins=SCATTER_BINS, range=[[0, 1], [0, 1]])
    positive_a = a >= spec['thresholds']['a']
    positive_b = b >= spec['thresholds']['b']
    signal_pixels = int((positive_a | positive_b).sum())
    signal_pearson = _correlation(a[positive_a | positive_b], b[positive_a | positive_b]) if signal_pixels >= MIN_POSITIVE_PIXELS else None
    warnings = []
    if signal_pixels < MIN_POSITIVE_PIXELS:
        warnings.append('任一通道阳性像素仅%d个，正信号内相关无法稳定计算。' % signal_pixels)
    if observed is None:
        warnings.append('至少一个通道在当前阈值下没有阳性强度，强度比例无法计算，请调整阳性阈值。')
    elif min(observed['positive_a'], observed['positive_b']) < MIN_POSITIVE_PIXELS:
        warnings.append('至少一个通道的阳性像素少于%d，强度比例的数值稳定性有限。' % MIN_POSITIVE_PIXELS)
    for name, mask in (('A', positive_a), ('B', positive_b)):
        if mask.mean() > .9:
            warnings.append('%s通道阳性像素占ROI的%.1f%%，阈值可能过低，共定位比例解释需谨慎。' % (name, mask.mean() * 100))
    control = _control(ia, ib, visible_b, visible_a, roi, spec['thresholds'], spec['control'], observed, int(polygon.sum()))
    if control['polygon_pixels'] > pixels:
        warnings.append('共同ROI多边形内有%d像素在至少一幅图中透明，已排除，不参与对照；见“实际参与对照的区域”。'
                        % (control['polygon_pixels'] - pixels))
    if control['available'] and control['candidates'] < spec['control']['trials']:
        warnings.append('可用不越界位移%d个少于对照次数%d，随机位移有重复。'
                        % (control['candidates'], spec['control']['trials']))
    metrics = {'roi_pixels': pixels, 'polygon_pixels': int(polygon.sum()),
               'roi_components': int(cv2.connectedComponents(roi.astype(np.uint8), connectivity=8)[0] - 1),
               'positive_a_pixels': int(positive_a.sum()), 'positive_b_pixels': int(positive_b.sum()),
               'overlap_pixels': int((positive_a & positive_b).sum()),
               'pixel_fraction_a': float(observed['overlap'] / observed['positive_a']) if observed and observed['positive_a'] else None,
               'pixel_fraction_b': float(observed['overlap'] / observed['positive_b']) if observed and observed['positive_b'] else None,
               'intensity_fraction_a_in_b': observed['m_ab'] if observed else None,
               'intensity_fraction_b_in_a': observed['m_ba'] if observed else None,
               'pearson_r': observed['pearson'] if observed else None,
               'pearson_positive_r': signal_pearson, 'signal_pixels': signal_pixels}
    summary = {'intensity_fraction_a_in_b': metrics['intensity_fraction_a_in_b'],
               'intensity_fraction_b_in_a': metrics['intensity_fraction_b_in_a'],
               'pearson_r': metrics['pearson_r'], 'control_available': control['available']}
    spots = {'a': _spots((ia >= spec['thresholds']['a']) & roi),
             'b': _spots((ib >= spec['thresholds']['b']) & roi)}
    return {'thresholds': spec['thresholds'], 'roi': spec['roi'], 'roi_full_frame': not spec['roi'],
            'aligned': True, 'metrics': metrics, 'spots': spots,
            'scatter': {'bins': SCATTER_BINS, 'range': [0, 1], 'max_count': int(counts.max()) if counts.size else 0,
                        'counts': counts.astype(np.int32).tolist()},
            'control': control, 'warnings': warnings, 'summary': summary,
            'overlay': _overlay(ia, ib, roi), 'participation': _participation(polygon, roi)}


def pair_record(spec, result, field_a, field_b):
    control = {key: value for key, value in result['control'].items() if key != 'values'}
    return {'id': uuid.uuid4().hex, 'created': time.time(),
            'field_a': {'id': field_a['id'], 'name': field_a['name'], 'timestamp': field_a['timestamp']},
            'field_b': {'id': field_b['id'], 'name': field_b['name'], 'timestamp': field_b['timestamp']},
            'size': list(field_a['size']), 'aligned': result['aligned'],
            'roi': result['roi'], 'roi_full_frame': result['roi_full_frame'],
            'thresholds': result['thresholds'], 'control_configured': spec['control'],
            'metrics': result['metrics'], 'spots': result['spots'], 'scatter': result['scatter'],
            'control': control, 'warnings': result['warnings'], 'summary': result['summary'],
            'sources': {'a': field_a['source'], 'b': field_b['source']}}


def demo_channels():
    rng = np.random.default_rng(73)
    height, width = 360, 512
    polygon = [[48, 250], [70, 60], [170, 45], [230, 120], [190, 200], [280, 250],
               [430, 180], [468, 300], [300, 330], [140, 300]]
    sites = [(float(rng.uniform(55, 460)), float(rng.uniform(55, 320)), float(rng.uniform(.65, 1.0)))
             for _ in range(48)]
    shared = [site for index, site in enumerate(sites) if index % 3 != 2]
    extra = [(float(rng.uniform(55, 460)), float(rng.uniform(55, 320)), float(rng.uniform(.45, .9)))
             for _ in range(18)]

    def render(points):
        impulses = np.zeros((height, width), dtype=np.float32)
        for x, y, strength in points:
            disk = Image.new('L', (7, 7), 0)
            ImageDraw.Draw(disk).ellipse((1, 1, 5, 5), fill=int(round(strength * 255)))
            left, top = int(round(x)) - 3, int(round(y)) - 3
            patch = np.asarray(disk, dtype=np.float32) / 255
            low_x, high_x = max(0, left), min(width, left + 7)
            low_y, high_y = max(0, top), min(height, top + 7)
            if low_x < high_x and low_y < high_y:
                view = impulses[low_y:high_y, low_x:high_x]
                np.maximum(view, patch[low_y - top:high_y - top, low_x - left:high_x - left], out=view)
        blurred = cv2.GaussianBlur(impulses, (0, 0), 1.1)
        background = np.clip(rng.normal(.04, .012, (height, width)).astype(np.float32), 0, 1)
        field = np.clip(blurred + background, 0, 1)
        return Image.fromarray(np.rint(field * 255).astype(np.uint8)).convert('RGBA')

    return [render(sites), render(shared + extra)], polygon
