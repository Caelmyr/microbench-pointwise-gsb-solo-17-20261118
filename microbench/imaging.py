import hashlib
import io
import math
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw


class InputError(ValueError):
    pass


def finite(value, lower, upper, name):
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise InputError(name + '必须为数值')
    if not math.isfinite(value) or not lower <= value <= upper:
        raise InputError(name + '超出允许范围')
    return value


def decode(raw):
    if len(raw) > 20 * 1024 * 1024:
        raise InputError('图像不得超过20MB')
    try:
        image = Image.open(io.BytesIO(raw))
        if image.format not in ('PNG', 'JPEG', 'TIFF', 'WEBP'):
            raise InputError('支持PNG/JPEG/单页TIFF/WebP')
        if image.width * image.height > 12_000_000 or max(image.size) > 4096:
            raise InputError('图像上限1200万像素、4096边长')
        image = ImageOps.exif_transpose(image).convert('RGBA')
        if min(image.size) < 16:
            raise InputError('图像边长至少16')
        return image
    except InputError:
        raise
    except Exception as error:
        raise InputError('无法解码图像') from error


def png(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def intensity(image):
    a = np.asarray(image, dtype=np.float32) / 255
    rgb = a[..., :3] * a[..., 3:4]
    return rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)


def roi_mask(size, points):
    image = Image.new('L', size, 255 if not points else 0)
    if points:
        if len(points) < 3 or len(points) > 100:
            raise InputError('ROI多边形需要3–100点')
        vertices = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise InputError('ROI顶点格式不正确')
            vertices.append((finite(point[0], 0, size[0], 'x'), finite(point[1], 0, size[1], 'y')))
        ImageDraw.Draw(image).polygon(vertices, fill=255)
    return np.asarray(image) > 0


def settings(value):
    polarity = value.get('polarity', 'bright')
    mode = value.get('mode', 'otsu')
    if polarity not in ('bright', 'dark') or mode not in ('otsu', 'manual', 'adaptive'):
        raise InputError('未知分割方式')
    return {'mode': mode, 'polarity': polarity,
            'threshold': finite(value.get('threshold', .4), 0, 1, '阈值'),
            'blur': int(finite(value.get('blur', 1), 0, 12, '平滑半径')),
            'opening': int(finite(value.get('opening', 0), 0, 8, '开运算半径')),
            'closing': int(finite(value.get('closing', 0), 0, 8, '闭运算半径')),
            'min_area': int(finite(value.get('min_area', 12), 1, 1000000, '最小面积')),
            'max_area': int(finite(value.get('max_area', 1000000), 1, 12000000, '最大面积')),
            'exclude_border': bool(value.get('exclude_border', True)),
            'fill_holes': bool(value.get('fill_holes', True))}


def roi_edge(valid):
    boundary = valid & ~cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    boundary[[0, -1], :] |= valid[[0, -1], :]
    boundary[:, [0, -1]] |= valid[:, [0, -1]]
    return boundary


def measure_object(field, local, x, y, oid, touches):
    contours, _ = cv2.findContours(local.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    ys, xs = np.nonzero(local)
    area = int(local.sum())
    cov = np.cov(np.vstack([xs, ys]), bias=True) if len(xs) > 1 else np.zeros((2, 2))
    eig = np.maximum(np.linalg.eigvalsh(cov), 0)
    major, minor = (4 * np.sqrt(eig[::-1])).tolist()
    values = field[y:y + local.shape[0], x:x + local.shape[1]][local]
    return {'id': oid, 'area_px': area, 'cx': float(xs.mean() + x), 'cy': float(ys.mean() + y),
            'bbox': [x, y, int(local.shape[1]), int(local.shape[0])], 'perimeter_px': float(perimeter),
            'equivalent_diameter_px': float(np.sqrt(4 * area / np.pi)),
            'major_px': major, 'minor_px': minor,
            'circularity': float(min(1, 4 * np.pi * area / perimeter ** 2)) if perimeter else 0,
            'mean_intensity': float(values.mean()), 'std_intensity': float(values.std()),
            'integrated_intensity': float(values.sum()), 'touches_boundary': touches}


def segment(image, configuration, polygon):
    config = settings(configuration)
    field = intensity(image)
    valid = roi_mask(image.size, polygon) & (np.asarray(image)[..., 3] > 0)
    u8 = np.rint(field * 255).astype(np.uint8)
    if config['blur']:
        radius = config['blur']
        u8 = cv2.GaussianBlur(u8, (2 * radius + 1, 2 * radius + 1), 0)
    if config['polarity'] == 'dark':
        u8 = 255 - u8
    if config['mode'] == 'adaptive':
        bw = cv2.adaptiveThreshold(u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -4)
    elif config['mode'] == 'otsu':
        selected = u8[valid]
        if not selected.size:
            raise InputError('ROI没有可见像素')
        limit, _ = cv2.threshold(selected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = (u8 > limit).astype(np.uint8) * 255
    else:
        bw = (u8 >= config['threshold'] * 255).astype(np.uint8) * 255
    bw[~valid] = 0
    for option, operation in [('opening', cv2.MORPH_OPEN), ('closing', cv2.MORPH_CLOSE)]:
        radius = config[option]
        if radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            bw = cv2.morphologyEx(bw, operation, kernel)
            bw[~valid] = 0
    if config['fill_holes']:
        count, labels, stats, _ = cv2.connectedComponentsWithStats((bw == 0).astype(np.uint8), connectivity=4)
        border = set(np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1], labels[~valid]]).tolist())
        for label in range(1, count):
            if label not in border:
                bw[labels == label] = 255
    count, labels, stats, centers = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), connectivity=8)
    objects = []
    accepted = np.zeros(labels.shape, dtype=np.int32)
    boundary = roi_edge(valid)
    for label in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[label]]
        if not config['min_area'] <= area <= config['max_area']:
            continue
        mask = labels[y:y + height, x:x + width] == label
        touches = bool(np.any(boundary[y:y + height, x:x + width] & mask))
        if config['exclude_border'] and touches:
            continue
        oid = len(objects) + 1
        accepted[y:y + height, x:x + width][mask] = oid
        objects.append(measure_object(field, mask, x, y, oid, touches))
    return accepted, objects, {'roi_pixels': int(valid.sum()), 'foreground_pixels': int((accepted > 0).sum()),
                               'threshold_mode': config['mode'], 'configuration': config}


def label_png(labels):
    return png(Image.fromarray(labels.astype(np.uint16)))


def read_labels(raw):
    return np.asarray(Image.open(io.BytesIO(raw))).astype(np.int32)


def overlay(image, labels, excluded=()):
    canvas = image.convert('RGB')
    data = np.array(canvas)
    for label in np.unique(labels):
        if not label:
            continue
        mask = labels == label
        color = np.array([(label * 71) % 170 + 60, (label * 137) % 170 + 60, (label * 193) % 170 + 60])
        if int(label) in excluded:
            color = np.array([180, 180, 180])
        boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        data[boundary] = color
    canvas = Image.fromarray(data)
    draw = ImageDraw.Draw(canvas)
    for label in np.unique(labels):
        if label:
            y, x = np.argwhere(labels == label).mean(axis=0)
            draw.text((int(x), int(y)), str(label), fill='white', stroke_width=1, stroke_fill='black')
    return canvas


def line_profile(image, start, end, count):
    count = int(finite(count, 2, 2048, '采样点数'))
    points = []
    for endpoint in (start, end):
        if len(endpoint) != 2:
            raise InputError('剖面端点需有x/y')
        points.append([finite(endpoint[0], 0, image.width - 1, 'x'), finite(endpoint[1], 0, image.height - 1, 'y')])
    x = np.linspace(points[0][0], points[1][0], count).astype(np.float32)
    y = np.linspace(points[0][1], points[1][1], count).astype(np.float32)
    samples = cv2.remap(intensity(image), x[None, :], y[None, :], cv2.INTER_LINEAR)[0]
    distance = np.hypot(x - x[0], y - y[0])
    return [{'distance_px': float(d), 'x': float(a), 'y': float(b), 'intensity': float(v)}
            for d, a, b, v in zip(distance, x, y, samples)]


def quality(image):
    field = intensity(image)
    return {'width': image.width, 'height': image.height,
            'visible_pixels': int((np.asarray(image)[..., 3] > 0).sum()),
            'mean': float(field.mean()), 'std': float(field.std()),
            'black_fraction': float((field <= 1 / 255).mean()),
            'white_fraction': float((field >= 254 / 255).mean()),
            'laplacian_variance': float(cv2.Laplacian(field, cv2.CV_32F).var())}


def demo_images():
    images = []
    rng = np.random.default_rng(29)
    particles = [(float(rng.uniform(32, 480)), float(rng.uniform(32, 330)), int(rng.integers(5, 13))) for _ in range(22)]
    for frame in range(4):
        image = Image.new('RGB', (512, 360), (15, 19, 25))
        draw = ImageDraw.Draw(image)
        for i, (x, y, radius) in enumerate(particles):
            x = x + frame * (2 + (i % 3))
            y = y + frame * ((i % 4) - 1)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(170 + i % 70, 210, 190))
        draw.text((8, 8), 'SYNTHETIC MICROBENCH ' + str(frame), fill=(220, 220, 220))
        images.append(image.convert('RGBA'))
    return images
