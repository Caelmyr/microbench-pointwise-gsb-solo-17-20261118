import heapq
import cv2
import numpy as np
from PIL import Image
from .imaging import InputError, finite, png


def parse_seeds(value, size):
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise InputError('需要1–64个种子点')
    seeds = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise InputError('种子点格式不正确')
        seeds.append((int(round(finite(point[0], 0, size[0] - 1, 'x'))),
                      int(round(finite(point[1], 0, size[1] - 1, 'y')))))
    if len(set(seeds)) != len(seeds):
        raise InputError('种子点位置重叠')
    return seeds


def parse_barrier(value, size):
    mask = np.zeros((size[1], size[0]), np.uint8)
    if value is None:
        return mask.astype(bool)
    if not isinstance(value, list) or len(value) > 200:
        raise InputError('背景笔划数量超限')
    total = 0
    for stroke in value:
        if not isinstance(stroke, dict):
            raise InputError('笔划格式不正确')
        points = stroke.get('points')
        width = int(finite(stroke.get('width', 9), 1, 64, '笔划宽度'))
        if not isinstance(points, list) or not 1 <= len(points) <= 4096:
            raise InputError('笔划顶点数超出范围')
        total += len(points)
        if total > 20000:
            raise InputError('笔划总顶点数超限')
        xy = np.array([[finite(p[0], 0, size[0] - 1, 'x'), finite(p[1], 0, size[1] - 1, 'y')]
                       for p in points], dtype=np.int32)
        if len(xy) == 1:
            cv2.circle(mask, tuple(xy[0]), max(1, width // 2), 1, -1)
        else:
            cv2.polylines(mask, [xy], False, 1, width)
    return mask.astype(bool)


def flood(cost, region, seeds, limit=2_000_000):
    height, width = region.shape
    cost_l = cost.tolist()
    region_l = region.tolist()
    labels = [[0] * width for _ in range(height)]
    heap = []
    push = heapq.heappush
    pop = heapq.heappop
    for index, (x, y) in enumerate(seeds):
        labels[y][x] = index + 1
        push(heap, (cost_l[y][x], index, index + 1, x, y))
    pushes = len(seeds)
    while heap:
        level, _, label, x, y = pop(heap)
        for ny in (y - 1, y, y + 1):
            if not 0 <= ny < height:
                continue
            lrow, rrow, crow = labels[ny], region_l[ny], cost_l[ny]
            for nx in (x - 1, x, x + 1):
                if not 0 <= nx < width or (nx == x and ny == y):
                    continue
                if rrow[nx] and not lrow[nx]:
                    lrow[nx] = label
                    c = crow[nx]
                    push(heap, (level if level > c else c, pushes, label, nx, ny))
                    pushes += 1
        if pushes > limit:
            raise InputError('修整区域过大，请用背景笔划先隔开')
    return np.array(labels, dtype=np.int32)


def split_labels(field, labels, oid, seeds, barrier):
    target = labels == oid
    if not np.any(target):
        raise InputError('对象不存在于标签图')
    region = target & ~barrier
    for sx, sy in seeds:
        if not region[sy, sx]:
            raise InputError('种子点不在可修整区域内')
    ys, xs = np.nonzero(target)
    y0, y1 = max(0, ys.min() - 2), min(labels.shape[0], ys.max() + 3)
    x0, x1 = max(0, xs.min() - 2), min(labels.shape[1], xs.max() + 3)
    local = region[y0:y1, x0:x1]
    count, comp = cv2.connectedComponents(local.astype(np.uint8), connectivity=8)
    sizes = np.bincount(comp.ravel(), minlength=count)
    by_comp = {}
    for index, (sx, sy) in enumerate(seeds):
        by_comp.setdefault(int(comp[sy - y0, sx - x0]), []).append((index, sx - x0, sy - y0))
    if sum(int(sizes[c]) for c in by_comp) > 2_000_000:
        raise InputError('修整区域过大，请用背景笔划先隔开')
    crop = field[y0:y1, x0:x1]
    grad = np.hypot(cv2.Sobel(crop, cv2.CV_32F, 1, 0), cv2.Sobel(crop, cv2.CV_32F, 0, 1))
    sub = np.zeros(local.shape, np.int32)
    for cid, members in by_comp.items():
        cys, cxs = np.nonzero(comp == cid)
        cy0, cy1 = int(cys.min()), int(cys.max()) + 1
        cx0, cx1 = int(cxs.min()), int(cxs.max()) + 1
        reached = flood(grad[cy0:cy1, cx0:cx1], comp[cy0:cy1, cx0:cx1] == cid,
                        [(x - cx0, y - cy0) for _, x, y in members])
        view = sub[cy0:cy1, cx0:cx1]
        for basin, (index, _, _) in enumerate(members, start=1):
            view[reached == basin] = index + 1
    sub[local & (sub == 0)] = len(seeds) + 1
    sub[target[y0:y1, x0:x1] & barrier[y0:y1, x0:x1]] = len(seeds) + 1
    return (x0, y0), sub


def preview_png(size, origin, sub, count):
    canvas = np.zeros((size[1], size[0], 4), np.uint8)
    x0, y0 = origin
    view = canvas[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
    for index in range(1, count + 1):
        view[sub == index] = [(index * 71) % 170 + 60, (index * 137) % 170 + 60,
                              (index * 193) % 170 + 60, 96]
    pending = sub == count + 1
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    view[pending & ((xx + yy) % 2 == 0)] = [235, 235, 235, 150]
    view[pending & ((xx + yy) % 2 == 1)] = [110, 110, 110, 150]
    return png(Image.fromarray(canvas, 'RGBA'))
