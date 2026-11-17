"""追踪叠加图渲染：在源图上绘制对象轮廓、轨迹连线、待核对候选与事件标记。

已确认边按轨迹色实线绘制；自动候选为橙色虚线；人工驳回为灰虚线；合并/分裂节点
加描边圆环，漏检接续边标注 GAP。渲染只反映派生结果，不改变会话数据。
"""
import hashlib
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw

from .imaging import read_labels

PALETTE = [(74, 222, 128), (96, 165, 250), (251, 191, 36), (244, 114, 182),
           (52, 211, 153), (167, 139, 250), (251, 146, 60), (45, 212, 191),
           (248, 113, 113), (163, 230, 53), (125, 211, 252), (232, 121, 249)]


def track_color(name):
    digest = hashlib.sha256((name or '').encode()).digest()
    return PALETTE[digest[0] % len(PALETTE)]


def _dashed(draw, start, end, fill, width=2, dash=7, gap=5):
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    travelled = 0
    while travelled < length:
        piece = min(dash, length - travelled)
        draw.line([(x1 + ux * travelled, y1 + uy * travelled),
                   (x1 + ux * (travelled + piece), y1 + uy * (travelled + piece))],
                  fill=fill, width=width)
        travelled += dash + gap


def render_tracking_frame(store, derived, frame_index, selected=None, show_rejected=False):
    frame = derived['frames'][frame_index]
    image = store.image(frame['source']).convert('RGB')
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    nodes = {n['node_id']: n for n in frame['nodes']}
    all_nodes = {n['node_id']: n for f in derived['frames'] for n in f['nodes']}

    # 对象轮廓淡描边，保留原图细节
    label_image = read_labels(store.asset(frame['labels']))
    live_oids = {n['oid'] for n in frame['nodes']}
    outline = np.array(canvas)
    for oid in live_oids:
        mask = label_image == oid
        boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        outline[boundary] = (255, 255, 255)
    canvas = Image.fromarray(outline)
    draw = ImageDraw.Draw(canvas)

    # 关联边：仅在端点所在帧绘制，避免无观测的中间帧出现伪造轨迹
    for edge in derived['edges']:
        if edge['src_frame'] != frame_index and edge['dst_frame'] != frame_index:
            continue
        if edge['status'] == 'rejected' and not show_rejected:
            continue
        src, dst = all_nodes[edge['src']], all_nodes[edge['dst']]
        a = (src['cx_px'], src['cy_px'])
        b = (dst['cx_px'], dst['cy_px'])
        track = next((n['track'] for n in frame['nodes']
                      if n['node_id'] in (edge['src'], edge['dst']) and n['track']), None)
        if edge['status'] == 'confirmed':
            color = track_color(track or 'T0')
            width = 3 if edge['origin'] == 'manual' else 2
            draw.line([a, b], fill=color, width=width)
            if edge['frame_gap'] > 1:
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                draw.text(mid, 'GAP%d' % (edge['frame_gap'] - 1), fill=(255, 235, 130),
                          stroke_width=1, stroke_fill='black')
        elif edge['status'] == 'pending':
            _dashed(draw, a, b, (255, 170, 40), width=2)
        else:
            _dashed(draw, a, b, (150, 150, 150), width=1)

    # 节点圆点、轨迹名与事件圆环
    for node in frame['nodes']:
        x, y = node['cx_px'], node['cy_px']
        color = track_color(node.get('track') or 'T0')
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color,
                     outline=(10, 20, 25), width=1)
        if node.get('role'):
            draw.ellipse((x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4),
                         outline=(255, 90, 90), width=2)
        label = node.get('track') or ('#' + str(node['oid']))
        draw.text((x + 7, y - 6), label, fill='white', stroke_width=1, stroke_fill='black')
        if node['node_id'] == selected:
            draw.ellipse((x - radius - 7, y - radius - 7, x + radius + 7, y + radius + 7),
                         outline=(255, 255, 255), width=2)

    draw.rectangle((0, 0, 230, 20), fill=(12, 24, 30))
    draw.text((6, 4), 'f%d t=%ss %s' % (frame_index, frame['timestamp'], frame['unit']),
              fill=(225, 235, 230))
    return canvas
