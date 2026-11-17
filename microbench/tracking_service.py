"""追踪会话的服务编排：创建、派生详情、人工改接与演示工程。"""
import time
import numpy as np
from PIL import Image, ImageDraw
from .imaging import InputError
from . import tracking as domain


MAX_SESSIONS = 20


def session_summary(session):
    return {'id': session['id'], 'name': session['name'],
            'frames': len(session['frames']),
            'stale_frames': sum(1 for f in session['frames']
                                if f['field_active_run'] != f['run_id']),
            'edge_count': len(session['edges']),
            'confirmed': sum(1 for e in session['edges'] if e['status'] == 'confirmed'),
            'pending': sum(1 for e in session['edges'] if e['status'] == 'pending'),
            'rejected': sum(1 for e in session['edges'] if e['status'] == 'rejected'),
            'created': session['created']}


class TrackingService:
    def __init__(self, store):
        self.store = store

    def create(self, doc, data):
        run_ids = data.get('run_ids') or []
        if not isinstance(run_ids, list) or not run_ids:
            raise InputError('请选择至少2个分析结果')
        run_ids = [str(rid) for rid in run_ids]
        if len(set(run_ids)) != len(run_ids):
            raise InputError('同一结果不能重复选择')
        for rid in run_ids:
            self.store.run(doc, rid)
        if doc.get('trackings') and len(doc['trackings']) >= MAX_SESSIONS:
            raise InputError('每工程最多%d个追踪会话' % MAX_SESSIONS)
        session = domain.build_session(self.store, doc, run_ids, data.get('params'), data.get('name'))
        session['created'] = time.time()
        doc.setdefault('trackings', []).append(session)
        return self.detail(doc, session['id'])

    def detail(self, doc, tid):
        # 重新读取各视野当前 active_run，以标记重新分析导致的过期状态
        session = self.store.tracking(doc, tid)
        for frame in session['frames']:
            field = self.store.field(doc, frame['field_id'])
            frame['field_active_run'] = field['active_run']
        return domain.derive(session)

    def summaries(self, doc):
        return [session_summary(s) for s in doc.get('trackings', [])]

    def edit_link(self, doc, tid, data):
        session = self.store.tracking(doc, tid)
        nodes = domain.node_map(session)
        action = str(data.get('action', ''))
        if action == 'confirm_group':
            pairs = data.get('pairs')
            if not isinstance(pairs, list):
                raise InputError('整组确认需要关联列表')
            domain.confirm_group(session, pairs)
            return self.detail(doc, tid)
        if action not in ('add', 'confirm', 'reject', 'delete'):
            raise InputError('未知关联操作')
        src, dst = str(data.get('src', '')), str(data.get('dst', ''))
        if action == 'add':
            if src not in nodes or dst not in nodes:
                raise InputError('改接端点不属于本会话')
        domain.mutate_edge(session, nodes, action, src, dst)
        return self.detail(doc, tid)

    def rename(self, doc, tid, name):
        session = self.store.tracking(doc, tid)
        session['name'] = str(name or session['name'])[:120]
        return self.detail(doc, tid)


def tracking_demo_images():
    """5 帧合成场景：稳定漂移、贴近再分开、合并团、整帧漏检可接续。"""
    rng = np.random.default_rng(47)
    width, height = 512, 360
    base = [(float(rng.uniform(40, 440)), float(rng.uniform(40, 300)), int(rng.integers(6, 10)))
            for _ in range(8)]
    # 靠近再分开的一对 A/B：第1帧贴近产生竞争候选，第2帧合成一团，第3帧分开
    pair_track = [
        ((150.0, 120.0), (178.0, 120.0)),   # t0 分离，间距28
        ((158.0, 121.0), (170.0, 119.0)),   # t1 贴近，间距约12，互相落入门限
        None,                                # t2 合并为单团
        ((172.0, 121.0), (160.0, 119.0)),   # t3 重新分开
        ((180.0, 120.0), (152.0, 120.0)),   # t4 继续远离
    ]
    # 漏检颗粒 C：第3帧不绘制，第4帧在预期位置重现，可由 gap=1 接续
    gap_particle = [360.0, 250.0]
    frames = []
    for step in range(5):
        image = Image.new('RGB', (width, height), (15, 19, 25))
        draw = ImageDraw.Draw(image)
        for i, (x, y, radius) in enumerate(base):
            x += step * (1.5 + (i % 3))
            y += step * ((i % 4) - 1) * 0.8
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill=(170 + i % 70, 210, 190))
        if pair_track[step] is None:
            draw.ellipse((164 - 13, 120 - 11, 164 + 13, 120 + 11), fill=(210, 225, 205))
        else:
            for x, y in pair_track[step]:
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(200, 225, 210))
        if step != 3:
            gx = gap_particle[0] + step * 5
            gy = gap_particle[1]
            draw.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), fill=(190, 220, 230))
        draw.text((width - 96, height - 16), 't=%d' % step, fill=(60, 66, 72))
        frames.append(image.convert('RGBA'))
    return frames
