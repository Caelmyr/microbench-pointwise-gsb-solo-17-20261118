import base64
import copy
import time
import uuid
import numpy as np
from .imaging import InputError, segment, label_png, read_labels, overlay, settings, roi_mask, roi_edge, measure_object, finite, intensity, line_profile, quality
from .measurements import calibration, annotation, object_rows, summary
from .splitting import parse_seeds, parse_barrier, split_labels, preview_png


class Bench:
    def __init__(self, store):
        self.store = store

    def update_field(self, doc, fid, value):
        field = self.store.field(doc, fid)
        changed = False
        if 'name' in value:
            field['name'] = str(value['name'])[:120]
        if 'timestamp' in value:
            field['timestamp'] = finite(value['timestamp'], -1e9, 1e9, '观测秒')
        if 'settings' in value:
            field['settings'] = settings(value['settings'])
            changed = True
        if 'roi' in value:
            roi_mask(tuple(field['size']), value['roi'])
            field['roi'] = copy.deepcopy(value['roi'])
            changed = True
        if 'calibration' in value:
            field['calibration'] = calibration(value['calibration'], field['size'])
            changed = True
        if changed:
            field['active_run'] = None
        doc['active_field'] = fid

    def analyze(self, pid, revision, fid):
        with self.store.lock:
            doc = self.store.load(pid)
            if revision != doc['revision']:
                from .storage import RevisionConflict
                raise RevisionConflict('分析输入已变化')
            field = copy.deepcopy(self.store.field(doc, fid))
        image = self.store.image(field['source'])
        labels, objects, diagnostics = segment(image, field['settings'], field['roi'])
        if len(objects) > 2000:
            raise InputError('每幅最多2000对象，请调整最小面积')
        result = {'id': uuid.uuid4().hex, 'field_id': fid, 'name': field['name'], 'source': field['source'],
                  'timestamp': field['timestamp'], 'size': field['size'], 'input_revision': revision,
                  'created': time.time(), 'settings': field['settings'], 'roi': field['roi'],
                  'calibration': field['calibration'], 'objects': objects, 'excluded': [],
                  'diagnostics': diagnostics, 'labels': self.store.put(label_png(labels))}
        def publish(current):
            if len(current['runs']) >= 100:
                raise InputError('每工程最多100份分析历史')
            current['runs'].append(result)
            self.store.field(current, fid)['active_run'] = result['id']
        return self.store.mutate(pid, revision, publish)

    def exclude(self, doc, rid, ids):
        old = self.store.run(doc, rid)
        valid = {obj['id'] for obj in old['objects']}
        ids = sorted(set(int(v) for v in ids))
        if not set(ids) <= valid:
            raise InputError('排除对象不属于该结果')
        ids = sorted(set(ids) | {obj['id'] for obj in old['objects'] if obj.get('pending')})
        if len(doc['runs']) >= 100:
            raise InputError('每工程最多100份分析历史')
        new = copy.deepcopy(old)
        new.update(id=uuid.uuid4().hex, excluded=ids, created=time.time(), parent_run=rid)
        doc['runs'].append(new)
        field = self.store.field(doc, old['field_id'])
        if field['active_run'] == rid:
            field['active_run'] = new['id']

    def _split_layout(self, doc, rid, data):
        run = self.store.run(doc, rid)
        try:
            oid = int(data.get('object_id'))
        except (TypeError, ValueError):
            raise InputError('修整对象不属于该结果')
        if oid not in {obj['id'] for obj in run['objects']}:
            raise InputError('修整对象不属于该结果')
        image = self.store.image(run['source'])
        labels = read_labels(self.store.asset(run['labels']))
        seeds = parse_seeds(data.get('seeds'), image.size)
        barrier = parse_barrier(data.get('barriers'), image.size)
        origin, sub = split_labels(intensity(image), labels, oid, seeds, barrier)
        return run, image, labels, oid, seeds, barrier, origin, sub

    def _checked_doc(self, pid, revision):
        with self.store.lock:
            doc = self.store.load(pid)
            if revision != doc['revision']:
                from .storage import RevisionConflict
                raise RevisionConflict('工程已变化，请重新载入后操作')
            return doc

    def split_preview(self, pid, revision, rid, data):
        run, image, _, oid, seeds, _, origin, sub = self._split_layout(self._checked_doc(pid, revision), rid, data)
        raw = preview_png(image.size, origin, sub, len(seeds))
        return {'object_id': oid, 'pieces': [int((sub == i).sum()) for i in range(1, len(seeds) + 1)],
                'pending_px': int((sub == len(seeds) + 1).sum()),
                'image': 'data:image/png;base64,' + base64.b64encode(raw).decode()}

    def split_confirm(self, pid, revision, rid, data):
        doc = self._checked_doc(pid, revision)
        run, image, labels, oid, seeds, barrier, origin, sub = self._split_layout(doc, rid, data)
        field_img = intensity(image)
        valid = roi_mask(image.size, run['roi']) & (np.asarray(image)[..., 3] > 0)
        edge = roi_edge(valid)
        x0, y0 = origin
        new_labels = labels.copy()
        view = new_labels[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
        next_id = max(obj['id'] for obj in run['objects']) + 1
        kept = copy.deepcopy([obj for obj in run['objects'] if obj['id'] != oid])
        produced, pending_id, additions = [], None, []
        for slot in range(1, len(seeds) + 2):
            local = sub == slot
            if slot == len(seeds) + 1 and not np.any(local):
                continue
            if next_id > 65000:
                raise InputError('对象编号超出标签容量')
            mask = np.zeros(labels.shape, bool)
            mask[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = local
            ys, xs = np.nonzero(mask)
            my0, my1 = int(ys.min()), int(ys.max()) + 1
            mx0, mx1 = int(xs.min()), int(xs.max()) + 1
            piece = mask[my0:my1, mx0:mx1]
            touches = bool(np.any(edge[my0:my1, mx0:mx1] & piece))
            obj = measure_object(field_img, piece, mx0, my0, next_id, touches)
            if slot == len(seeds) + 1:
                obj['pending'] = True
                pending_id = next_id
            else:
                produced.append(next_id)
            view[local] = next_id
            additions.append(obj)
            next_id += 1
        if len(kept) + len(additions) > 2000:
            raise InputError('每幅最多2000对象，请减少拆分数量')
        new = copy.deepcopy(run)
        new['objects'] = kept + additions
        new['excluded'] = sorted((set(run['excluded']) - {oid}) | ({pending_id} if pending_id else set()))
        new['diagnostics'] = dict(run['diagnostics'], foreground_pixels=int((new_labels > 0).sum()))
        new['labels'] = self.store.put(label_png(new_labels))
        new.update(id=uuid.uuid4().hex, created=time.time(), parent_run=rid,
                   split={'parent_run': rid, 'object_id': oid, 'seeds': [list(s) for s in seeds],
                          'barriers': data.get('barriers') or [], 'produced': produced,
                          'pending': pending_id, 'created': time.time()})
        def publish(current):
            if len(current['runs']) >= 100:
                raise InputError('每工程最多100份分析历史')
            current['runs'].append(new)
            field = self.store.field(current, run['field_id'])
            if field['active_run'] == rid:
                field['active_run'] = new['id']
        return self.store.mutate(pid, revision, publish)

    def add_annotation(self, doc, fid, value):
        field = self.store.field(doc, fid)
        if len(field['annotations']) >= 100:
            raise InputError('每幅最多100标注')
        item = annotation(value, field['size'], field['calibration']['pixel_size'])
        item['id'] = uuid.uuid4().hex
        item['calibration'] = copy.deepcopy(field['calibration'])
        field['annotations'].append(item)

    def render(self, doc, rid):
        run = self.store.run(doc, rid)
        return overlay(self.store.image(run['source']), read_labels(self.store.asset(run['labels'])), run['excluded'])

    def comparison(self, doc):
        rows = []
        for field in doc['fields']:
            if field['active_run']:
                run = self.store.run(doc, field['active_run'])
                rows.append({'field_id': field['id'], 'name': field['name'], 'timestamp': field['timestamp'],
                             'run_id': run['id'], **summary(run)})
        return rows

    def profile(self, doc, fid, value):
        field = self.store.field(doc, fid)
        samples = line_profile(self.store.image(field['source']), value['start'], value['end'], value.get('count', 256))
        for sample in samples:
            sample['distance'] = sample['distance_px'] * field['calibration']['pixel_size']
        return {'unit': field['calibration']['unit'], 'samples': samples, 'source': field['source']}

    def detail(self, doc):
        result = copy.deepcopy(doc)
        for run in result['runs']:
            run['summary'] = summary(run)
            run['rows'] = object_rows(run)
        return result
