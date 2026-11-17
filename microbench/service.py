import copy
import time
import uuid
from .imaging import InputError, segment, label_png, read_labels, overlay, settings, roi_mask, finite, line_profile, quality, png
from .measurements import calibration, annotation, object_rows, summary
from . import pairing


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
        if len(doc['runs']) >= 100:
            raise InputError('每工程最多100份分析历史')
        new = copy.deepcopy(old)
        new.update(id=uuid.uuid4().hex, excluded=ids, created=time.time(), parent_run=rid)
        doc['runs'].append(new)
        field = self.store.field(doc, old['field_id'])
        if field['active_run'] == rid:
            field['active_run'] = new['id']

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

    def preview_pair(self, doc, value):
        spec, result, _ = self._evaluate_pair(doc, value)
        return spec, result

    def _evaluate_pair(self, doc, value):
        fields = value.get('fields') if isinstance(value.get('fields'), dict) else {}
        field_a = self.store.field(doc, fields.get('a'))
        field_b = self.store.field(doc, fields.get('b'))
        spec = pairing.specification(value, field_a, field_b)
        result = pairing.analyze(self.store.image(field_a['source']), self.store.image(field_b['source']), spec)
        return spec, result, (field_a, field_b)

    def save_pair(self, doc, value):
        if len(doc.setdefault('pairs', [])) >= 100:
            raise InputError('每工程最多100份双通道对照结果')
        spec, result, (field_a, field_b) = self._evaluate_pair(doc, value)
        record = pairing.pair_record(spec, result, field_a, field_b)
        record['thumbs'] = {'overlay': self.store.put(png(result['overlay'])),
                            'participation': self.store.put(png(result['participation']))}
        doc['pairs'].append(record)
        return record

    def pair(self, doc, pair_id):
        return copy.deepcopy(self.store.pair(doc, pair_id))

    def pair_image(self, doc, pair_id, name):
        pair = self.store.pair(doc, pair_id)
        key = pair['thumbs'].get(name)
        if not key:
            raise InputError('对照图像不存在')
        return self.store.asset(key)

    def detail(self, doc):
        result = copy.deepcopy(doc)
        for run in result['runs']:
            run['summary'] = summary(run)
            run['rows'] = object_rows(run)
        result['pairs'] = [{'id': pair['id'], 'created': pair['created'],
                            'field_a': pair['field_a'], 'field_b': pair['field_b'],
                            'roi': pair['roi'], 'aligned': pair['aligned'], 'summary': pair['summary'],
                            'thresholds': pair['thresholds'], 'control_configured': pair['control_configured']}
                           for pair in result.get('pairs', [])]
        return result
