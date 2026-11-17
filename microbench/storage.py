import copy
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from .imaging import InputError, png, decode, label_png, read_labels


class RevisionConflict(InputError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'assets').mkdir(exist_ok=True)
        (self.root / 'projects').mkdir(exist_ok=True)
        self.lock = threading.RLock()

    def identifier(self, value):
        if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{32}', value):
            raise InputError('无效工程ID')
        return value

    def path(self, pid):
        return self.root / 'projects' / (self.identifier(pid) + '.json')

    def load(self, pid):
        path = self.path(pid)
        if not path.exists():
            raise InputError('工程不存在')
        doc = json.loads(path.read_text())
        doc.setdefault('pairs', [])
        return doc

    def write(self, doc):
        target = self.path(doc['id'])
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False))
        os.replace(temporary, target)

    def create(self, name):
        with self.lock:
            doc = {'id': uuid.uuid4().hex, 'name': str(name or '未命名视野')[:120], 'revision': 1,
                   'created': time.time(), 'fields': [], 'snapshots': [], 'runs': [], 'pairs': [],
                   'active_field': None}
            self.write(doc)
            return doc

    def listing(self):
        with self.lock:
            out = []
            for path in (self.root / 'projects').glob('*.json'):
                doc = json.loads(path.read_text())
                out.append({key: doc[key] for key in ('id', 'name', 'revision', 'created')})
            return sorted(out, key=lambda d: -d['created'])

    def mutate(self, pid, revision, callback):
        with self.lock:
            doc = self.load(pid)
            if revision != doc['revision']:
                raise RevisionConflict('工程已变化，请重新载入后操作')
            callback(doc)
            doc['revision'] += 1
            self.write(doc)
            return copy.deepcopy(doc)

    def put(self, raw):
        key = hashlib.sha256(raw).hexdigest()
        target = self.root / 'assets' / key
        with self.lock:
            if not target.exists():
                temporary = target.with_suffix('.tmp')
                temporary.write_bytes(raw)
                os.replace(temporary, target)
        return key

    def asset(self, key):
        if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{64}', key):
            raise InputError('无效资源引用')
        target = self.root / 'assets' / key
        if not target.exists():
            raise InputError('资源缺失')
        raw = target.read_bytes()
        if hashlib.sha256(raw).hexdigest() != key:
            raise InputError('资源完整性校验失败')
        return raw

    def image(self, key):
        return decode(self.asset(key))

    def field(self, doc, fid):
        for field in doc['fields']:
            if field['id'] == fid:
                return field
        raise InputError('视野不存在')

    def add_image(self, doc, image, name, timestamp=0):
        if len(doc['fields']) >= 32:
            raise InputError('每工程最多32幅图像')
        fid = uuid.uuid4().hex
        doc['fields'].append({'id': fid, 'name': str(name)[:120], 'source': self.put(png(image)),
                              'size': list(image.size), 'timestamp': timestamp,
                              'calibration': {'unit': 'um', 'pixel_size': 1, 'references': []},
                              'roi': [], 'settings': {'mode': 'otsu', 'polarity': 'bright', 'min_area': 12},
                              'annotations': [], 'active_run': None})
        doc['active_field'] = fid
        return fid

    def run(self, doc, rid):
        for run in doc['runs']:
            if run['id'] == rid:
                return run
        raise InputError('分析结果不存在')

    def pair(self, doc, pair_id):
        for pair in doc.get('pairs', []):
            if pair['id'] == pair_id:
                return pair
        raise InputError('双通道对照结果不存在')

    def current_run(self, doc, field):
        if not field['active_run']:
            raise InputError('请先分析当前视野')
        return self.run(doc, field['active_run'])

    def snapshot(self, doc, name):
        if len(doc['snapshots']) >= 30:
            raise InputError('最多30份命名快照')
        doc['snapshots'].append({'id': uuid.uuid4().hex, 'name': str(name)[:120], 'created': time.time(),
                                 'fields': copy.deepcopy(doc['fields']), 'active_field': doc['active_field']})

    def restore(self, doc, sid):
        for item in doc['snapshots']:
            if item['id'] == sid:
                doc['fields'] = copy.deepcopy(item['fields'])
                doc['active_field'] = item['active_field']
                return
        raise InputError('快照不存在')
