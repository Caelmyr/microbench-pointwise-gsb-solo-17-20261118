import copy
import csv
import io
import json
import hashlib
import time
import uuid
import zipfile
from PIL import Image
from .imaging import InputError, png, decode, read_labels
from .measurements import object_rows, summary


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    if rows:
        keys = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    return ('\ufeff' + stream.getvalue()).encode('utf-8')


def output_zip(bench, doc, rid):
    run = bench.store.run(doc, rid)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('overlay.png', png(bench.render(doc, rid)))
        archive.writestr('labels.png', bench.store.asset(run['labels']))
        archive.writestr('source.png', bench.store.asset(run['source']))
        archive.writestr('objects.csv', csv_bytes(object_rows(run)))
        archive.writestr('summary.json', json.dumps(summary(run), ensure_ascii=False))
        archive.writestr('analysis.json', json.dumps(run, ensure_ascii=False))
    return stream.getvalue()


def project_zip(store, doc):
    raw = json.dumps(doc, ensure_ascii=False, allow_nan=False).encode()
    assets = {field['source'] for field in doc['fields']}
    for snapshot in doc['snapshots']:
        assets.update(field['source'] for field in snapshot['fields'])
    for run in doc['runs']:
        assets.update((run['source'], run['labels']))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('project.json', raw)
        archive.writestr('format.json', json.dumps({'format': 'microbench-project', 'version': 1}))
        for key in sorted(assets):
            archive.writestr('assets/' + key, store.asset(key))
    return stream.getvalue()


def import_project(store, raw):
    if len(raw) > 96 * 1024 * 1024:
        raise InputError('工程压缩包不得超过96MB')
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        infos = archive.infolist()
        if len(infos) > 500 or sum(item.file_size for item in infos) > 192 * 1024 * 1024:
            raise InputError('工程展开资源超限')
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise InputError('工程存在重复成员')
        if any(item.flag_bits & 1 for item in infos):
            raise InputError('不支持加密工程')
        format_doc = json.loads(archive.read('format.json'))
        if format_doc != {'format': 'microbench-project', 'version': 1}:
            raise InputError('不支持的工程格式')
        doc = json.loads(archive.read('project.json'))
        if len(doc['fields']) > 32 or len(doc['runs']) > 100 or len(doc['snapshots']) > 30:
            raise InputError('工程对象超限')
        refs = set()
        fields = list(doc['fields'])
        for snapshot in doc['snapshots']:
            fields.extend(snapshot['fields'])
        for field in fields:
            refs.add(field['source'])
        for run in doc['runs']:
            refs.update((run['source'], run['labels']))
        expected = {'project.json', 'format.json'} | {'assets/' + key for key in refs}
        if set(names) != expected:
            raise InputError('工程资源清单不一致')
        resources = {}
        for key in refs:
            data = archive.read('assets/' + key)
            if hashlib.sha256(data).hexdigest() != key:
                raise InputError('工程资源哈希不符')
            resources[key] = data
        for field in fields:
            image = decode(resources[field['source']])
            if list(image.size) != field['size']:
                raise InputError('视野尺寸与源图不一致')
            from .imaging import roi_mask, settings
            from .measurements import calibration
            roi_mask(image.size, field['roi'])
            settings(field['settings'])
            calibration(field['calibration'], image.size)
        run_ids = set()
        for run in doc['runs']:
            if run['id'] in run_ids:
                raise InputError('分析结果ID重复')
            run_ids.add(run['id'])
            image = decode(resources[run['source']])
            labels = read_labels(resources[run['labels']])
            if labels.shape != (image.height, image.width):
                raise InputError('对象标签尺寸不一致')
            ids = {int(obj['id']) for obj in run['objects']}
            if set(int(v) for v in __import__('numpy').unique(labels)) - {0} != ids:
                raise InputError('对象标签与量测行不一致')
            if not set(run['excluded']) <= ids:
                raise InputError('排除引用无效')
        for field in fields:
            if field['active_run'] and field['active_run'] not in run_ids:
                raise InputError('视野引用了缺失结果')
        with store.lock:
            for key, data in resources.items():
                store.put(data)
            doc['id'] = uuid.uuid4().hex
            doc['name'] = str(doc['name'])[:110] + '（导入）'
            doc['revision'] = 1
            doc['created'] = time.time()
            store.write(doc)
        return doc
    except InputError:
        raise
    except Exception as error:
        raise InputError('工程格式或内容无效') from error
