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
    for session in doc.get('trackings', []):
        for frame in session['frames']:
            assets.update((frame['source'], frame['labels']))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('project.json', raw)
        archive.writestr('format.json', json.dumps({'format': 'microbench-project', 'version': 2}))
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
        if format_doc.get('format') != 'microbench-project' or format_doc.get('version') not in (1, 2):
            raise InputError('不支持的工程格式')
        doc = json.loads(archive.read('project.json'))
        if len(doc['fields']) > 32 or len(doc['runs']) > 100 or len(doc['snapshots']) > 30:
            raise InputError('工程对象超限')
        if len(doc.get('trackings', [])) > 20:
            raise InputError('追踪会话超限')
        refs = set()
        fields = list(doc['fields'])
        for snapshot in doc['snapshots']:
            fields.extend(snapshot['fields'])
        for field in fields:
            refs.add(field['source'])
        for run in doc['runs']:
            refs.update((run['source'], run['labels']))
        for session in doc.get('trackings', []):
            for frame in session['frames']:
                refs.update((frame['source'], frame['labels']))
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
        # 追踪会话只引用不可变 run 与其中对象编号；重新分析后的新编号不会迁移到这里
        for session in doc.get('trackings', []):
            if len(session['frames']) < 2 or len(session['frames']) > 32:
                raise InputError('追踪会话帧数无效')
            frame_runs = []
            for frame in session['frames']:
                run = next((r for r in doc['runs'] if r['id'] == frame['run_id']), None)
                if run is None:
                    raise InputError('追踪帧引用了缺失分析结果')
                frame_runs.append(run)
                valid_oids = {int(o['id']) for o in run['objects'] if o['id'] not in run['excluded']}
                for node in frame['nodes']:
                    if int(node['oid']) not in valid_oids:
                        raise InputError('追踪节点对象在原结果中不存在或已排除')
            if [r['id'] for r in frame_runs] != [f['run_id'] for f in session['frames']]:
                raise InputError('追踪帧顺序无效')
            node_ids = {n['node_id'] for f in session['frames'] for n in f['nodes']}
            for edge in session['edges']:
                if edge['src'] not in node_ids or edge['dst'] not in node_ids:
                    raise InputError('追踪关联引用了缺失节点')
                if edge['status'] not in ('confirmed', 'pending', 'rejected'):
                    raise InputError('追踪关联状态无效')
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
