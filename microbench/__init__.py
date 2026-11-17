import copy
import io
import json
from flask import Blueprint, request, jsonify, render_template, send_file
from .storage import Store, RevisionConflict
from .service import Bench
from .tracking_service import TrackingService, tracking_demo_images
from .imaging import InputError, decode, png, demo_images, quality, finite
from .exporting import output_zip, project_zip, import_project, csv_bytes
from .measurements import object_rows
from .tracking_export import tracking_zip
from .track_render import render_tracking_frame


def install(app, root):
    store = Store(root)
    bench = Bench(store)
    tracker = TrackingService(store)
    app.extensions['microbench'] = bench
    app.extensions['microbench_tracking'] = tracker
    api = Blueprint('microbench', __name__)

    def body():
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            raise InputError('需要JSON对象')
        return value

    def mutate(pid, callback):
        data = body()
        doc = store.mutate(pid, data.get('revision'), lambda current: callback(current, data))
        return jsonify(bench.detail(doc))

    def download(raw, filename, mimetype):
        return send_file(io.BytesIO(raw), mimetype=mimetype, as_attachment=True, download_name=filename)

    @api.errorhandler(RevisionConflict)
    def conflict(error):
        return jsonify(error=str(error)), 409

    @api.errorhandler(InputError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @api.get('/')
    def home():
        return render_template('bench.html')

    @api.get('/bench/health')
    def health():
        return jsonify(ok=True, product='MicroBench')

    @api.get('/bench/projects')
    def projects():
        return jsonify(store.listing())

    @api.post('/bench/projects')
    def create():
        return jsonify(bench.detail(store.create(body().get('name'))))

    @api.post('/bench/demo')
    def demo():
        doc = store.create('合成颗粒观察组')
        def fill(current):
            for index, image in enumerate(demo_images()):
                store.add_image(current, image, '合成视野 ' + str(index + 1), index * 2.5)
        doc = store.mutate(doc['id'], doc['revision'], fill)
        return jsonify(bench.detail(doc))

    @api.post('/bench/tracking-demo')
    def tracking_demo():
        doc = store.create('时序颗粒追踪演示')
        def fill(current):
            for index, image in enumerate(tracking_demo_images()):
                store.add_image(current, image, '追踪视野 t%d' % index, float(index))
        doc = store.mutate(doc['id'], doc['revision'], fill)
        # 对每帧发布分析结果，再用固定结果建立追踪会话
        for field in doc['fields']:
            doc = bench.analyze(doc['id'], doc['revision'], field['id'])
        run_ids = [store.field(doc, f['id'])['active_run'] for f in
                   sorted(doc['fields'], key=lambda f: f['timestamp'])]
        def build(current):
            tracker.create(current, {'name': '演示追踪', 'run_ids': run_ids,
                                     'params': {'max_gap': 1, 'gate_diameters': 2.5, 'area_ratio': 4}})
        doc = store.mutate(doc['id'], doc['revision'], build)
        return jsonify(bench.detail(doc))

    @api.get('/bench/projects/<pid>')
    def project(pid):
        with store.lock:
            return jsonify(bench.detail(store.load(pid)))

    @api.post('/bench/projects/<pid>/images')
    def images(pid):
        uploads = request.files.getlist('images')
        if not uploads or len(uploads) > 32:
            raise InputError('请选择1–32张图像')
        decoded = [(decode(upload.read()), upload.filename) for upload in uploads]
        revision = int(request.form.get('revision', 0))
        def add(current):
            for image, name in decoded:
                store.add_image(current, image, name)
        return jsonify(bench.detail(store.mutate(pid, revision, add)))

    @api.post('/bench/projects/<pid>/fields/<fid>')
    def update_field(pid, fid):
        return mutate(pid, lambda doc, data: bench.update_field(doc, fid, data))

    @api.post('/bench/projects/<pid>/fields/<fid>/analyze')
    def analyze(pid, fid):
        data = body()
        return jsonify(bench.detail(bench.analyze(pid, data.get('revision'), fid)))

    @api.post('/bench/projects/<pid>/fields/<fid>/annotation')
    def add_annotation(pid, fid):
        return mutate(pid, lambda doc, data: bench.add_annotation(doc, fid, data))

    @api.post('/bench/projects/<pid>/fields/<fid>/clear-annotations')
    def clear_annotation(pid, fid):
        def clear(doc, data):
            store.field(doc, fid)['annotations'] = []
        return mutate(pid, clear)

    @api.post('/bench/projects/<pid>/runs/<rid>/selection')
    def selection(pid, rid):
        return mutate(pid, lambda doc, data: bench.exclude(doc, rid, data.get('excluded', [])))

    @api.get('/bench/projects/<pid>/source/<fid>.png')
    def source(pid, fid):
        doc = store.load(pid)
        field = store.field(doc, fid)
        return send_file(io.BytesIO(store.asset(field['source'])), mimetype='image/png')

    @api.get('/bench/projects/<pid>/runs/<rid>/overlay.png')
    def render(pid, rid):
        return send_file(io.BytesIO(png(bench.render(store.load(pid), rid))), mimetype='image/png')

    @api.get('/bench/projects/<pid>/runs/<rid>/export.zip')
    def export(pid, rid):
        return download(output_zip(bench, store.load(pid), rid), 'microbench-analysis.zip', 'application/zip')

    @api.get('/bench/projects/<pid>/runs/<rid>/objects.csv')
    def objects(pid, rid):
        run = store.run(store.load(pid), rid)
        return download(csv_bytes(object_rows(run)), 'objects.csv', 'text/csv')

    @api.post('/bench/projects/<pid>/fields/<fid>/profile')
    def profile(pid, fid):
        return jsonify(bench.profile(store.load(pid), fid, body()))

    @api.get('/bench/projects/<pid>/fields/<fid>/quality')
    def image_quality(pid, fid):
        doc = store.load(pid)
        return jsonify(quality(store.image(store.field(doc, fid)['source'])))

    @api.get('/bench/projects/<pid>/comparison')
    def comparison(pid):
        return jsonify(bench.comparison(store.load(pid)))

    @api.post('/bench/projects/<pid>/snapshot')
    def snapshot(pid):
        return mutate(pid, lambda doc, data: store.snapshot(doc, data.get('name', '命名观察')))

    @api.post('/bench/projects/<pid>/restore')
    def restore(pid):
        return mutate(pid, lambda doc, data: store.restore(doc, data.get('snapshot_id')))

    @api.get('/bench/projects/<pid>/project.zip')
    def project_export(pid):
        return download(project_zip(store, store.load(pid)), 'microbench-project.zip', 'application/zip')

    @api.post('/bench/import')
    def project_import():
        upload = request.files.get('project')
        if upload is None:
            raise InputError('请选择工程ZIP')
        return jsonify(bench.detail(import_project(store, upload.read())))

    @api.post('/bench/projects/<pid>/trackings')
    def tracking_create(pid):
        def build(doc, data):
            return tracker.create(doc, data)
        return mutate_tracking(pid, build)

    @api.get('/bench/projects/<pid>/trackings/<tid>')
    def tracking_detail(pid, tid):
        with store.lock:
            return jsonify(tracker.detail(store.load(pid), tid))

    @api.post('/bench/projects/<pid>/trackings/<tid>/links')
    def tracking_links(pid, tid):
        def edit(doc, data):
            return tracker.edit_link(doc, tid, data)
        return mutate_tracking(pid, edit)

    @api.post('/bench/projects/<pid>/trackings/<tid>/rename')
    def tracking_rename(pid, tid):
        def edit(doc, data):
            return tracker.rename(doc, tid, data.get('name'))
        return mutate_tracking(pid, edit)

    @api.get('/bench/projects/<pid>/trackings/<tid>/frame/<int:index>.png')
    def tracking_frame(pid, tid, index):
        with store.lock:
            derived = tracker.detail(store.load(pid), tid)
        if not 0 <= index < len(derived['frames']):
            raise InputError('帧序号超出范围')
        selected = request.args.get('selected') or None
        show_rejected = request.args.get('rejected') == '1'
        with store.lock:
            image = render_tracking_frame(store, derived, index, selected, show_rejected)
        return send_file(io.BytesIO(png(image)), mimetype='image/png')

    @api.get('/bench/projects/<pid>/trackings/<tid>/tracking.zip')
    def tracking_download(pid, tid):
        with store.lock:
            doc = store.load(pid)
            session = store.tracking(doc, tid)
            derived = tracker.detail(doc, tid)
            raw = tracking_zip(store, session, derived)
        return send_file(io.BytesIO(raw), mimetype='application/zip', as_attachment=True,
                         download_name='microbench-tracking.zip')

    def mutate_tracking(pid, callback):
        data = body()
        with store.lock:
            doc = store.load(pid)
            if data.get('revision') != doc['revision']:
                raise RevisionConflict('工程已变化，请重新载入后操作')
            derived = callback(doc, data)
            doc['revision'] += 1
            store.write(doc)
        return jsonify(derived)

    app.register_blueprint(api)
