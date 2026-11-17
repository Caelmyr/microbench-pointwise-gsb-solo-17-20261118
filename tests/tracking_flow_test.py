#!/usr/bin/env python3
"""MicroBench 追踪子系统端到端测试（标准库 unittest，无外部夹具）。"""
import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_lab import create_app
from microbench import install


class TrackingFlowTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='mb-test-')
        app = create_app(str(Path(self.root) / 'preparation'))
        install(app, str(Path(self.root) / 'bench'))
        self.client = app.test_client()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def post_json(self, path, payload):
        response = self.client.post(path, data=json.dumps(payload),
                                    headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 200, response.data.decode())
        return response.get_json()

    def tracking_demo(self):
        return self.post_json('/bench/tracking-demo', {})

    def _demo_detail(self):
        doc = self.tracking_demo()
        tid = doc['trackings'][0]['id']
        detail = self.client.get('/bench/projects/%s/trackings/%s' % (doc['id'], tid)).get_json()
        return doc, detail

    def test_demo_builds_session_with_pending_candidates_and_events(self):
        doc, detail = self._demo_detail()
        self.assertGreaterEqual(detail['lineage_count'], 1)
        # 靠近/合并场景必须保留待核对候选，不能擅自全部续号
        self.assertTrue(any(e['status'] == 'pending' for e in detail['edges']))
        self.assertTrue(detail['pending_groups'])
        # 所有节点都有轨迹或作为事件团出现
        node_count = sum(len(f['nodes']) for f in detail['frames'])
        self.assertGreater(node_count, 10)
        # 漏检帧的接续边必须带 frame_gap=2 且默认不进可靠速度
        gap_edges = [e for e in detail['edges'] if e['frame_gap'] > 1]
        self.assertTrue(gap_edges)
        self.assertTrue(all(not e['reliable'] for e in gap_edges))

    def test_manual_relink_demotes_conflicts_and_recomputes_tracks(self):
        doc, detail = self._demo_detail()
        tid = doc['trackings'][0]['id']
        group = detail['pending_groups'][0]
        edge = next(e for e in detail['edges'] if e['id'] == group['edges'][0])
        before_names = {n['node_id']: n['track'] for f in detail['frames'] for n in f['nodes']}
        updated = self.post_json('/bench/projects/%s/trackings/%s/links' % (doc['id'], tid),
                                 {'revision': doc['revision'], 'action': 'confirm',
                                  'src': edge['src'], 'dst': edge['dst']})
        self.assertTrue(any(e['id'] == edge['id'] and e['status'] == 'confirmed'
                            for e in updated['edges']))
        # 同组里其他共享端点的已确认边应被降级为候选
        conflict = [e for e in group['edges'] if e != edge['id']]
        if conflict:
            other = next(e for e in updated['edges'] if e['id'] == conflict[0])
            self.assertIn(other['status'], ('pending', 'rejected'))
        # 改接后至少有一个节点的轨迹身份发生变化，且全部节点仍有归属
        after = {n['node_id']: n['track'] for f in updated['frames'] for n in f['nodes']}
        self.assertTrue(any(before_names.get(k) != v for k, v in after.items()))
        self.assertTrue(all(v for v in after.values()))

    def test_add_and_reject_manual_edge(self):
        doc, detail = self._demo_detail()
        tid = doc['trackings'][0]['id']
        first = detail['frames'][0]['nodes'][0]
        # 找下一帧中没有确认入边的节点作为手动目标
        used_dst = {e['dst'] for e in detail['edges'] if e['status'] == 'confirmed'}
        target = next(n for n in detail['frames'][1]['nodes'] if n['node_id'] not in used_dst)
        updated = self.post_json('/bench/projects/%s/trackings/%s/links' % (doc['id'], tid),
                                 {'revision': doc['revision'], 'action': 'add',
                                  'src': first['node_id'], 'dst': target['node_id']})
        manual = next(e for e in updated['edges']
                      if e['src'] == first['node_id'] and e['dst'] == target['node_id'])
        self.assertEqual(manual['origin'], 'manual')
        self.assertEqual(manual['status'], 'confirmed')
        # 手动边可以删除，自动边只能驳回
        rejected = self.post_json('/bench/projects/%s/trackings/%s/links' % (doc['id'], tid),
                                  {'revision': doc['revision'] + 1, 'action': 'delete',
                                   'src': first['node_id'], 'dst': target['node_id']})
        self.assertFalse(any(e['id'] == manual['id'] for e in rejected['edges']))
        auto = next(e for e in rejected['edges'] if e['origin'] in ('auto', 'gap'))
        again = self.post_json('/bench/projects/%s/trackings/%s/links' % (doc['id'], tid),
                               {'revision': doc['revision'] + 2, 'action': 'reject',
                                'src': auto['src'], 'dst': auto['dst']})
        self.assertEqual(next(e for e in again['edges'] if e['id'] == auto['id'])['status'],
                         'rejected')

    def test_event_blob_velocity_excluded_from_reliable_segments(self):
        doc, detail = self._demo_detail()
        tid = doc['trackings'][0]['id']
        revision = doc['revision']
        # 一对多的合并/分裂候选必须整组确认；驳回距离过远的竞争候选
        far = next(e for e in detail['edges'] if e['status'] == 'pending' and (e['score'] or 0) > 1.4)
        detail = self.post_json('/bench/projects/%s/trackings/%s/links' % (doc['id'], tid),
                                {'revision': revision, 'action': 'reject',
                                 'src': far['src'], 'dst': far['dst']})
        revision += 1
        for group in detail['pending_groups']:
            pairs = [{'src': next(e['src'] for e in detail['edges'] if e['id'] == eid),
                      'dst': next(e['dst'] for e in detail['edges'] if e['id'] == eid)}
                     for eid in group['edges']
                     if next(e for e in detail['edges'] if e['id'] == eid)['status'] == 'pending']
            detail = self.post_json('/bench/projects/%s/trackings/%s/links' % (doc['id'], tid),
                                    {'revision': revision, 'action': 'confirm_group', 'pairs': pairs})
            revision += 1
        self.assertFalse([e for e in detail['edges'] if e['status'] == 'pending'])
        events = [e for e in detail['events'] if e['kind'] in ('merge', 'split')]
        self.assertTrue(events)
        self.assertTrue(all(e['status'] == 'confirmed' for e in events))
        blob_tracks = {t['name'] for t in detail['tracks'] if t['is_event_blob']}
        self.assertTrue(blob_tracks)
        for track in detail['tracks']:
            if track['name'] in blob_tracks:
                for segment in track['reliable_segments']:
                    self.assertIsNone(segment['mean_speed'])
        # 可靠边不能触碰任何事件节点
        event_nodes = {e['node'] for e in events}
        for edge in detail['edges']:
            if edge['reliable']:
                self.assertNotIn(edge['src'], event_nodes)
                self.assertNotIn(edge['dst'], event_nodes)

    def test_reanalysis_keeps_old_session_reviewable_and_marks_stale(self):
        doc = self.tracking_demo()
        tid = doc['trackings'][0]['id']
        field = sorted(doc['fields'], key=lambda f: f['timestamp'])[2]
        original_run = field['active_run']
        # 修改分割设置后重新分析该帧，产生新编号的 run
        doc = self.post_json('/bench/projects/%s/fields/%s' % (doc['id'], field['id']),
                             {'revision': doc['revision'],
                              'settings': {**field_of(doc, field)['settings'], 'min_area': 25}})
        doc = self.post_json('/bench/projects/%s/fields/%s/analyze' % (doc['id'], field['id']),
                             {'revision': doc['revision']})
        field = field_of(doc, field)
        self.assertNotEqual(field['active_run'], original_run)
        detail = self.client.get('/bench/projects/%s/trackings/%s' % (doc['id'], tid)).get_json()
        stale = [f for f in detail['frames'] if f['stale']]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]['run_id'], _session_run(detail, field))
        # 旧会话仍可导出
        response = self.client.get('/bench/projects/%s/trackings/%s/tracking.zip'
                                   % (doc['id'], tid))
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            names = set(archive.namelist())
            self.assertIn('positions.csv', names)
            self.assertIn('links_velocities.csv', names)
            self.assertIn('reliable_segments.csv', names)
            self.assertIn('events.csv', names)
            self.assertIn('tracking.json', names)

    def test_project_roundtrip_preserves_judgments(self):
        doc = self.tracking_demo()
        tid = doc['trackings'][0]['id']
        detail = self.client.get('/bench/projects/%s/trackings/%s' % (doc['id'], tid)).get_json()
        edge = next(e for e in detail['edges'] if e['status'] == 'pending')
        self.post_json('/bench/projects/%s/trackings/%s/links' % (doc['id'], tid),
                       {'revision': doc['revision'], 'action': 'reject',
                        'src': edge['src'], 'dst': edge['dst']})
        raw = self.client.get('/bench/projects/%s/project.zip' % doc['id']).data
        response = self.client.post('/bench/import', data={'project': (io.BytesIO(raw), 'p.zip')},
                                    content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200, response.data.decode())
        imported = response.get_json()
        self.assertEqual(len(imported['trackings']), 1)
        new_tid = imported['trackings'][0]['id']
        restored = self.client.get('/bench/projects/%s/trackings/%s'
                                   % (imported['id'], new_tid)).get_json()
        statuses = {(e['src'], e['dst']): e['status'] for e in restored['edges']}
        self.assertEqual(statuses[(edge['src'], edge['dst'])], 'rejected')
        self.assertEqual(restored['track_count'], detail['track_count'])

    def test_revision_conflict_blocks_stale_edit(self):
        doc, detail = self._demo_detail()
        tid = doc['trackings'][0]['id']
        edge = next(e for e in detail['edges'] if e['status'] == 'pending')
        response = self.client.post('/bench/projects/%s/trackings/%s/links' % (doc['id'], tid),
                                    data=json.dumps({'revision': doc['revision'] - 1,
                                                     'action': 'confirm',
                                                     'src': edge['src'], 'dst': edge['dst']}),
                                    headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 409)

    def test_tracking_frame_overlay_renders(self):
        doc, detail = self._demo_detail()
        tid = doc['trackings'][0]['id']
        response = self.client.get('/bench/projects/%s/trackings/%s/frame/2.png'
                                   % (doc['id'], tid))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.startswith(b'\x89PNG'))


def field_of(doc, field):
    return next(f for f in doc['fields'] if f['id'] == field['id'])


def _session_run(detail, field):
    frame = next(f for f in detail['frames'] if f['field_id'] == field['id'])
    return frame['run_id']


if __name__ == '__main__':
    unittest.main(verbosity=2)
