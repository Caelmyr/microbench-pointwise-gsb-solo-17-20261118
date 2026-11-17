from . import model
from . import validation as v
from . import network as net
from .preparation import prepare


class AlignmentNetwork:
    def __init__(self, store, assets):
        self.store = store
        self.assets = assets

    def _status(self, project):
        network = project.get('alignment_network')
        # 只要分析过（带指纹）就不是 empty；边数可能为 0（单帧或无可估关系）
        if not network or network.get('fingerprint') is None:
            return 'empty'
        if network.get('fingerprint') != net.frame_fingerprint(project):
            return 'stale'
        return 'current'

    def analyze(self, key, revision, body):
        reset = v.boolean(body.get('reset', False), '是否清空已有判断')
        with self.store.lock:
            project = self.store.get(key)
            if project['revision'] != revision:
                raise v.Conflict('工程已更新，请刷新')
            model.enabled_frames(project)
            if reset or not project.get('alignment_network'):
                project['alignment_network'] = net.empty_network()
            prepared = prepare(project, self.assets, False)
            network, summary = net.analyze_network(prepared, project)
        with self.store.edit(key, revision, '分析视野重叠网络') as working:
            working['alignment_network'] = network
        return self.report(key)

    def update_edge(self, key, revision, edge_id, body):
        state = v.choice(body.get('state'), '关系状态', list(net.USER_STATES))
        with self.store.edit(key, revision, '调整视野关系判断') as project:
            network = project.get('alignment_network')
            if not network or not network.get('edges'):
                raise v.Invalid('请先分析视野重叠网络')
            edge = next((item for item in network['edges'] if item['id'] == edge_id), None)
            if edge is None:
                raise v.Invalid('视野关系不存在')
            edge['state'] = state
            if state == net.STATE_EXCLUDED:
                edge['state_reason'] = '人工排除该关系'
            elif state == net.STATE_PROPOSED:
                edge['state_reason'] = ''
            elif state == net.STATE_UNRELIABLE and not edge.get('state_reason'):
                edge['state_reason'] = '人工标记为不可靠，暂不参与定位'
        return self.report(key)

    def confirm(self, key, revision):
        with self.store.lock:
            project = self.store.get(key)
            if project['revision'] != revision:
                raise v.Conflict('工程已更新，请刷新')
            if self._status(project) == 'stale':
                raise v.Conflict('源图或配准参数已改变，请重新分析视野网络')
            report = self._build_report(project)
            if not report['ready']:
                raise v.Invalid('仍存在冲突或待确认关系，无法应用；请先排除或确认')
            prepared = prepare(project, self.assets, False)
            width, height = prepared['size']
            matrices = net.build_matrices(report, project, width, height)
        with self.store.edit(key, revision, '应用视野网络配准') as project:
            for frame_id, matrix in matrices.items():
                model.find_frame(project, frame_id)['matrix'] = matrix
            if project.get('alignment_network'):
                project['alignment_network']['reference'] = project['reference']
        return self.store.get(key)

    def reset(self, key, revision):
        with self.store.edit(key, revision, '清除视野网络判断') as project:
            project['alignment_network'] = None
        return self.store.get(key)

    def report(self, key):
        with self.store.lock:
            return self._build_report(self.store.get(key))

    def _build_report(self, project):
        network = project.get('alignment_network')
        enabled_ids = [f['id'] for f in project['frames'] if f['enabled']]
        if not network or network.get('fingerprint') is None:
            return {'status': 'empty', 'nodes': [], 'edges': [], 'components': [],
                    'connected_ids': [], 'disconnected_ids': enabled_ids,
                    'conflict_ids': [], 'ambiguous_ids': [], 'blocking_ambiguous_ids': [],
                    'failures': [], 'positions': {}, 'ready': False,
                    'reference': project['reference']}
        solved = net.solve_network(network, project)
        solved['status'] = self._status(project)
        solved['reference'] = project['reference']
        solved['failures'] = network.get('failures', [])
        return solved
