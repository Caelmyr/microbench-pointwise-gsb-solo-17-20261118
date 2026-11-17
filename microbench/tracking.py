"""跨帧对象追踪领域逻辑：会话构建、自动关联、人工改接与派生统计。

身份模型：会话固定引用不可变分析结果（run），节点 id 由帧序号与 run 内对象编号
组成，因此重新分析某幅图不会把旧确认悄悄套到新对象上。轨迹、谱系与事件全部由
“已确认边”即时派生；合并团自身成为独立轨迹，其位移不会计入任一单体速度。
"""
import math
import uuid
from .imaging import InputError, finite

MAX_FRAMES = 32
MAX_SESSION_NODES = 8000
MAX_EDGES_PER_NODE = 6


def params(value):
    value = value or {}
    gap = int(finite(value.get('max_gap', 1), 0, 3, '允许漏检帧数'))
    return {'max_gap': gap,
            'gate_diameters': finite(value.get('gate_diameters', 2.0), 0.5, 10, '关联门限（直径倍数）'),
            'area_ratio': finite(value.get('area_ratio', 3.0), 1.1, 20, '面积比上限'),
            'event_area_ratio': finite(value.get('event_area_ratio', 6.0), 2.0, 30,
                                       '合并分裂面积比上限')}


def snapshot_nodes(run):
    """从不可变 run 快照纳入统计的对象，物理坐标使用该 run 自己的标定。"""
    excluded = set(run['excluded'])
    nodes = []
    for row in run['objects']:
        if row['id'] in excluded:
            continue
        nodes.append({'oid': int(row['id']), 'cx_px': float(row['cx']), 'cy_px': float(row['cy']),
                      'x': float(row['cx']) * run['calibration']['pixel_size'],
                      'y': float(row['cy']) * run['calibration']['pixel_size'],
                      'area_px': float(row['area_px']),
                      'area': float(row['area_px']) * run['calibration']['pixel_size'] ** 2,
                      'diameter_px': float(row['equivalent_diameter_px']),
                      'diameter': float(row['equivalent_diameter_px']) * run['calibration']['pixel_size'],
                      'circularity': float(row['circularity']),
                      'mean_intensity': float(row['mean_intensity'])})
    return nodes


def build_session(store, doc, run_ids, configuration, name):
    runs = []
    for rid in run_ids:
        run = store.run(doc, rid)
        runs.append(run)
    if len(runs) < 2:
        raise InputError('时间序列至少选择2个分析结果')
    if len(runs) > MAX_FRAMES:
        raise InputError('每个追踪会话最多%d帧' % MAX_FRAMES)
    unit = runs[0]['calibration']['unit']
    size = tuple(runs[0]['size'])
    for run in runs:
        if tuple(run['size']) != size:
            raise InputError('同一时间序列的视野尺寸必须一致')
        if run['calibration']['unit'] != unit:
            raise InputError('同一时间序列的标定单位必须一致')
    ordered = sorted(runs, key=lambda r: float(r['timestamp']))
    timestamps = [float(r['timestamp']) for r in ordered]
    if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        raise InputError('观测秒必须严格递增，不能有相同时刻')
    frames = []
    total = 0
    for index, run in enumerate(ordered):
        field = store.field(doc, run['field_id'])
        nodes = snapshot_nodes(run)
        total += len(nodes)
        frames.append({'field_id': run['field_id'], 'run_id': run['id'], 'name': run['name'],
                       'timestamp': float(run['timestamp']), 'size': list(run['size']),
                       'source': run['source'], 'labels': run['labels'],
                       'unit': unit, 'pixel_size': float(run['calibration']['pixel_size']),
                       'field_active_run': field['active_run'], 'nodes': nodes})
    if total > MAX_SESSION_NODES:
        raise InputError('会话节点超过%d，请缩小范围或调整最小面积' % MAX_SESSION_NODES)
    session = {'id': uuid.uuid4().hex, 'name': str(name or '追踪会话')[:120], 'created': 0,
               'params': params(configuration), 'frames': frames, 'edges': [],
               'track_names': {}, 'track_counter': 0}
    for index, frame in enumerate(session['frames']):
        for node in frame['nodes']:
            node['node_id'] = 'f%d-%d' % (index, node['oid'])
    auto_link(session)
    return session


def node_map(session):
    out = {}
    for index, frame in enumerate(session['frames']):
        for node in frame['nodes']:
            out[node['node_id']] = node
    return out


def frame_of(node_id):
    return int(node_id.split('-', 1)[0][1:])


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def _candidates(session, i, j, gate, area_ratio, existing):
    """空间网格分桶的门限候选对，避免稠密场景的 O(n²) 比较。"""
    a = session['frames'][i]['nodes']
    b = session['frames'][j]['nodes']
    if not a or not b or gate <= 0:
        return []
    cell = max(gate, 1e-12)
    buckets = {}
    for node in b:
        buckets.setdefault((math.floor(node['x'] / cell), math.floor(node['y'] / cell)), []).append(node)
    found = []
    for src in a:
        cx, cy = math.floor(src['x'] / cell), math.floor(src['y'] / cell)
        for u in range(cx - 1, cx + 2):
            for v in range(cy - 1, cy + 2):
                for dst in buckets.get((u, v), []):
                    pair = (src['node_id'], dst['node_id'])
                    if pair in existing:
                        continue
                    distance = math.hypot(src['x'] - dst['x'], src['y'] - dst['y'])
                    if distance > gate:
                        continue
                    ratio = max(src['area'], dst['area']) / max(min(src['area'], dst['area']), 1e-12)
                    if ratio > area_ratio:
                        continue
                    score = distance / gate + abs(math.log(ratio))
                    found.append({'src': src['node_id'], 'dst': dst['node_id'],
                                  'score': float(score), 'distance': float(distance)})
    return found


def auto_link(session):
    """按帧间隔从小到大贪心建链；靠近产生一对多候选时一律保留为待核对。"""
    p = session['params']
    diameters = [n['diameter'] for f in session['frames'] for n in f['nodes']]
    global_gate = median(diameters) * p['gate_diameters']
    existing = {(e['src'], e['dst']) for e in session['edges']}
    incident = {n['node_id'] for e in session['edges'] for n in (e['src'], e['dst'])}
    for gap in range(1, p['max_gap'] + 2):
        for i in range(0, len(session['frames']) - gap):
            j = i + gap
            source_nodes = session['frames'][i]['nodes']
            gate = median([n['diameter'] for n in source_nodes]) * p['gate_diameters'] or global_gate
            # 常规接续要求面积接近；面积阶跃只允许出现在显式的合并/分裂候选中
            candidates = [c for c in _candidates(session, i, j, gate, p['area_ratio'], existing)
                          if c['src'] not in incident and c['dst'] not in incident]
            # 贪心逐对配对；只有在仍可配对的候选里存在分数接近的竞争对手（贴近难分）时
            # 才留给人工核对，分数悬殊的唯一近邻直接确认。
            ordered = sorted(candidates, key=lambda c: c['score'])
            matched_src, matched_dst = set(), set()
            for cand in ordered:
                if cand['src'] in incident or cand['dst'] in incident:
                    continue
                if cand['src'] in matched_src or cand['dst'] in matched_dst:
                    continue
                rivals = [r for r in ordered
                          if r is not cand and r['src'] not in matched_src
                          and r['dst'] not in matched_dst
                          and r['src'] not in incident and r['dst'] not in incident
                          and (r['src'] == cand['src'] or r['dst'] == cand['dst'])]
                ambiguous = any(r['score'] <= cand['score'] + 0.35 for r in rivals)
                edge = {'id': uuid.uuid4().hex, 'src': cand['src'], 'dst': cand['dst'],
                        'status': 'pending' if ambiguous else 'confirmed',
                        'origin': 'gap' if gap > 1 else 'auto', 'score': cand['score'],
                        'distance': cand['distance'], 'frame_gap': gap, 'demoted': False}
                session['edges'].append(edge)
                existing.add((cand['src'], cand['dst']))
                if edge['status'] == 'confirmed':
                    # 确认边才占用节点；待核对边留给第二轮与竞争候选一起呈现
                    incident.update((cand['src'], cand['dst']))
                    matched_src.add(cand['src'])
                    matched_dst.add(cand['dst'])
            # 贪心后仍未配对的节点（合并/分裂、入图出图、贴近竞争）：放宽面积比后，门限内
            # 的配对全部保留为候选由人工核对，程序不擅自把任意单体续到合体上。
            leftovers = [c for c in _candidates(session, i, j, gate, p['event_area_ratio'], existing)
                         if c['src'] not in matched_src and c['dst'] not in matched_dst
                         and c['src'] not in incident and c['dst'] not in incident]
            for cand in sorted(leftovers, key=lambda c: c['score']):
                edge = {'id': uuid.uuid4().hex, 'src': cand['src'], 'dst': cand['dst'],
                        'status': 'pending', 'origin': 'gap' if gap > 1 else 'auto',
                        'score': cand['score'], 'distance': cand['distance'],
                        'frame_gap': gap, 'demoted': False}
                session['edges'].append(edge)
                existing.add((cand['src'], cand['dst']))
        if len(session['edges']) > MAX_SESSION_NODES * MAX_EDGES_PER_NODE:
            raise InputError('候选关联过多，请调小关联门限或面积比上限')
    assign_names(session)


def _confirmed_graph(session):
    indeg, outdeg, forward, backward = {}, {}, {}, {}
    for edge in session['edges']:
        indeg.setdefault(edge['dst'], []).append(edge)
        outdeg.setdefault(edge['src'], []).append(edge)
        if edge['status'] == 'confirmed':
            forward.setdefault(edge['src'], []).append(edge)
            backward.setdefault(edge['dst'], []).append(edge)
    return indeg, outdeg, forward, backward


def assign_names(session):
    """轨迹显示名按稳定头节点持久化；改接后新头取新号，旧号不复用。"""
    _, _, forward, backward = _confirmed_graph(session)
    heads = set()
    for frame in session['frames']:
        for node in frame['nodes']:
            if len(backward.get(node['node_id'], [])) != 1:
                heads.add(node['node_id'])
    for head in sorted(heads, key=lambda n: (frame_of(n), n)):
        if head not in session['track_names']:
            session['track_counter'] += 1
            session['track_names'][head] = 'T%d' % session['track_counter']


def _lineages(session, forward, backward):
    nodes = [n['node_id'] for f in session['frames'] for n in f['nodes']]
    seen, groups = set(), []
    for start in nodes:
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for edge in forward.get(current, []) + backward.get(current, []):
                other = edge['dst'] if edge['src'] == current else edge['src']
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        groups.append(sorted(component, key=lambda n: (frame_of(n), n)))
    return groups


def derive(session):
    """从边派生谱系、轨迹、事件、可靠连续段与速度；结构变化即整体重算。"""
    nodes = node_map(session)
    indeg, outdeg, forward, backward = _confirmed_graph(session)
    assign_names(session)
    lineages = _lineages(session, forward, backward)
    lineage_of = {}
    tracks, node_track = [], {}
    for component in lineages:
        root = component[0]
        for node_id in component:
            lineage_of[node_id] = root
        starts = [n for n in component if len(backward.get(n, [])) != 1]
        for start in sorted(starts, key=lambda n: (frame_of(n), n)):
            chain, current = [], start
            while True:
                chain.append(current)
                outs = forward.get(current, [])
                if len(outs) != 1:
                    break
                nxt = outs[0]['dst']
                if len(backward.get(nxt, [])) != 1:
                    break
                current = nxt
            name = session['track_names'].get(start) or session['track_names'].get(current) or ('T?' + start)
            for node_id in chain:
                node_track[node_id] = name
            tracks.append(_track_stats(session, nodes, name, root, chain, forward, indeg, outdeg))
    events = _events(session, nodes, indeg, outdeg, node_track)
    edges = [_edge_row(session, nodes, edge, indeg, outdeg) for edge in session['edges']]
    pending_groups = _pending_groups(session, nodes)
    frames = []
    for index, frame in enumerate(session['frames']):
        rows = []
        for node in frame['nodes']:
            nid = node['node_id']
            row = dict(node)
            row['track'] = node_track.get(nid)
            row['lineage'] = lineage_of.get(nid)
            row['role'] = _role(nid, indeg, outdeg)
            rows.append(row)
        frames.append({'index': index, 'field_id': frame['field_id'], 'run_id': frame['run_id'],
                       'name': frame['name'], 'timestamp': frame['timestamp'], 'unit': frame['unit'],
                       'pixel_size': frame['pixel_size'], 'size': frame['size'], 'source': frame['source'],
                       'labels': frame['labels'], 'field_active_run': frame['field_active_run'],
                       'stale': frame['field_active_run'] != frame['run_id'], 'nodes': rows})
    return {'id': session['id'], 'name': session['name'], 'created': session['created'],
            'params': session['params'], 'frames': frames, 'edges': edges, 'tracks': tracks,
            'events': events, 'pending_groups': pending_groups,
            'lineage_count': len(lineages), 'track_count': len(tracks)}


def _role(node_id, indeg, outdeg):
    role = []
    if len(indeg.get(node_id, [])) >= 2:
        role.append('merge')
    if len(outdeg.get(node_id, [])) >= 2:
        role.append('split')
    return '+'.join(role)


def _edge_row(session, nodes, edge, indeg, outdeg):
    src, dst = nodes[edge['src']], nodes[edge['dst']]
    dt = float(session['frames'][frame_of(edge['dst'])]['timestamp'] -
               session['frames'][frame_of(edge['src'])]['timestamp'])
    distance = math.hypot(src['x'] - dst['x'], src['y'] - dst['y'])
    speed = distance / dt if dt > 0 else None
    event_edge = bool(_role(edge['dst'], indeg, outdeg) or _role(edge['src'], indeg, outdeg))
    reasons = []
    if edge['frame_gap'] > 1:
        reasons.append('gap')
    if event_edge:
        reasons.append('event')
    reliable = edge['status'] == 'confirmed' and edge['frame_gap'] == 1 and not event_edge
    return {**edge, 'src_frame': frame_of(edge['src']), 'dst_frame': frame_of(edge['dst']),
            'dt': dt, 'distance': distance, 'speed': speed,
            'vx': (dst['x'] - src['x']) / dt if dt > 0 else None,
            'vy': (dst['y'] - src['y']) / dt if dt > 0 else None,
            'reliable': reliable, 'reason': '+'.join(reasons) if reasons else 'ok'}


def _track_stats(session, nodes, name, lineage, chain, forward, indeg, outdeg):
    chain_nodes = [nodes[n] for n in chain]
    times = [session['frames'][frame_of(n)]['timestamp'] for n in chain]
    path, reliable_edges, segments = 0.0, [], []
    current_seg = []
    for a, b in zip(chain, chain[1:]):
        edge = next(e for e in forward[a] if e['dst'] == b)
        row = _edge_row(session, nodes, edge, indeg, outdeg)
        path += row['distance']
        if row['reliable']:
            reliable_edges.append(row)
            current_seg.append(row)
        elif current_seg:
            segments.append(_segment(session, current_seg))
            current_seg = []
    if current_seg:
        segments.append(_segment(session, current_seg))
    speeds = [e['speed'] for e in reliable_edges if e['speed'] is not None]
    first, last = chain_nodes[0], chain_nodes[-1]
    net = math.hypot(last['x'] - first['x'], last['y'] - first['y'])
    return {'name': name, 'lineage': lineage, 'nodes': list(chain),
            'start_time': min(times), 'end_time': max(times), 'point_count': len(chain),
            'gap_count': (len(chain) - 1) - len(reliable_edges),
            'path_length': path, 'net_displacement': net,
            'straightness': net / path if path else None,
            'mean_speed': sum(speeds) / len(speeds) if speeds else None,
            'max_speed': max(speeds) if speeds else None,
            'reliable_segments': segments,
            'is_event_blob': any(_role(n, indeg, outdeg) for n in chain)}


def _segment(session, edges):
    speeds = [e['speed'] for e in edges if e['speed'] is not None]
    length = sum(e['distance'] for e in edges)
    net = math.hypot(sum(e['vx'] * e['dt'] for e in edges), sum(e['vy'] * e['dt'] for e in edges))
    return {'src': edges[0]['src'], 'dst': edges[-1]['dst'],
            'start_time': session['frames'][edges[0]['src_frame']]['timestamp'],
            'end_time': session['frames'][edges[-1]['dst_frame']]['timestamp'],
            'points': len(edges) + 1, 'length': length,
            'mean_speed': sum(speeds) / len(speeds) if speeds else None,
            'max_speed': max(speeds) if speeds else None,
            'straightness': net / length if length else None}


def _events(session, nodes, indeg, outdeg, node_track):
    events = []
    for node_id, node in nodes.items():
        incoming = [e for e in indeg.get(node_id, []) if e['status'] != 'rejected']
        outgoing = [e for e in outdeg.get(node_id, []) if e['status'] != 'rejected']
        when = session['frames'][frame_of(node_id)]['timestamp']
        if len(incoming) >= 2:
            events.append({'kind': 'merge', 'time': when, 'node': node_id, 'track': node_track.get(node_id),
                           'members': sorted([e['src'] for e in incoming]),
                           'status': 'confirmed' if all(e['status'] == 'confirmed' for e in incoming) else 'pending'})
        if len(outgoing) >= 2:
            events.append({'kind': 'split', 'time': when, 'node': node_id, 'track': node_track.get(node_id),
                           'members': sorted([e['dst'] for e in outgoing]),
                           'status': 'confirmed' if all(e['status'] == 'confirmed' for e in outgoing) else 'pending'})
    for edge in session['edges']:
        if edge['frame_gap'] > 1:
            events.append({'kind': 'gap', 'time': None, 'node': edge['id'], 'track': None,
                           'members': [edge['src'], edge['dst']], 'status': edge['status'],
                           'missing_frames': list(range(frame_of(edge['src']) + 1, frame_of(edge['dst'])))})
    return sorted(events, key=lambda e: (e['time'] if e['time'] is not None else -1, e['kind']))


def _pending_groups(session, nodes):
    adjacency, pending = {}, []
    for edge in session['edges']:
        if edge['status'] != 'pending':
            continue
        pending.append(edge)
        adjacency.setdefault(edge['src'], set()).add(edge['dst'])
        adjacency.setdefault(edge['dst'], set()).add(edge['src'])
    seen, groups = set(), []
    for edge in pending:
        roots = [n for n in (edge['src'], edge['dst']) if n not in seen]
        if not roots:
            continue
        stack, members, edge_ids = roots[:], set(roots), set()
        seen.update(roots)
        while stack:
            current = stack.pop()
            for other in adjacency.get(current, ()):
                if other not in seen:
                    seen.add(other)
                    members.add(other)
                    stack.append(other)
        for cand in pending:
            if cand['src'] in members and cand['dst'] in members:
                edge_ids.add(cand['id'])
        groups.append({'nodes': sorted(members, key=lambda n: (frame_of(n), n)),
                       'edges': sorted(edge_ids),
                       'frame_span': [min(frame_of(n) for n in members), max(frame_of(n) for n in members)]})
    return groups


def confirm_group(session, pairs):
    """整组确认合并/分裂：同一节点的多条确认边共同定义事件，不做互相降级。"""
    if not pairs:
        raise InputError('请提供要整组确认的关联')
    edges = []
    for item in pairs:
        edge = next((e for e in session['edges']
                     if e['src'] == item.get('src') and e['dst'] == item.get('dst')), None)
        if edge is None:
            raise InputError('组内有关联不存在')
        edges.append(edge)
    srcs = {e['src'] for e in edges}
    dsts = {e['dst'] for e in edges}
    if len(srcs) < 2 and len(dsts) < 2:
        raise InputError('整组确认仅用于一对多的合并/分裂候选')
    for edge in edges:
        edge['status'] = 'confirmed'
    assign_names(session)
    return [e['id'] for e in edges]


def mutate_edge(session, nodes, action, src, dst):
    """人工改接：加边/确认会把冲突的已确认边降级为候选，绝不并行延续两个编号。"""
    if action == 'add':
        if src not in nodes or dst not in nodes:
            raise InputError('改接端点不属于本会话')
        if frame_of(src) >= frame_of(dst):
            raise InputError('关联必须从较早帧指向较晚帧')
        if frame_of(dst) - frame_of(src) > session['params']['max_gap'] + 2:
            raise InputError('手动跨帧不能超过允许漏检范围+1')
        existing = next((e for e in session['edges'] if e['src'] == src and e['dst'] == dst), None)
        if existing is None:
            if len(session['edges']) >= MAX_SESSION_NODES * MAX_EDGES_PER_NODE:
                raise InputError('关联数量已达上限')
            edge = {'id': uuid.uuid4().hex, 'src': src, 'dst': dst, 'status': 'confirmed',
                    'origin': 'manual', 'score': None, 'distance': float(math.hypot(
                        nodes[src]['x'] - nodes[dst]['x'], nodes[src]['y'] - nodes[dst]['y'])),
                    'frame_gap': frame_of(dst) - frame_of(src), 'demoted': False}
            session['edges'].append(edge)
        else:
            edge = existing
            edge['status'] = 'confirmed'
        notes = _demote_conflicts(session, edge)
        assign_names(session)
        return notes
    edge = next((e for e in session['edges'] if e['src'] == src and e['dst'] == dst), None)
    if edge is None:
        raise InputError('关联不存在')
    if action == 'confirm':
        edge['status'] = 'confirmed'
        notes = _demote_conflicts(session, edge)
        assign_names(session)
        return notes
    if action == 'reject':
        edge['status'] = 'rejected'
        assign_names(session)
        return []
    if action == 'delete':
        if edge['origin'] != 'manual':
            raise InputError('自动候选只能驳回保留记录，手动边才可删除')
        session['edges'].remove(edge)
        assign_names(session)
        return []
    raise InputError('未知关联操作')


def _demote_conflicts(session, kept):
    notes = []
    for edge in session['edges']:
        if edge is kept or edge['status'] != 'confirmed':
            continue
        if edge['src'] == kept['src'] or edge['dst'] == kept['dst']:
            edge['status'] = 'pending'
            edge['demoted'] = True
            notes.append(edge['id'])
    return notes
