"""视野重叠网络配准。

不要求每张照片都能直接对到参考帧：先在有实际重叠的视野之间估计两两平移关系
（网络的边），再以参考帧为锚做带权全局最小二乘，间接连接的帧也能定位。
环路闭合或重复纹理造成的矛盾以“冲突 / 待确认”关系显式呈现，不沿单链硬拼。
"""
import time
import numpy as np
from . import model
from . import validation as v
from . import registration as reg
from .geometry import normalized_matrix


STATE_PROPOSED = 'proposed'        # 自动检出，参与求解
STATE_UNRELIABLE = 'unreliable'    # 自动检出但证据不足，默认不参与
STATE_AMBIGUOUS = 'ambiguous'      # 多个位移方案证据相当，等待人工确认
STATE_CONFLICT = 'conflict'        # 与其它关系矛盾（环闭合残差大）
STATE_EXCLUDED = 'excluded'        # 人工排除，永不参与求解
ACTIVE_STATES = {STATE_PROPOSED, STATE_CONFLICT, STATE_AMBIGUOUS, STATE_UNRELIABLE}
SOLVING_STATES = {STATE_PROPOSED, STATE_CONFLICT}
USER_STATES = {STATE_PROPOSED, STATE_EXCLUDED, STATE_UNRELIABLE}


def edge_id(first, second):
    import hashlib
    seed = '|'.join(sorted([first, second]))
    return hashlib.sha256(seed.encode()).hexdigest()[:12]


def frame_fingerprint(project):
    """参与网络的输入指纹：源素材、启停、工作长边与配准参数。"""
    import hashlib
    import json
    alignment = project['settings']['alignment']
    payload = {
        'size': project['settings']['size'],
        'alignment': {key: alignment[key] for key in (
            'method', 'network_min_response', 'network_min_correlation', 'network_min_overlap',
            'network_ambiguity_ratio', 'network_conflict_pixels')},
        'frames': [{'id': f['id'], 'asset': f['asset'], 'enabled': f['enabled'], 'mask': f['mask']}
                   for f in project['frames']],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def empty_network():
    return {'fingerprint': None, 'reference': None, 'edges': [], 'failures': [], 'analyzed_at': None}


def validate_network(network):
    if network is None:
        return None
    if not isinstance(network, dict):
        raise v.Invalid('视野网络数据无效')
    allowed = {'fingerprint', 'reference', 'edges', 'analyzed_at', 'failures'}
    if set(network) - allowed:
        raise v.Invalid('视野网络含未知字段')
    edges = v.list_value(network.get('edges', []), '视野关系', 276)
    seen = set()
    for edge in edges:
        v.object_value(edge, '视野关系')
        v.identifier(edge.get('first'))
        v.identifier(edge.get('second'))
        if edge['first'] == edge['second']:
            raise v.Invalid('视野关系不能连接同一照片')
        key = tuple(sorted((edge['first'], edge['second'])))
        if key in seen:
            raise v.Invalid('视野关系重复')
        seen.add(key)
        v.choice(edge.get('state'), '关系状态', list(ACTIVE_STATES | {STATE_EXCLUDED}))
        for name in ('shift_x', 'shift_y'):
            v.number(edge.get(name), name, -4096, 4096)
        for name in ('response', 'overlap', 'correlation', 'confidence'):
            value = edge.get(name)
            if value is not None:
                v.number(value, name, 0, 10)
        residual = edge.get('residual')
        if residual is not None:
            v.number(residual, '环闭合残差', 0, 30000)
        ratio = edge.get('ambiguity_ratio')
        if ratio is not None:
            v.number(ratio, '歧义比率', 0, 5)
        alternatives = edge.get('alternatives', [])
        v.list_value(alternatives, '候选位移', 4)
        for item in alternatives:
            v.vector(item, '候选位移', 2, -4096, 4096)
        if edge.get('state_reason') is not None:
            v.text(edge['state_reason'], '关系说明', 300)
    fingerprint = network.get('fingerprint')
    if fingerprint is not None:
        v.text(fingerprint, '网络指纹', 80, False)
    for failure in v.list_value(network.get('failures', []), '失败关系', 276):
        v.object_value(failure, '失败关系')
        v.identifier(failure.get('first'))
        v.identifier(failure.get('second'))
        v.text(failure.get('reason', ''), '失败原因', 300)
    reference = network.get('reference')
    if reference is not None:
        v.identifier(reference)
    return network


def _measurement(reference_image, moving_image, settings):
    """估计 moving 相对 reference 的平移，返回证据指标。

    位移约定 shift = p_moving - p_reference（两帧画布原点之差，即内容位移）。
    配准器矩阵 M 是把 moving 搬回 reference 的 warp，平移量与内容位移相反，
    故 shift = -M_t（ORB/ECC/MTB）；phaseCorrelate 直接给出内容位移 shift。
    对齐质量校验统一使用 warp 平移量 -shift。
    """
    method = settings['method']
    started = time.monotonic()
    if method == 'orb':
        matrix, details = reg.orb_align(reference_image, moving_image, settings)
        shift = (-matrix[0, 2], -matrix[1, 2])
        response = min(1., details['inliers'] / max(8, details['matches']))
        basis = {'method': 'orb', 'matches': details['matches'], 'inliers': details['inliers']}
    elif method == 'ecc':
        matrix, details = reg.ecc_align(reference_image, moving_image, settings)
        shift = (-matrix[0, 2], -matrix[1, 2])
        response = max(0., min(1., details['correlation']))
        basis = {'method': 'ecc'}
    elif method == 'mtb':
        matrix, details = reg.mtb_align(reference_image, moving_image)
        shift = (-matrix[0, 2], -matrix[1, 2])
        response = max(0., 1. - details['bitmap_mismatch'])
        basis = {'method': 'mtb'}
    else:
        surface_data = reg.phase_surface(reference_image, moving_image)
        shift = (surface_data['shift'][0], surface_data['shift'][1])
        response = surface_data['response']
        surface, w, h = surface_data['surface'], surface_data['width'], surface_data['height']
        # 主峰位置直接取相关面 argmax，避免位移符号换算歧义
        peak_y, peak_x = np.unravel_index(np.argmax(surface), surface.shape)
        primary = float(surface[peak_y, peak_x])
        # 主峰主瓣半径取短边的 8%，屏蔽主瓣后再找真正的竞争峰
        lobe = max(4, round(min(w, h) * .08))
        basis = {'method': 'phase',
                 'rivals': reg.rival_peaks(surface_data, neighborhood=(int(peak_x), int(peak_y), lobe)),
                 'primary': primary}
    # 同画布平移标定：f1=warp(f0,t) 时 phase shift=-t，故对齐 warp 平移量 = -shift
    quality = reg.alignment_quality(reference_image, moving_image,
                                    np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]]))
    ambiguity_ratio, alternatives = _ambiguity(basis, response)
    return {
        'shift_x': float(shift[0]), 'shift_y': float(shift[1]),
        'response': float(response), 'overlap': float(quality['overlap']),
        'correlation': float(max(0., quality['correlation'])),
        'normalized_mae': float(quality['normalized_mae']),
        'ambiguity_ratio': float(ambiguity_ratio),
        'alternatives': alternatives, 'details': basis,
        'seconds': time.monotonic() - started,
    }


def _ambiguity(basis, response):
    rivals = basis.get('rivals')
    if not rivals:
        return 0., []
    strongest = max(basis.get('primary', response), 1e-6)
    alternatives, best_ratio = [], 0.
    for item in rivals:
        ratio = item['strength'] / strongest
        best_ratio = max(best_ratio, ratio)
        alternatives.append([item['dx'], item['dy']])
    return best_ratio, alternatives[:3]


def _classify(measurement, settings, shape):
    """依据证据强度与几何合理性给关系定性。"""
    height, width = shape[:2]
    shift_magnitude = float(np.hypot(measurement['shift_x'], measurement['shift_y']))
    # 真实重叠的两帧，位移不可能达到或超过整帧对角线；超出说明相位峰是噪声
    if shift_magnitude >= np.hypot(width, height) * .98:
        return STATE_UNRELIABLE, '位移 %.0fpx 达到整帧尺度，判定无真实重叠' % shift_magnitude
    if measurement['overlap'] < settings['network_min_overlap']:
        return STATE_UNRELIABLE, '重叠覆盖率 %.0f%% 低于阈值 %.0f%%' % (
            measurement['overlap'] * 100, settings['network_min_overlap'] * 100)
    if measurement['response'] < settings['network_min_response']:
        return STATE_UNRELIABLE, '配准响应 %.2f 低于阈值 %.2f' % (
            measurement['response'], settings['network_min_response'])
    # 对齐后仍无相关，说明位移是噪声峰而非真实重叠，先于歧义判断
    if measurement['correlation'] < settings['network_min_correlation']:
        return STATE_UNRELIABLE, '对齐相关度 %.2f 低于阈值 %.2f，缺乏真实重叠证据' % (
            measurement['correlation'], settings['network_min_correlation'])
    if measurement['ambiguity_ratio'] >= settings['network_ambiguity_ratio']:
        return STATE_AMBIGUOUS, '存在强度 %.0f%% 的竞争位移，证据无法区分' % (
            measurement['ambiguity_ratio'] * 100)
    return STATE_PROPOSED, ''


def _confidence(measurement):
    # 以响应为主、重叠与相关度作温和修正，避免低纹理区域权重被过度压低
    response = max(measurement['response'], 1e-4)
    overlap = max(measurement['overlap'], 1e-4)
    correlation = max(measurement['correlation'], .05)
    return float(response * (0.5 + 0.5 * overlap) * (0.5 + 0.5 * min(1., correlation)))


def analyze_network(prepared, project, progress=None, cancelled=None):
    """测量全部启用帧之间的两两关系，复用未受影响的已有判断。"""
    settings = project['settings']['alignment']
    images, frames = prepared['images'], prepared['frames']
    ids = [frame['id'] for frame in frames]
    existing = project.get('alignment_network') or empty_network()
    lookup = {tuple(sorted((e['first'], e['second']))): e for e in existing.get('edges', [])}
    edges, failures = [], []
    total = len(ids) * (len(ids) - 1) // 2
    done = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if cancelled and cancelled():
                raise InterruptedError('视野网络分析已取消')
            first_id, second_id = ids[i], ids[j]  # 位移方向固定为 second 相对 first
            try:
                measurement = _measurement(images[i], images[j], settings)
            except v.Invalid as error:
                failures.append({'first': first_id, 'second': second_id, 'reason': str(error)})
                done += 1
                continue
            key = tuple(sorted((first_id, second_id)))
            kept = lookup.get(key)
            if kept is not None and kept.get('state') == STATE_EXCLUDED:
                # 保留人工排除状态，但端点方向与位移换算到当前帧顺序
                edge = dict(kept)
                edge.update(_directed(kept, first_id, second_id))
                edge.update({k: measurement[k] for k in measurement if k != 'details'})
                edge['details'] = measurement['details']
            elif kept is not None and _compatible(kept, measurement, first_id, second_id):
                edge = dict(kept)
                edge.update(_directed(kept, first_id, second_id))
                edge.update({k: measurement[k] for k in measurement if k != 'details'})
                edge['details'] = measurement['details']
            else:
                state, reason = _classify(measurement, settings, images[i].shape)
                edge = {
                    'id': edge_id(first_id, second_id), 'first': first_id, 'second': second_id,
                    'state': state, 'state_reason': reason,
                    'residual': None, **{k: measurement[k] for k in measurement if k != 'details'},
                }
            edge['confidence'] = _confidence(measurement)
            edges.append(edge)
            done += 1
            if progress:
                progress(.15 + .7 * done / max(total, 1), '分析视野重叠 %d/%d' % (done, max(total, 1)))
    network = {
        'fingerprint': frame_fingerprint(project), 'reference': project['reference'],
        'edges': edges, 'failures': failures, 'analyzed_at': v.timestamp(),
    }
    return network, {'failures': failures, 'pair_count': total}


def _directed(kept, first_id, second_id):
    """把历史边的位移方向换算为 second 相对 first。"""
    if kept['first'] == first_id and kept['second'] == second_id:
        return {}
    # 方向相反：位移取反，端点互换
    return {'first': first_id, 'second': second_id,
            'shift_x': -kept['shift_x'], 'shift_y': -kept['shift_y'],
            'alternatives': [[-x, -y] for x, y in kept.get('alternatives', [])]}


def _compatible(kept, measurement, first_id, second_id):
    # 源图未变（指纹一致才会复用）时，按当前方向换算后新旧位移偏差在 3 像素内视为同一关系
    sx = kept['shift_x'] if kept['first'] == first_id else -kept['shift_x']
    sy = kept['shift_y'] if kept['first'] == first_id else -kept['shift_y']
    return abs(sx - measurement['shift_x']) <= 3 and abs(sy - measurement['shift_y']) <= 3


def _adjacency(edges, ids, states):
    graph = {frame_id: {} for frame_id in ids}
    for edge in edges:
        if edge['state'] not in states:
            continue
        if edge['first'] not in graph or edge['second'] not in graph:
            continue
        graph[edge['first']][edge['second']] = edge
        graph[edge['second']][edge['first']] = edge
    return graph


def _components(graph, ids, reference):
    """在可用关系上求连通分量与到参考帧的最短关系路径。"""
    connected, paths = set(), {}
    if reference in graph:
        # 参考帧即使没有任何关系，也自身构成锚点
        queue = [reference]
        connected.add(reference)
        paths[reference] = []
        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor in connected:
                    continue
                connected.add(neighbor)
                paths[neighbor] = paths[current] + [graph[current][neighbor]['id']]
                queue.append(neighbor)
    components, visited = [], set()
    for frame_id in ids:
        if frame_id in visited:
            continue
        stack, members = [frame_id], []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            members.append(node)
            stack.extend(graph[node].keys())
        components.append(sorted(members))
    return connected, paths, components


def solve_network(network, project):
    """以参考帧为锚的带权全局平移最小二乘，并标记环闭合冲突。

    只有 proposed/conflict 关系参与定位与连通判定；ambiguous/unreliable/excluded
    关系仍完整返回供人工查看，但不把帧算作已定位。
    """
    settings = project['settings']['alignment']
    frames = [f for f in project['frames'] if f['enabled']]
    ids = [f['id'] for f in frames]
    index = {frame_id: i for i, frame_id in enumerate(ids)}
    reference = project['reference']
    edges = [dict(edge) for edge in network.get('edges', [])
             if edge['first'] in index and edge['second'] in index]
    graph = _adjacency(edges, ids, SOLVING_STATES)
    connected, paths, components = _components(graph, ids, reference)

    positions = {frame_id: None for frame_id in ids}
    conflict_threshold = settings['network_conflict_pixels']
    conflicts = []
    if reference in index:
        positions[reference] = np.zeros(2, np.float64)
        solving = [e for e in edges if e['state'] in SOLVING_STATES]
        unknown = [frame_id for frame_id in ids if frame_id != reference and frame_id in connected]
        column = {frame_id: k for k, frame_id in enumerate(unknown)}
        n = len(unknown)
        ata = np.zeros((n, n), np.float64)
        atb = np.zeros((n, 2), np.float64)
        for edge in solving:
            a, b = edge['first'], edge['second']
            if a not in connected or b not in connected:
                continue
            weight = max(edge['confidence'], 1e-5) ** 2
            # 残差 r = (p_b - p_a) - shift；最小化 sum w*r^2
            measured = np.array([edge['shift_x'], edge['shift_y']], np.float64)
            coefficient = {a: -1., b: 1.}
            cols = [(column[node], sign) for node, sign in coefficient.items() if node != reference]
            for col, sign in cols:
                ata[col, col] += weight * sign * sign
            if len(cols) == 2:
                (col_a, sign_a), (col_b, sign_b) = cols
                ata[col_a, col_b] += weight * sign_a * sign_b
                ata[col_b, col_a] += weight * sign_a * sign_b
            # 右端 atb += w * sign * measured；参考帧列缺失时其贡献为 0
            for col, sign in cols:
                atb[col] += weight * sign * measured
        if n:
            solved, *_ = np.linalg.lstsq(ata, atb, rcond=None)
            for frame_id, col in column.items():
                positions[frame_id] = solved[col]
        for edge in solving:
            a, b = edge['first'], edge['second']
            if positions[a] is None or positions[b] is None:
                continue
            predicted = positions[b] - positions[a]
            residual = float(np.linalg.norm(predicted - np.array([edge['shift_x'], edge['shift_y']])))
            edge['residual'] = round(residual, 3)
            if residual > conflict_threshold:
                conflicts.append(edge['id'])

    conflict_set = set(conflicts)
    for edge in edges:
        if edge['id'] in conflict_set and edge['state'] == STATE_PROPOSED:
            edge['state'] = STATE_CONFLICT
            edge['state_reason'] = '环闭合残差 %.1f 像素，超过阈值 %.1f，请排除或确认' % (
                edge['residual'], conflict_threshold)
        elif edge['id'] not in conflict_set and edge['state'] == STATE_CONFLICT:
            edge['state'] = STATE_PROPOSED
            edge['state_reason'] = ''
    state_lookup = {edge['id']: edge for edge in edges}
    for original in network['edges']:
        if original['id'] in state_lookup:
            original.update({k: state_lookup[original['id']][k] for k in ('state', 'state_reason', 'residual')})

    # 全部待确认关系都返回给前端可见；其中只有两端已被可靠关系锚定的才会真正污染
    # 已定位帧、阻塞应用。仅靠歧义关系连接的帧本就保持未定位，不阻塞其余帧。
    all_ambiguous = [e['id'] for e in network['edges'] if e['state'] == STATE_AMBIGUOUS]
    reliable_graph = _adjacency(network['edges'], ids, (STATE_PROPOSED, STATE_CONFLICT))
    reliable_anchored = set()
    if reference in reliable_graph:
        stack = [reference]
        reliable_anchored.add(reference)
        while stack:
            node = stack.pop()
            for neighbor in reliable_graph[node]:
                if neighbor not in reliable_anchored:
                    reliable_anchored.add(neighbor)
                    stack.append(neighbor)
    pending_ambiguous = [edge_id for edge_id in all_ambiguous
                         if (next(e for e in network['edges'] if e['id'] == edge_id)['first']
                             in reliable_anchored
                             and next(e for e in network['edges'] if e['id'] == edge_id)['second']
                             in reliable_anchored)]
    nodes = [{
        'frame_id': frame_id,
        'name': next(f['name'] for f in frames if f['id'] == frame_id),
        'connected': frame_id in connected,
        'is_reference': frame_id == reference,
        'position': positions[frame_id].tolist() if positions[frame_id] is not None else None,
        'path_edges': paths.get(frame_id, []),
        'component': next(k for k, members in enumerate(components) if frame_id in members),
    } for frame_id in ids]
    return {
        'nodes': nodes,
        'edges': network['edges'],
        'components': [{'index': k, 'frame_ids': members, 'anchored': reference in members}
                       for k, members in enumerate(components)],
        'connected_ids': sorted(connected),
        'disconnected_ids': [frame_id for frame_id in ids if frame_id not in connected],
        'conflict_ids': sorted(conflict_set),
        'ambiguous_ids': all_ambiguous,
        'blocking_ambiguous_ids': pending_ambiguous,
        'positions': {frame_id: (value.tolist() if value is not None else None)
                      for frame_id, value in positions.items()},
        'ready': not conflicts and not pending_ambiguous,
    }


def build_matrices(report, project, width, height):
    """把确认后的全局平移写回各帧归一化矩阵；未连通帧保持原矩阵。"""
    if not report['ready']:
        raise v.Invalid('仍存在冲突或待确认关系，无法应用网络配准')
    matrices = {}
    for node in report['nodes']:
        frame = model.find_frame(project, node['frame_id'])
        if node['position'] is None:
            matrices[node['frame_id']] = frame['matrix']
            continue
        x, y = node['position']
        # position 即该帧在参考画布上的原点位移，warp 平移量与之一致（平移模型）
        pixel = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)
        matrices[node['frame_id']] = normalized_matrix(pixel, width, height)
    return matrices
