"""追踪结果交付：真实时间与标定下的位置/面积、可靠段速度与事件关系。

四个 CSV 均为 UTF-8 BOM，便于电子表格直接打开；速度仅统计 reliable=true 的逐帧
确认边，合并团的位移以 is_event_blob=true 单列、不混入单体速度。
"""
import io
import json
import zipfile
from .imaging import png
from .exporting import csv_bytes
from .track_render import render_tracking_frame


def position_rows(derived):
    rows = []
    for frame in derived['frames']:
        for node in frame['nodes']:
            rows.append({'frame_index': frame['index'], 'time_s': frame['timestamp'],
                         'field_id': frame['field_id'], 'run_id': frame['run_id'],
                         'node_id': node['node_id'], 'object_id_in_run': node['oid'],
                         'track': node['track'] or '', 'lineage': node['lineage'] or '',
                         'role': node['role'], 'x_px': node['cx_px'], 'y_px': node['cy_px'],
                         'x': node['x'], 'y': node['y'],
                         'area_px': node['area_px'], 'area': node['area'],
                         'equivalent_diameter': node['diameter'],
                         'circularity': node['circularity'],
                         'mean_intensity': node['mean_intensity'], 'unit': frame['unit']})
    rows.sort(key=lambda r: (r['time_s'], r['node_id']))
    return rows


def edge_rows(derived):
    rows = []
    for edge in derived['edges']:
        rows.append({'edge_id': edge['id'], 'src_node': edge['src'], 'dst_node': edge['dst'],
                     'src_frame': edge['src_frame'], 'dst_frame': edge['dst_frame'],
                     'dt_s': edge['dt'], 'distance': edge['distance'],
                     'speed': edge['speed'], 'vx': edge['vx'], 'vy': edge['vy'],
                     'status': edge['status'], 'origin': edge['origin'],
                     'frame_gap': edge['frame_gap'], 'reliable': edge['reliable'],
                     'unreliable_reason': edge['reason'],
                     'score': '' if edge['score'] is None else edge['score'],
                     'unit': derived['frames'][0]['unit']})
    rows.sort(key=lambda r: (r['src_frame'], r['dst_node']))
    return rows


def segment_rows(derived):
    rows = []
    for track in derived['tracks']:
        for seg in track['reliable_segments']:
            rows.append({'track': track['name'], 'start_node': seg['src'], 'end_node': seg['dst'],
                         'start_time_s': seg['start_time'], 'end_time_s': seg['end_time'],
                         'points': seg['points'], 'length': seg['length'],
                         'mean_speed': seg['mean_speed'], 'max_speed': seg['max_speed'],
                         'straightness': seg['straightness'],
                         'is_event_blob': track['is_event_blob'],
                         'unit': derived['frames'][0]['unit']})
    return rows


def event_rows(derived):
    rows = []
    for event in derived['events']:
        rows.append({'kind': event['kind'],
                     'time_s': '' if event['time'] is None else event['time'],
                     'node_or_edge': event['node'], 'track': event.get('track') or '',
                     'members': '|'.join(event['members']),
                     'missing_frames': '|'.join(map(str, event.get('missing_frames', []))),
                     'status': event['status']})
    return rows


def track_rows(derived):
    rows = []
    for track in derived['tracks']:
        rows.append({'track': track['name'], 'lineage': track['lineage'],
                     'start_time_s': track['start_time'], 'end_time_s': track['end_time'],
                     'points': track['point_count'], 'gap_edges': track['gap_count'],
                     'path_length': track['path_length'],
                     'net_displacement': track['net_displacement'],
                     'straightness': track['straightness'],
                     'reliable_mean_speed': track['mean_speed'],
                     'reliable_max_speed': track['max_speed'],
                     'is_event_blob': track['is_event_blob'],
                     'unit': derived['frames'][0]['unit']})
    return rows


def tracking_zip(store, session, derived):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('positions.csv', csv_bytes(position_rows(derived)))
        archive.writestr('links_velocities.csv', csv_bytes(edge_rows(derived)))
        archive.writestr('reliable_segments.csv', csv_bytes(segment_rows(derived)))
        archive.writestr('events.csv', csv_bytes(event_rows(derived)))
        archive.writestr('tracks.csv', csv_bytes(track_rows(derived)))
        archive.writestr('tracking.json', json.dumps(derived, ensure_ascii=False, allow_nan=False))
        for index, frame in enumerate(derived['frames']):
            overlay = render_tracking_frame(store, derived, index)
            archive.writestr('overlays/frame_%02d.png' % index, png(overlay))
    return stream.getvalue()
