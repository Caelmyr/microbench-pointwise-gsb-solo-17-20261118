/* 跨帧轨迹核对面板：会话构建、叠加图点选、候选确认/驳回、手动改接与导出。 */
let tdoc = null, tid = null, frameIndex = 0, trackPicture = null;
let picked = [];
const trackCanvas = $('trackCanvas'), tctx = trackCanvas.getContext('2d');

function tmessage(text) { $('trackMessage').textContent = text; }
function tbase() { return base() + '/trackings'; }

async function syncProject() { doc = await api(base()); renderRunPicker(); renderSessions(); }

function renderRunPicker() {
  if (!doc) return;
  const runs = [...doc.runs].sort((a, b) => a.timestamp - b.timestamp);
  $('runPicker').innerHTML = runs.map(r => {
    const field = doc.fields.find(f => f.id === r.field_id);
    const stale = field && field.active_run && field.active_run !== r.id;
    return `<label class="runRow check"><input type="checkbox" class="runCheck" value="${r.id}">
      <span>${escapeHTML(r.name)} · t=${r.timestamp}s · 对象${r.summary.count}
      ${stale ? '<em class="staleTag">非当前结果</em>' : ''}</span></label>`;
  }).join('') || '<p class="muted">请先对各视野执行“应用设置并分析”。</p>';
}

function renderSessions() {
  const select = $('sessionSelect');
  const list = doc?.trackings || [];
  select.innerHTML = '<option value="">选择追踪会话</option>' + list.map(s =>
    `<option value="${s.id}">${escapeHTML(s.name)} · ${s.frames}帧 · 候选${s.pending}${s.stale_frames ? ` · ${s.stale_frames}帧已过期` : ''}</option>`).join('');
  if (tid) select.value = tid;
}

async function loadTracking(id) {
  if (!id) { $('trackWork').hidden = true; tdoc = null; tid = null; return; }
  tid = id;
  tdoc = await api(tbase() + '/' + id);
  frameIndex = Math.min(frameIndex, tdoc.frames.length - 1);
  picked = [];
  $('trackWork').hidden = false;
  renderSessions();
  await renderTracking();
}

async function renderTracking() {
  if (!tdoc) return;
  const stale = tdoc.frames.filter(f => f.stale).length;
  tmessage(stale ? `${stale} 个帧的源视野已重新分析；本会话仍固定引用旧结果，可继续核对与导出，确认不会套用到新对象编号。`
                 : `谱系 ${tdoc.lineage_count} · 轨迹 ${tdoc.track_count} · 待核对 ${tdoc.pending_groups.length} 组`);
  $('framePos').textContent = `第 ${frameIndex + 1}/${tdoc.frames.length} 帧 · ${escapeHTML(tdoc.frames[frameIndex].name)} · t=${tdoc.frames[frameIndex].timestamp}s`;
  $('trackingDownload').href = tbase() + '/' + tid + '/tracking.zip';
  const url = tbase() + '/' + tid + '/frame/' + frameIndex + '.png?rejected=' + ($('showRejected').checked ? 1 : 0);
  const img = new Image();
  await new Promise((res, rej) => { img.onload = res; img.onerror = () => rej(Error('叠加图读取失败')); img.src = url + '&r=' + doc.revision; });
  trackPicture = img;
  trackCanvas.width = img.width;
  trackCanvas.height = img.height;
  drawTrackCanvas();
  renderPending();
  renderEvents();
  renderTracks();
  renderLinkBox();
}

function drawTrackCanvas() {
  if (!trackPicture) return;
  tctx.drawImage(trackPicture, 0, 0);
  const frame = tdoc.frames[frameIndex];
  for (const node of frame.nodes) {
    if (!picked.includes(node.node_id)) continue;
    tctx.lineWidth = 2;
    tctx.strokeStyle = '#ffffff';
    tctx.beginPath();
    tctx.arc(node.cx_px, node.cy_px, 11, 0, Math.PI * 2);
    tctx.stroke();
  }
}

function renderPending() {
  $('pendingCount').textContent = `（${tdoc.pending_groups.length} 组）`;
  if (!tdoc.pending_groups.length) { $('pendingList').innerHTML = '<p class="muted">无待核对关联。</p>'; return; }
  $('pendingList').innerHTML = tdoc.pending_groups.map((group, gi) => {
    const rows = group.edges.map(eid => {
      const e = tdoc.edges.find(x => x.id === eid);
      if (e.status !== 'pending') return '';
      return `<tr><td>${e.src}→${e.dst}</td><td>${e.frame_gap > 1 ? '漏检' + (e.frame_gap - 1) + '帧' : '相邻'}</td>
      <td>${e.score === null ? '手动' : e.score.toFixed(2)}</td>
      <td><button data-link="confirm" data-src="${e.src}" data-dst="${e.dst}">确认</button>
      <button data-link="reject" data-src="${e.src}" data-dst="${e.dst}">驳回</button></td></tr>`;
    }).join('');
    const pendingPairs = group.edges.map(eid => tdoc.edges.find(x => x.id === eid))
      .filter(e => e.status === 'pending')
      .map(e => ({ src: e.src, dst: e.dst }));
    const multi = pendingPairs.length >= 2;
    const groupBtn = multi ? `<button class="primary" data-group="${gi}">整组确认（合并/分裂，保留多条关联）</button>` : '';
    return `<table><thead><tr><th>候选（帧${group.frame_span[0]}→${group.frame_span[1]}）</th><th>类型</th><th>分数</th><th>操作</th></tr></thead><tbody>${rows}</tbody></table>${groupBtn}`;
  }).join('');
  document.querySelectorAll('button[data-group]').forEach(btn => {
    btn.onclick = () => action(async () => {
      const group = tdoc.pending_groups[+btn.dataset.group];
      const pairs = group.edges.map(eid => tdoc.edges.find(x => x.id === eid))
        .filter(e => e.status === 'pending').map(e => ({ src: e.src, dst: e.dst }));
      await api(tbase() + '/' + tid + '/links', { revision: doc.revision, action: 'confirm_group', pairs });
      await syncProject();
      await loadTracking(tid);
    });
  });
}

function renderEvents() {
  const label = { merge: '合并', split: '分裂', gap: '漏检接续' };
  if (!tdoc.events.length) { $('eventList').innerHTML = '<p class="muted">无事件。</p>'; return; }
  $('eventList').innerHTML = '<table><thead><tr><th>事件</th><th>时刻/帧</th><th>对象</th><th>状态</th></tr></thead><tbody>' +
    tdoc.events.map(e => `<tr><td>${label[e.kind]}</td>
      <td>${e.time === null ? (e.missing_frames || []).join(',') : e.time + 's'}</td>
      <td>${escapeHTML(e.kind === 'gap' ? e.members.join(' → ') : e.node)}</td>
      <td>${e.status === 'confirmed' ? '已确认' : '待核对'}</td></tr>`).join('') + '</tbody></table>';
}

function renderTracks() {
  $('trackList').innerHTML = '<table><thead><tr><th>轨迹</th><th>时段s</th><th>点数</th><th>事件团</th><th>均速</th><th>峰速</th><th>可靠段</th></tr></thead><tbody>' +
    tdoc.tracks.map(t => `<tr><td>${escapeHTML(t.name)}</td><td>${t.start_time}–${t.end_time}</td><td>${t.point_count}</td>
      <td>${t.is_event_blob ? '是' : ''}</td>
      <td>${t.mean_speed === null ? '—' : t.mean_speed.toFixed(2)}</td>
      <td>${t.max_speed === null ? '—' : t.max_speed.toFixed(2)}</td>
      <td>${t.reliable_segments.length}</td></tr>`).join('') + '</tbody></table>';
}

function renderLinkBox() {
  const frame = tdoc.frames[frameIndex];
  if (!picked.length) { $('nodeInfo').textContent = '点击叠加图上的对象节点选择；选择两个不同帧的节点后可手动建立关联。'; $('linkBox').innerHTML = ''; return; }
  const nodes = picked.map(id => tdoc.frames.flatMap(f => f.nodes).find(n => n.node_id === id));
  $('nodeInfo').textContent = nodes.map(n => `${n.node_id}（轨迹 ${n.track || '未命名'} · 帧${tdoc.frames.findIndex(f => f.nodes.includes(n))} · x ${n.x.toFixed(1)} / y ${n.y.toFixed(1)}）`).join('　');
  const incident = tdoc.edges.filter(e => picked.includes(e.src) || picked.includes(e.dst));
  let html = '<table><thead><tr><th>关联</th><th>状态</th><th>速度</th><th>操作</th></tr></thead><tbody>' +
    incident.map(e => `<tr><td>${e.src}→${e.dst}</td><td>${({ confirmed: '已确认', pending: '待核对', rejected: '已驳回' })[e.status]}</td>
      <td>${e.speed === null ? '—' : e.speed.toFixed(2)}${e.reliable ? '' : '（不计可靠速度）'}</td>
      <td>${e.status !== 'confirmed' ? `<button data-link="confirm" data-src="${e.src}" data-dst="${e.dst}">确认</button>` : ''}
      ${e.status !== 'rejected' ? `<button data-link="reject" data-src="${e.src}" data-dst="${e.dst}">驳回</button>` : ''}
      ${e.origin === 'manual' ? `<button data-link="delete" data-src="${e.src}" data-dst="${e.dst}">删除</button>` : ''}</td></tr>`).join('') + '</tbody></table>';
  if (picked.length === 2) {
    const frames = picked.map(id => Number(id.split('-')[0].slice(1)));
    if (frames[0] !== frames[1]) {
      const early = picked[frames[0] < frames[1] ? 0 : 1];
      const late = picked[frames[0] < frames[1] ? 1 : 0];
      html += `<button id="manualAdd" class="primary">手动确认关联 ${early} → ${late}</button>`;
    }
  }
  html += ' <button id="clearPicked">清除选择</button>';
  $('linkBox').innerHTML = html;
  $('clearPicked').onclick = () => { picked = []; drawTrackCanvas(); renderLinkBox(); };
  const addBtn = $('manualAdd');
  if (addBtn) addBtn.onclick = () => action(async () => {
    const frames2 = picked.map(id => Number(id.split('-')[0].slice(1)));
    const early = picked[frames2[0] < frames2[1] ? 0 : 1];
    const late = picked[frames2[0] < frames2[1] ? 1 : 0];
    await postLink('add', early, late);
  });
}

async function postLink(kind, src, dst) {
  await api(tbase() + '/' + tid + '/links', { revision: doc.revision, action: kind, src, dst });
  await syncProject();
  await loadTracking(tid);
}

trackCanvas.addEventListener('click', event => {
  if (!tdoc) return;
  const rect = trackCanvas.getBoundingClientRect();
  const px = (event.clientX - rect.left) * trackCanvas.width / rect.width;
  const py = (event.clientY - rect.top) * trackCanvas.height / rect.height;
  const frame = tdoc.frames[frameIndex];
  let best = null, bestD = 14;
  for (const node of frame.nodes) {
    const d = Math.hypot(node.cx_px - px, node.cy_px - py);
    if (d < bestD) { bestD = d; best = node; }
  }
  if (!best) return;
  if (picked.includes(best.node_id)) picked = picked.filter(id => id !== best.node_id);
  else { picked.push(best.node_id); if (picked.length > 2) picked.shift(); }
  drawTrackCanvas();
  renderLinkBox();
});

document.addEventListener('click', event => {
  const btn = event.target.closest('button[data-link]');
  if (!btn || !tdoc) return;
  action(() => postLink(btn.dataset.link, btn.dataset.src, btn.dataset.dst));
});

$('buildTracking').onclick = () => action(async () => {
  const run_ids = Array.from(document.querySelectorAll('.runCheck:checked')).map(c => c.value);
  if (run_ids.length < 2) throw Error('请至少勾选2个分析结果');
  const payload = { revision: doc.revision, name: $('tgName').value || '追踪会话', run_ids,
    params: { max_gap: +$('tgGap').value, gate_diameters: +$('tgGate').value,
              area_ratio: +$('tgArea').value, event_area_ratio: +$('tgEventArea').value } };
  const created = await api(tbase(), payload);
  await syncProject();
  frameIndex = 0;
  await loadTracking(created.id);
});

$('sessionSelect').onchange = () => action(async () => { frameIndex = 0; await loadTracking($('sessionSelect').value); });
$('framePrev').onclick = () => action(async () => { if (frameIndex > 0) { frameIndex--; await renderTracking(); } });
$('frameNext').onclick = () => action(async () => { if (frameIndex < tdoc.frames.length - 1) { frameIndex++; await renderTracking(); } });
$('showRejected').onchange = () => action(renderTracking);
$('trackingDemo').onclick = () => action(async () => {
  const value = await api('/bench/tracking-demo', {});
  await accept(value);
  renderRunPicker();
  renderSessions();
  if (doc.trackings.length) { frameIndex = 0; await loadTracking(doc.trackings[0].id); }
  $('trackingPanel').scrollIntoView({ behavior: 'smooth' });
});

// 工程切换或分析后刷新选择器
const originalAccept = accept;
accept = async function (value) {
  await originalAccept(value);
  renderRunPicker();
  renderSessions();
  if (tid && (!doc.trackings.some(s => s.id === tid))) { $('trackWork').hidden = true; tdoc = null; tid = null; }
};
