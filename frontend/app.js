const $ = id => document.getElementById(id);
const direct = location.port === '8001';
const bases = direct ? {query: '/api/v1', import: `${location.protocol}//${location.hostname}:8000/api/v1`} : {query:'/query/api/v1', import:'/import/api/v1'};
let csrf = sessionStorage.getItem('kb_csrf') || '', kbs = [], versions = [], chatId = null, messageId = null, taskId = null, streaming = false;
function notice(text) { $('notice').textContent = text; $('notice').style.display = 'block'; }
function clear(node) { node.replaceChildren(); }
function item(text, tag = 'div') { const node = document.createElement(tag); node.textContent = text; return node; }
async function api(role, path, options = {}) {
  const {body, method = 'GET'} = options;
  const headers = {'X-CSRF-Token':csrf};
  if (method !== 'GET') headers['Idempotency-Key'] = crypto.randomUUID();
  if (body && !(body instanceof FormData)) headers['Content-Type'] = 'application/json';
  const response = await fetch(bases[role] + path, {method, credentials:'include', headers, body:body instanceof FormData ? body : body ? JSON.stringify(body) : undefined});
  const payload = await response.json();
  if (!response.ok) throw new Error(`${payload.error?.code || response.status}：${payload.error?.message || '请求失败'}`);
  return payload.data;
}
function safe(handler) { return async event => { try { await handler(event); } catch(error) { notice(error.message); } }; }
function choose(select, rows, label) {
  clear(select);
  for (const row of rows) { const option = item(label(row), 'option'); option.value = row._id; select.append(option); }
}
function kbId() { if (!$('kb').value) throw new Error('当前没有可访问的知识库'); return $('kb').value; }
function docId() { if (!$('document').value) throw new Error('请先选择文档'); return $('document').value; }
function versionId() { if (!$('version').value) throw new Error('请先选择版本'); return $('version').value; }
async function loadKbs() { kbs = await api('query','/knowledge-bases'); choose($('kb'),kbs,k => k.name); await loadDocuments(); }
async function loadDocuments() {
  const kb = kbs.find(k => k._id === $('kb').value);
  if (!kb) return;
  $('role').textContent = `角色：${kb.role}`;
  $('admin-panel').hidden = kb.role !== 'admin';
  $('upload').hidden = !['maintainer','admin'].includes(kb.role);
  $('submit').hidden = !['maintainer','admin'].includes(kb.role);
  for (const id of ['approve','reject','publish','withdraw','delete','rollback']) $(id).hidden = !['reviewer','admin'].includes(kb.role);
  $('parser-egress').checked = kb.allow_parser_egress;
  $('llm-egress').checked = kb.allow_llm_egress;
  $('policy-summary').textContent = `解析外发：${kb.allow_parser_egress?'允许':'关闭'} · 生成外发：${kb.allow_llm_egress?'允许':'关闭'}`;
  choose($('document'),await api('import',`/knowledge-bases/${kbId()}/documents`),d => d.title || d._id);
  await loadVersions();
}
async function loadVersions() {
  versions = $('document').value ? await api('import',`/documents/${docId()}/versions`) : [];
  versions.sort((a,b)=>b.revision-a.revision);
  choose($('version'), versions, v => `v${v.revision} · ${v.state} · ${v.filename}`);
  showVersion();
}
function showVersion() { const v = versions.find(v => v._id === $('version').value); $('version-status').textContent = v ? `${v._id} · 解析状态：${v.processing_status}` : '暂无资料'; }
async function showTask(id) { taskId=id; $('task-panel').hidden=false; $('task-id').textContent=id; await checkTask(); }
async function checkTask() { const t=await api('import',`/tasks/${taskId}`); $('task-status').textContent=`${t.kind} · ${t.status} · 尝试 ${t.attempts}${t.error_code?' · '+t.error_code:''}`; $('task-retry').disabled=t.status!=='failed'; }
async function history() {
  if (!chatId) return;
  const data = await api('query',`/conversations/${chatId}`); clear($('history'));
  for (const m of data.messages) $('history').append(item(m.status==='access_revoked'?'该条记录的资料访问权限已变化。':`问：${m.question}\n答：${m.answer || ''}\n状态：${m.status}`,'article'));
}
function sources(rows, chunks = false) {
  clear($('sources'));
  rows.forEach((source,index)=>{
    const box=item(`${source.label || `S${index+1}`} · ${source.heading} · ${JSON.stringify(source.locator)}`); box.className='source';
    if (chunks) box.append(item(source.text));
    const read=item('核对原文','button'); read.type='button'; read.addEventListener('click',safe(async()=>{ const c=await api('query',`/citations/${source.chunk_id || source._id}`); $('citation').hidden=false; $('citation').textContent=JSON.stringify(c.locator,null,2)+'\n\n'+c.text; }));
    const download=item('下载来源文件','button'); download.type='button'; download.addEventListener('click',()=>{ const a=document.createElement('a'); a.href=bases.query+`/versions/${source.version_id}/source`; a.click(); });
    box.append(read,download); $('sources').append(box);
  });
}
function consumeFrame(frame) {
  const kind=frame.split('\n').find(l=>l.startsWith('event:'))?.slice(6).trim();
  const data=frame.split('\n').filter(l=>l.startsWith('data:')).map(l=>l.slice(5).trim()).join('\n');
  if (!data) return;
  const payload=JSON.parse(data); messageId=payload.message_id;
  if(kind==='accepted') $('cancel').disabled=false;
  if(kind==='delta') $('answer').textContent+=payload.text;
  if(kind==='sources') sources(payload.sources);
  if(kind==='status') $('answer-status').textContent=payload.stage;
  if(kind==='warning') $('answer-status').textContent=`提示：${payload.message || payload.code}`;
  if(kind==='done') { $('answer-status').textContent=`已结束：${payload.status}`; $('feedback-panel').hidden=false; }
  if(['error','cancelled','access_revoked'].includes(kind)) {
    $('answer-status').textContent=`已中止：${payload.code || kind}`;
    if(kind==='access_revoked') { $('answer').textContent='资料访问权限已变化。'; clear($('sources')); $('citation').hidden=true; }
  }
}
$('login').addEventListener('submit',safe(async event=>{
  event.preventDefault(); const loginForm=event.currentTarget; const form=new FormData(loginForm);
  const result=await api('query','/auth/login',{method:'POST',body:{login:form.get('login'),password:form.get('password')}});
  csrf=result.csrf_token; sessionStorage.setItem('kb_csrf',csrf); loginForm.reset(); await enter();
}));
async function enter() { const me=await api('query','/auth/me'); $('identity').textContent=me.login; $('login-panel').hidden=true; $('workspace').hidden=false; $('logout').hidden=false; await loadKbs(); }
$('logout').addEventListener('click',safe(async()=>{await api('query','/auth/logout',{method:'POST'}); sessionStorage.removeItem('kb_csrf'); location.reload();}));
$('kb').addEventListener('change',safe(async()=>{chatId=null;clear($('history'));await loadDocuments();}));
$('refresh').addEventListener('click',safe(loadDocuments));
$('document').addEventListener('change',safe(loadVersions)); $('version').addEventListener('change',showVersion);
$('upload').addEventListener('submit',safe(async event=>{event.preventDefault();const form=new FormData(event.currentTarget);const payload=new FormData();payload.set('file',form.get('file'));payload.set('metadata',JSON.stringify({title:form.get('title')}));const path=form.get('newVersion')?`/documents/${docId()}/versions`:`/knowledge-bases/${kbId()}/documents`;const result=await api('import',path,{method:'POST',body:payload});await showTask(result.task_id);await loadDocuments();notice('上传成功，任务已入队。');}));
$('preview').addEventListener('click',safe(async()=>{const p=await api('import',`/versions/${versionId()}/preview`);$('preview-content').textContent=p.chunks.map(c=>`${c.heading}\n${JSON.stringify(c.locator)}\n${c.text}`).join('\n\n');}));
for(const [id,verb,path,body] of [
  ['submit','POST',()=>`/versions/${versionId()}/submit`,()=>({})],
  ['approve','POST',()=>`/versions/${versionId()}/reviews`,()=>({decision:'approved',reason:$('review-reason').value})],
  ['reject','POST',()=>`/versions/${versionId()}/reviews`,()=>({decision:'rejected',reason:$('review-reason').value})],
  ['publish','POST',()=>`/versions/${versionId()}/publish`,()=>({})],
  ['rollback','POST',()=>`/documents/${docId()}/rollback`,()=>({version_id:versionId()})],
  ['withdraw','POST',()=>`/documents/${docId()}/withdraw`,()=>({})],
  ['delete','DELETE',()=>`/documents/${docId()}`,()=>undefined]]) {
  $(id).addEventListener('click',safe(async()=>{if(['delete','withdraw'].includes(id)&&!confirm(id==='delete'?'删除文档会清理其全部版本与依赖内容，确认继续？':'下架后资料将立即停止提供，确认继续？'))return;const result=await api('import',path(),{method:verb,body:body()});if(result.task_id || result.task_ids?.length)await showTask(result.task_id || result.task_ids[0]);await loadDocuments();notice(result.task_ids?.length ? `操作已提交，${result.task_ids.length} 个清理任务：${result.task_ids.join(', ')}` : '操作已提交。');}));
}
$('task-check').addEventListener('click',safe(checkTask)); $('task-retry').addEventListener('click',safe(async()=>{await api('import',`/tasks/${taskId}/retry`,{method:'POST',body:{}});await checkTask();}));
$('new-chat').addEventListener('click',()=>{chatId=null;messageId=null;clear($('history'));$('answer').textContent='';clear($('sources'));$('feedback-panel').hidden=true;});
$('ask').addEventListener('submit',safe(async event=>{
  event.preventDefault();if(streaming)return;const question=$('question').value;
  $('answer').textContent='';clear($('sources'));$('citation').hidden=true;$('feedback-panel').hidden=true;
  if(event.submitter?.value==='search'){const result=await api('query','/search',{method:'POST',body:{kb_ids:[kbId()],question,mode:'hybrid',rerank:true,limit:5}});sources(result.candidates,true);$('answer-status').textContent=`检索 ${result.candidates.length} 条 · ${result.degraded.join(' / ')}`;return;}
  if(!chatId)chatId=(await api('query','/conversations',{method:'POST',body:{kb_ids:[kbId()]}}))._id;
  streaming=true;messageId=null;$('cancel').disabled=true;$('kb').disabled=true;$('new-chat').disabled=true;$('answer-status').textContent='正在检索';
  try {
    const response=await fetch(bases.query+`/conversations/${chatId}/answers`,{method:'POST',credentials:'include',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf,'Idempotency-Key':crypto.randomUUID()},body:JSON.stringify({question})});
    if(!response.ok){const data=await response.json();throw new Error(data.error?.message || '生成请求失败');}
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    while(true){const {done,value}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done}).replace(/\r\n/g,'\n');let boundary;while((boundary=buffer.indexOf('\n\n'))>=0){consumeFrame(buffer.slice(0,boundary));buffer=buffer.slice(boundary+2);}if(done)break;}
    await history();
  } finally {streaming=false;$('cancel').disabled=true;$('kb').disabled=false;$('new-chat').disabled=false;}
}));
$('cancel').addEventListener('click',safe(async()=>{if(messageId)await api('query',`/requests/${messageId}/cancel`,{method:'POST',body:{}});}));
$('feedback').addEventListener('submit',safe(async event=>{event.preventDefault();const f=await api('query','/feedback',{method:'POST',body:{message_id:messageId,category:$('feedback-type').value,detail:$('feedback-note').value}});notice(`反馈已登记：${f._id}`);}));
$('policy').addEventListener('submit',safe(async event=>{event.preventDefault();await api('import',`/knowledge-bases/${kbId()}`,{method:'PATCH',body:{allow_parser_egress:$('parser-egress').checked,allow_llm_egress:$('llm-egress').checked}});await loadKbs();notice('策略已保存。');}));
$('member').addEventListener('submit',safe(async event=>{event.preventDefault();const role=$('member-role').value;await api('import',`/knowledge-bases/${kbId()}/members/${encodeURIComponent($('member-user').value)}`,{method:role==='revoke'?'DELETE':'PUT',body:role==='revoke'?undefined:{role}});notice('成员权限已更新。');}));
$('acl').addEventListener('submit',safe(async event=>{event.preventDefault();const users=$('acl-users').value.split(',').map(s=>s.trim()).filter(Boolean);await api('import',`/documents/${docId()}/acl`,{method:'PATCH',body:{reader_ids:users.length?users:null}});notice('文档权限已更新。');}));
if(csrf)enter().catch(()=>{sessionStorage.removeItem('kb_csrf');csrf='';});
