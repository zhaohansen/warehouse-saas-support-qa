const fs = require('fs');
const path = require('path');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8').replace(/\r\n/g, '\n');

// 历史回答用 instant 创建，如果初始化 done=true，finalizeAssistant 会直接返回，
// 导致问题保留而回答区域空白。
assert(html.includes('done: false,'), '消息初始状态不应预设为已完成，否则历史回答不会渲染');

// 提取 finalizeAssistant，用最小替身验证渲染与异常回退行为。
const finalizeMatch = html.match(/        function finalizeAssistant\(msg\) \{[\s\S]*?\n        \}\n/);
assert(finalizeMatch, '未找到 finalizeAssistant 函数');
const finalizeSource = finalizeMatch[0].trim();

const stripMatch = html.match(/        function isSourceEntryLine\(line\) \{[\s\S]*?\n        \}\n\n        function stripAnswerSources[\s\S]*?\n        \}/);
assert(stripMatch, '未找到来源清理函数');
const strip = new Function(stripMatch[0] + '\nreturn stripAnswerSources;')();

function buildFinalize(renderer) {
  const calls = {render: [], extras: 0, scroll: 0};
  const finalize = new Function(
    'ensureMessage', 'stripAnswerSources', 'rebuildSourceIndexes', 'renderMarkdown',
    'node', 'renderExtras', 'scheduleScroll',
    finalizeSource + '\nreturn finalizeAssistant;'
  )(
    msg => { msg.prose = msg.prose || fakeProse(); return msg; },
    strip,
    msg => { msg.sourceIndexes = new Set((msg.sources || []).map(item => String(item.index))); },
    (container, text) => { calls.render.push(text); renderer(container, text); },
    (tag, cls, text) => ({tag, cls, text, children: []}),
    () => { calls.extras++; },
    () => { calls.scroll++; }
  );
  return {finalize, calls};
}

function fakeProse() {
  return {
    textContent: '',
    children: [],
    classList: {remove() {}, add() {}},
    replaceChildren(...nodes) { this.children = nodes; this.textContent = nodes.map(n => n.text || '').join(''); },
    append(...nodes) { this.children.push(...nodes); }
  };
}

// 1) 历史回答（instant）能正常渲染，来源索引按 sources 建立。
{
  const {finalize, calls} = buildFinalize((container, text) => {
    container.textContent = text;
  });
  const msg = {role: 'assistant', text: '历史回答正文 [1]。', sources: [{index: 1}], done: false,
    error: '', notice: '', status: '历史回答', prose: fakeProse(), extra: {replaceChildren() {}}};
  finalize(msg);
  assert.equal(msg.done, true, '渲染后应标记为已完成');
  assert.equal(calls.render.length, 1, '历史回答应渲染一次');
  assert.equal(msg.prose.textContent, '历史回答正文 [1]。', '历史回答正文必须显示');
  assert(msg.sourceIndexes.has('1'), '来源索引应建立');
}

// 2) 重复调用不会再清空或重复渲染。
{
  const {finalize, calls} = buildFinalize((container, text) => { container.textContent = text; });
  const msg = {role: 'assistant', text: '正文 [1]。', sources: [], done: false, error: '',
    notice: '', status: '', prose: fakeProse(), extra: {replaceChildren() {}}};
  finalize(msg);
  finalize(msg);
  assert.equal(calls.render.length, 1, '已完成的消息不应重复渲染');
  assert.equal(msg.prose.textContent, '正文 [1]。');
}

// 3) Markdown 渲染异常时回退为纯文本，回答不能变成空白。
{
  const {finalize} = buildFinalize(() => { throw new Error('render failed'); });
  const msg = {role: 'assistant', text: '异常回退正文 [1]。', sources: [], done: false, error: '',
    notice: '', status: '', prose: fakeProse(), extra: {replaceChildren() {}}};
  finalize(msg);
  assert.equal(msg.prose.textContent, '异常回退正文 [1]。', '渲染异常时应回退为纯文本，不能留下空白');
  assert(msg.notice.includes('纯文本'), '回退时应给出提示');
}

// 4) 流式中断且没有生成任何内容时，显示提示而不是空白。
{
  const {finalize} = buildFinalize((container, text) => { container.textContent = text; });
  const msg = {role: 'assistant', text: '', sources: [], done: false, error: '回答生成失败',
    notice: '', status: '生成失败', prose: fakeProse(), extra: {replaceChildren() {}}};
  finalize(msg);
  assert(msg.prose.textContent.includes('回答生成中断'), '没有内容时应显示中断提示，而不是空白');
}

console.log('回答可见性测试通过：历史回答可渲染、重复渲染受保护、渲染异常回退纯文本、空内容不空白。');
