const fs = require('fs');
const path = require('path');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8').replace(/\r\n/g, '\n');
assert(html.includes("node('div', 'sources-head', '参考来源')"), '来源区域标题应为“参考来源”');
assert(!html.includes('来源 · 知识库资料'), '不应再出现重复的“来源 · 知识库资料”标题');

const match = html.match(/        function isSourceEntryLine\(line\) \{[\s\S]*?\n        \}\n\n        function stripAnswerSources[\s\S]*?\n        \}\n\n        function renderSources/);
assert(match, '未找到 isSourceEntryLine / stripAnswerSources 函数');
const source = match[0].replace(/\n\n        function renderSources$/, '');
const strip = new Function(source + '\nreturn stripAnswerSources;')();

const answer = [
  '**适用条件**',
  '接口要求客户端 V2.4.5.0 及以上 [1]。',
  '',
  '**来源**',
  '- [1] trade_push.php（创建原始订单）；版本：客户端 V2.4.5.0 及以上；原文链接：https://open.wangdian.cn/open/apidoc/doc?path=trade_push.php'
].join('\n');
const stripped = strip(answer);
assert(!stripped.includes('**来源**'), '末尾来源清单应被移除');
assert(!stripped.includes('https://open.wangdian.cn'), '来源清单中的链接应随清单一并移除');
assert(stripped.includes('[1]'), '正文引用编号应保留');
assert(stripped.includes('适用条件'), '正文内容应保留');

const heading = '结论：不能直接使用 [2]。\n\n## 来源\n[1] 客户端版本限制\n原文链接：https://example.com/notes';
assert.equal(strip(heading), '结论：不能直接使用 [2]。');

const plain = '来源可以看接口文档，正文里的 [1] 是引用。';
assert.equal(strip(plain), plain, '正文提到“来源”时不应误删');

const noLink = '结论 [1]。\n\n**来源**\n[1] 某接口文档';
assert.equal(strip(noLink), noLink, '没有链接的来源清单不按重复区域处理');

// 回归：正文中的“来源单据/来源单号”是业务字段，不是来源清单，不能被截断。
const businessField = [
  '**适用条件**',
  '- 来源单据必须与订单一致 [1]。',
  '- 请求地址：https://api.wangdian.cn/openapi2/trade_push.php [1]。',
  '',
  '**排查顺序**',
  '1. 核对来源单据号是否与订单一致 [1]。',
  '2. 核对签名 [1]。'
].join('\n');
assert.equal(strip(businessField), businessField, '正文里的“来源单据”不能被误判为来源清单');

const midHeading = [
  '结论：可以先核对来源字段 [1]。',
  '',
  '来源：平台订单',
  '1. 核对订单号 [1]。',
  '2. 查看 https://example.com/doc 的说明 [1]。'
].join('\n');
assert.equal(strip(midHeading), midHeading, '“来源：”后是正文而不是来源条目时不能截断');

// 整段回答只有来源清单时，保留原文，避免回答区域变空白。
const onlySources = '来源：\n[1] trade_push.php\n原文链接：https://open.wangdian.cn/open/apidoc/doc?path=trade_push.php';
assert.equal(strip(onlySources), onlySources, '只有来源清单时不应把整段回答删空');

console.log('来源去重展示测试通过：旧来源清单被移除、正文引用与业务字段保留、正文“来源”不受影响。');
