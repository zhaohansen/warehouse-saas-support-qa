const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8').replace(/\r\n/g, '\n');
const source = html.match(/        function selectFile\(file\) \{[\s\S]*?\n        \}\n\n        async function upload/)[0].replace(/\n\n        async function upload$/, '');
const elements = {};
for (const id of ['document-file', 'selected-file', 'btn-upload', 'import-status']) {
  elements[id] = { value: '', textContent: '', disabled: false, hidden: false,
    classList: { remove() {} } };
}
const state = { uploading: false, file: null };
const select = new Function('state', '$', source + '\nreturn selectFile;')(state, id => elements[id]);
const assert = require('assert');
select({name:'ok.pdf', size: 20});
assert.equal(state.file.name, 'ok.pdf');
assert.equal(elements['btn-upload'].disabled, false);
select(null);
assert.equal(state.file, null);
assert.equal(elements['btn-upload'].disabled, true);
assert.equal(elements['selected-file'].textContent, '点击选择，或将文件拖到这里');
select({name:'bad.exe', size: 20});
assert.equal(state.file, null);
assert.equal(elements['btn-upload'].disabled, true);
assert.equal(elements['import-status'].hidden, false);
select({name:'ok.pdf', size: 20});
assert.equal(elements['btn-upload'].disabled, false);
select({name:'手册.DOCX', size: 20});
assert.equal(state.file.name, '手册.DOCX');
assert.equal(elements['btn-upload'].disabled, false);
console.log('文件选择、取消、无效文件及重新选择测试通过。');
