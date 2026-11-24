const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8').replace(/\r\n/g, '\n');
const definitions = ['async function switchSession', 'function send'].map(name =>
    html.match(new RegExp('        ' + name + '\\([\\s\\S]*?\\n        \\}\\n'))[0]
).join('\n');

test('slow history and stale responses cannot erase or replace the active session', async () => {
    const state = {historyRequest: 0, switching: false, sid: null, deletingSessions: new Set()};
    const pending = new Map(), notices = [], messages = [];
    const build = new Function('state', 'api', 'stop', 'saveSid', 'closeSidebar', 'resetMessages',
        'renderSessions', 'addMessage', 'scrollToBottom', 'toast', 'syncSendButton',
        definitions + '\nreturn {switchSession, send};');
    const {switchSession, send} = build(state, url => new Promise(resolve => pending.set(url, resolve)),
        () => {}, sid => {state.sid = sid;}, () => {}, () => {messages.length = 0;}, () => {},
        (role, text) => messages.push(text), () => {}, message => notices.push(message), () => {});
    const first = switchSession('first');
    const latest = switchSession('latest');
    send();
    assert.equal(notices.length, 1);
    pending.get('/api/history/first?limit=500')({history: [{question: '过期的问题'}]});
    await first;
    assert.equal(state.switching, true);
    assert.deepEqual(messages, []);
    pending.get('/api/history/latest?limit=500')({history: [{question: '旧问题', answer: '旧回答'}]});
    await latest;
    assert.equal(state.switching, false);
    assert.deepEqual(messages, ['旧问题', '旧回答']);
});
