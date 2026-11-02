"""FastAPI 服务：提供 HTTP 与 WebSocket 接口，并挂载前端静态页。"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

import os
import json
import uuid
import time
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from cli import IntegratedQASystem

# 所有静态资源路径以 app.py 所在目录为基准，避免从其他工作目录启动时找不到文件。
PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_ROOT / "static"
app = FastAPI(title="学科知识问答系统 API", description="基于 FAQ 与 RAG 双通道的学科知识问答系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态目录不存在就先建出来, 否则下面 mount 会直接抛异常
os.makedirs(STATIC_DIR, exist_ok=True)

# 注意: 这一行在 import 时就会执行, 会加载 VectorStore(可能加载 embedding 模型), 启动会慢几秒
qa_system = IntegratedQASystem()
GREETING_PATTERNS = [
    {"pattern": r"^(你好|您好|hi|hello)", "response": "你好！我是学科知识问答助手，很高兴为你服务！"},
    {"pattern": r"^(你是谁|您是谁|你叫什么|你的名字|who are you)", "response": "我是学科知识问答助手，覆盖 Python、Java、AI、MySQL、Redis 五个学科。"},
    {"pattern": r"^(在吗|在不在|有人吗)", "response": "我在！随时为你解答问题。"},
    {"pattern": r"^(干嘛呢|你在干嘛|做什么)", "response": "我正在待命，有什么可以帮你的？"},
]


def check_greeting(query: str) -> Optional[str]:
    """命中问候语就返回固定回复, 否则返回 None."""
    text = (query or "").strip()
    for item in GREETING_PATTERNS:
        if re.match(item["pattern"], text, re.IGNORECASE):
            return item["response"]
    return None


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str
    processing_time: float

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def read_root():
    """
    根路径: 有 static/index.html 就返回它, 没有就给一个内置的极简聊天页.
    直接 FileResponse('static/index.html'), 文件不存在会 500 报错,
             这里做了兜底, 克隆下来就能直接打开用.
    """
    index_path = STATIC_DIR / "index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(FALLBACK_PAGE)


@app.post("/api/create_session")
async def create_session():
    """新建会话, 返回一个 session_id. 前端把它存起来, 之后每轮都带上, 才能连续对话."""
    return {"session_id": str(uuid.uuid4())}


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    try:
        history = qa_system.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败: {e}")


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    if qa_system.clear_session_history(session_id):
        return {"status": "success", "message": "历史记录已清除"}
    raise HTTPException(status_code=500, detail="清除历史记录失败")


@app.get("/api/sources")
async def get_sources():
    """返回前端下拉框要用的学科列表."""
    return {"sources": qa_system.config.VALID_SOURCES}


@app.get("/health")
async def health_check():
    """健康检查: 只要返回 200, 就说明服务活着."""
    return {"status": "healthy"}


@app.post("/api/query")
async def query(request: QueryRequest):
    """
    一次性返回完整答案(不流式).
    这个接口只在"MySQL 能直接命中"时好用; 需要 RAG 生成时, 它会告诉前端去走 WebSocket.
    """
    start = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    greeting = check_greeting(request.query)
    if greeting:
        return QueryResponse(
            answer=greeting,
            is_streaming=False,
            session_id=session_id,
            processing_time=time.time() - start,
        )

    answer = qa_system.bm25_search.search(request.query, threshold=0.85)
    need_rag = answer is None

    if need_rag:
        return QueryResponse(
            answer="请使用 WebSocket 接口(/api/stream)获取流式响应",
            is_streaming=True,
            session_id=session_id,
            processing_time=time.time() - start,
        )

    return QueryResponse(
        answer=answer,
        is_streaming=False,
        session_id=session_id,
        processing_time=time.time() - start,
    )


@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    前端连上来之后, 发一条 {query, source_filter, session_id} 的 JSON,
    服务端按 type=start / token / end 三种消息往回推.
    消息协议:
        {"type": "start", "session_id": "..."}
        {"type": "token", "token": "一小段文字", "session_id": "..."}
        {"type": "end",   "is_complete": true, "processing_time": 1.23}
        {"type": "error", "error": "错误信息"}
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            query_text = data.get("query", "")
            source_filter = data.get("source_filter")
            session_id = data.get("session_id") or str(uuid.uuid4())
            start = time.time()
            await websocket.send_json({"type": "start", "session_id": session_id})
            greeting = check_greeting(query_text)
            if greeting:
                await websocket.send_json(
                    {"type": "token", "token": greeting, "session_id": session_id}
                )
                await websocket.send_json(
                    {
                        "type": "end",
                        "is_complete": True,
                        "processing_time": time.time() - start,
                    }
                )
                continue

            collected = ""
            for token, is_complete in qa_system.query(
                query_text, source_filter=source_filter, session_id=session_id
            ):
                collected += token
                if token and websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(
                        {"type": "token", "token": token, "session_id": session_id}
                    )
                if is_complete:
                    await websocket.send_json(
                        {
                            "type": "end",
                            "is_complete": True,
                            "processing_time": time.time() - start,
                        }
                    )
                    break

    except WebSocketDisconnect as e:
        print(f"WebSocket 断开: code={e.code}, reason={e.reason}")
    except Exception as e:
        print(f"WebSocket 异常: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

FALLBACK_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>学科知识问答系统</title>
<style>
 body{font-family:system-ui,"Microsoft YaHei";max-width:780px;margin:40px auto;padding:0 16px;color:#222}
 #log{border:1px solid #e3e3e3;border-radius:8px;padding:12px;height:420px;overflow:auto;background:#fafafa}
 .q{color:#0b6bcb;margin:10px 0 4px}
 .a{white-space:pre-wrap;margin:0 0 14px}
 .row{display:flex;gap:8px;margin-top:12px}
 input{flex:1;padding:9px 12px;border:1px solid #d5d5d5;border-radius:6px;font-size:14px}
 button{padding:9px 18px;border:0;border-radius:6px;background:#0b6bcb;color:#fff;cursor:pointer}
</style></head><body>
<h3>学科知识问答系统</h3>
<div id="log"></div>
<div class="row"><input id="q" placeholder="输入问题后回车..."><button onclick="send()">发送</button></div>
<script>
let sid = null;
const log = document.getElementById('log');
function add(cls, txt){const d=document.createElement('div');d.className=cls;d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
async function init(){const r=await fetch('/api/create_session',{method:'POST'});sid=(await r.json()).session_id;}
async function send(){
  const inp=document.getElementById('q'); const q=inp.value.trim(); if(!q) return; inp.value='';
  add('q','我：'+q); const node=add('a','AI：');
  const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/api/stream');
  ws.onopen=()=>ws.send(JSON.stringify({query:q,session_id:sid}));
  ws.onmessage=(e)=>{const m=JSON.parse(e.data);
    if(m.type==='token') node.textContent+=m.token;
    if(m.type==='end'){ws.close();}
    if(m.type==='error') node.textContent+='[错误] '+m.error;
  };
}
init();
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')send()});
</script></body></html>
"""

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8082))
    uvicorn.run("app:app", host=host, port=port, reload=False)
