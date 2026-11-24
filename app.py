"""本地优先的网页接口：HTTP 和 WebSocket 共用问答流程，并限制上传文件大小。"""
from contextlib import asynccontextmanager
import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from concurrent.futures import CancelledError
from typing import Optional
import uuid
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from base.logger import logger, filter_known_library_notices

filter_known_library_notices()

from rag_qa.core.document_importer import ImportValidationError
from rag_qa.core.rag_pipeline import RetrievalError

ROOT = Path(__file__).resolve().parent
LOCAL_HOSTS = {'127.0.0.1', '::1', 'localhost'}
GREETINGS = [
    (r'(你好|您好|hi|hello)[!！。,.，？?\s]*', '你好！我是仓配 SaaS 实施与售后知识库助手。你可以按产品、平台和版本筛选后提问。'),
    (r'(你是谁|您是谁|你叫什么|你的名字|who are you)[!！。,.，？?\s]*',
     '我是仓配 SaaS 实施与售后知识库助手，可以结合公开帮助文档、API 文档和版本说明回答问题。'),
    (r'(在吗|在不在|有人吗)[!！。,.，？?\s]*', '我在，请告诉我你想了解哪个产品、平台和版本下的问题。'),
]


def check_greeting(query):
    for pattern, answer in GREETINGS:
        if re.fullmatch(pattern, query.strip(), re.IGNORECASE): return answer
    return None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=6000)
    source_filter: Optional[str] = Field(default=None, max_length=50)
    product: Optional[str] = Field(default=None, max_length=80)
    platform: Optional[str] = Field(default=None, max_length=80)
    version: Optional[str] = Field(default=None, max_length=160)
    session_id: Optional[uuid.UUID] = None

    @field_validator('query')
    @classmethod
    def strip_query(cls, value):
        value = value.strip()
        if not value: raise ValueError('问题不能为空')
        return value


class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool = False
    session_id: str
    processing_time: float
    sources: list = Field(default_factory=list)


def create_app(qa_factory=None, importer_factory=None):
    private_mode = os.getenv('QA_PRIVATE_MODE', '').strip() == '1'
    access_token = os.getenv('QA_ACCESS_TOKEN', '')
    if private_mode and not access_token.strip():
        raise RuntimeError('启动私有知识库前请设置 QA_ACCESS_TOKEN')
    cookie_name = 'qa_private_auth' if private_mode else 'qa_auth'

    def browser_cookie():
        expiry = str(int(time.time()) + 86400)
        return expiry + ':' + hmac.new(access_token.encode(), expiry.encode(), hashlib.sha256).hexdigest()

    def valid_cookie(value):
        try:
            expiry, digest = value.split(':', 1)
            if int(expiry) <= time.time(): return False
            expected = hmac.new(access_token.encode(), expiry.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(digest, expected)
        except (ValueError, TypeError):
            return False

    @asynccontextmanager
    async def lifespan(application):
        application.state.qa = None
        application.state.importer = None
        try:
            factory = qa_factory
            if factory is None:
                from cli import IntegratedQASystem
                factory = IntegratedQASystem
            qa = await run_in_threadpool(factory)
            application.state.qa = qa
            if getattr(qa, 'vector_store', None) is not None:
                if importer_factory is not None:
                    application.state.importer = importer_factory(qa)
                else:
                    from rag_qa.core.document_importer import DocumentImporter
                    from rag_qa.core.document_processor import process_document_file
                    application.state.importer = DocumentImporter(qa.config.DATA_DIR, qa.config.VALID_SOURCES,
                                                                  qa.vector_store, process_document_file,
                                                                  batch_size=qa.config.BATCH_SIZE)
        except Exception:
            logger.exception('问答服务初始化失败')
        try:
            yield
        finally:
            qa = application.state.qa
            if qa is not None and hasattr(qa, 'close'):
                await run_in_threadpool(qa.close)

    title = '企业私有知识库智能问答系统' if private_mode else '仓配 SaaS 实施与售后知识问答系统'
    app = FastAPI(title=title, lifespan=lifespan)

    def allowed(connection):
        if access_token:
            bearer = connection.headers.get('authorization', '')
            return (valid_cookie(connection.cookies.get(cookie_name, ''))
                    or hmac.compare_digest(bearer.encode(), ('Bearer ' + access_token).encode()))
        return (connection.client is not None and connection.client.host in LOCAL_HOSTS
                and connection.url.hostname in LOCAL_HOSTS)

    def same_origin(connection):
        origin = connection.headers.get('origin')
        if not origin:
            return True

        def normalized(value):
            parsed = urlsplit(value)
            scheme = {'ws': 'http', 'wss': 'https'}.get(parsed.scheme, parsed.scheme)
            if scheme not in ('http', 'https') or not parsed.hostname:
                raise ValueError('Invalid origin')
            if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
                raise ValueError('Invalid origin')
            return scheme, parsed.hostname.lower(), parsed.port or (443 if scheme == 'https' else 80)

        try:
            return normalized(origin) == normalized(str(connection.url.replace(path='', query='', fragment='')))
        except ValueError:
            return False

    @app.middleware('http')
    async def access_control(request, call_next):
        if request.url.path.startswith('/api/'):
            if request.method not in ('GET', 'HEAD', 'OPTIONS') and not same_origin(request):
                return JSONResponse({'detail': '请从本服务页面进行操作'}, status_code=403)
            public = request.url.path in ('/api/auth/status', '/api/auth/login')
            if not public and not allowed(request):
                return JSONResponse(
                    {'detail': '请先验证访问口令' if access_token else '远程访问需设置 QA_ACCESS_TOKEN'},
                    status_code=401 if access_token else 403)
        response = await call_next(request)
        if private_mode and request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    def service():
        qa = getattr(app.state, 'qa', None)
        if qa is None: raise HTTPException(503, '服务尚未就绪，请检查配置和启动日志')
        return qa

    def importer():
        value = getattr(app.state, 'importer', None)
        if value is None: raise HTTPException(503, '资料导入尚未就绪，请确认本地模型和向量库可用')
        return value

    def validate_subject(qa, source):
        if source and source not in qa.config.VALID_SOURCES: raise HTTPException(422, '请选择有效的资料分类')

    def validate_filters(qa, product, platform, version):
        try:
            return qa._build_filters(product, platform, version)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get('/')
    async def index():
        return FileResponse(ROOT / 'static' / 'index.html', headers={'Cache-Control': 'no-cache'})

    @app.get('/api/auth/status')
    async def auth_status(request: Request):
        return {'required': bool(access_token), 'authenticated': allowed(request),
                'remote_disabled': not access_token and not allowed(request), 'private_mode': private_mode}

    @app.post('/api/auth/login')
    async def login(request: Request):
        try:
            body = await request.json()
        except (ValueError, TypeError):
            raise HTTPException(400, '请输入有效口令')
        supplied = body.get('token') if isinstance(body, dict) else None
        if not access_token or not isinstance(supplied, str) or not hmac.compare_digest(supplied.encode(),
                                                                                        access_token.encode()):
            raise HTTPException(401, '访问口令不正确')
        response = JSONResponse({'authenticated': True})
        response.set_cookie(cookie_name, browser_cookie(), httponly=True, secure=request.url.scheme == 'https',
                            samesite='strict', max_age=86400)
        return response

    @app.get('/health')
    async def health():
        qa = getattr(app.state, 'qa', None)
        ready = qa is not None
        return JSONResponse({'status': 'ready' if ready else 'unavailable', 'ready': ready,
                             'retrieval_ready': ready and getattr(qa, 'vector_store', None) is not None},
                            status_code=200 if ready else 503)

    @app.post('/api/create_session')
    async def create_session():
        return {'session_id': str(uuid.uuid4())}

    @app.get('/api/sessions')
    async def sessions(limit: int = 100):
        return {'sessions': await run_in_threadpool(service().list_sessions, max(1, min(limit, 200)))}

    @app.get('/api/history/{session_id}')
    async def history(session_id: uuid.UUID, limit: int = 500):
        return {'session_id': str(session_id),
                'history': await run_in_threadpool(service().get_session_history, str(session_id),
                                                   max(1, min(limit, 500)))}

    @app.delete('/api/history/{session_id}')
    async def clear_history(session_id: uuid.UUID):
        await run_in_threadpool(service().clear_session_history, str(session_id))
        return {'status': 'success'}

    @app.get('/api/sources')
    async def sources():
        return {'sources': service().config.VALID_SOURCES}

    @app.get('/api/filters')
    async def filters():
        config = service().config
        labels = getattr(config, 'SOURCE_LABELS', {})
        return {'products': getattr(config, 'FILTER_PRODUCTS', []),
                'platforms': getattr(config, 'FILTER_PLATFORMS', []),
                'versions': getattr(config, 'FILTER_VERSIONS', []),
                'sources': [{'key': key, 'label': labels.get(key, key)} for key in config.VALID_SOURCES]}

    @app.get('/api/documents')
    async def documents(source: Optional[str] = None):
        validate_subject(service(), source)
        return {'documents': await run_in_threadpool(importer().list_documents, source)}

    @app.delete('/api/documents')
    async def delete_document(source: str, document_id: str):
        target = importer()
        try:
            return await run_in_threadpool(target.delete_file, source, document_id)
        except ImportValidationError as exc:
            raise HTTPException(400, str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except Exception:
            logger.exception('资料删除失败')
            raise HTTPException(500, '资料删除失败，请检查向量库服务后重试')

    @app.post('/api/documents')
    async def import_document(request: Request, source: str, filename: str,
                              doc_title: Optional[str] = None, section: Optional[str] = None,
                              product: Optional[str] = None, platform: Optional[str] = None,
                              version: Optional[str] = None, source_url: Optional[str] = None,
                              doc_type: Optional[str] = None):
        target = importer()
        metadata = {key: value for key, value in (
            ('doc_title', doc_title), ('section', section), ('product', product),
            ('platform', platform), ('version', version), ('source_url', source_url),
            ('doc_type', doc_type)) if value}
        try:
            extension = target.validate(source, filename)
        except ImportValidationError as exc:
            raise HTTPException(400, str(exc))
        with tempfile.TemporaryDirectory(prefix='qa-document-') as temporary:
            path = Path(temporary) / ('upload' + extension)
            size = 0
            with path.open('wb') as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > target.MAX_BYTES: raise HTTPException(413, '文件不能超过 20 MB')
                    await run_in_threadpool(stream.write, chunk)
            if not size: raise HTTPException(400, '不能上传空文件')
            try:
                return await run_in_threadpool(target.import_file, path, source, filename, metadata)
            except ImportValidationError as exc:
                raise HTTPException(422, str(exc))
            except Exception:
                logger.exception('资料导入失败')
                raise HTTPException(500, '资料导入失败，请检查文件和向量库服务后重试')

    def fixed_answer(qa, question, sid, subject=None):
        answer = check_greeting(question)
        if answer and private_mode:
            answer = '你好，我是企业私有知识库助手。你可以查询已导入的内部手册、制度或案例。'
        if answer: qa.save_fixed_answer(sid, question, answer, subject=subject)
        return answer

    @app.post('/api/query', response_model=QueryResponse)
    async def query(request: QueryRequest):
        qa = service();
        validate_subject(qa, request.source_filter)
        validate_filters(qa, request.product, request.platform, request.version)
        sid = str(request.session_id or uuid.uuid4());
        start = time.monotonic();
        source_items = []

        def execute():
            answer = fixed_answer(qa, request.query, sid, request.source_filter)
            if answer: return answer
            return ''.join(token for token, complete in qa.query(
                request.query, source_filter=request.source_filter, session_id=sid,
                source_callback=lambda values: source_items.extend(values),
                product=request.product, platform=request.platform, version=request.version))

        try:
            answer = await run_in_threadpool(execute)
        except RetrievalError as exc:
            logger.warning('知识库检索失败', exc_info=True)
            raise HTTPException(503, str(exc))
        except Exception:
            logger.exception('问答失败')
            raise HTTPException(502, '回答生成失败，请检查模型配置或稍后重试')
        return QueryResponse(answer=answer, session_id=sid, processing_time=time.monotonic() - start,
                             sources=source_items)

    @app.websocket('/api/stream')
    async def stream(websocket: WebSocket):
        if not allowed(websocket) or not same_origin(websocket):
            await websocket.close(code=1008);
            return
        await websocket.accept()
        generator = None
        cancel_event = threading.Event()
        disconnect_task = None
        next_task = None
        try:
            # 仅把解析与校验阶段的 ValueError 归类为请求错误，运行期错误单独上报。
            try:
                raw = await websocket.receive_text()
                if len(raw) > 16000: raise ValueError('问题过长')
                request = QueryRequest.model_validate(json.loads(raw))
            except (ValidationError, ValueError):
                await websocket.send_json({'type': 'error', 'error': '请求无效，请检查问题内容和资料分类'})
                return
            try:
                qa = service()
                validate_subject(qa, request.source_filter)
                validate_filters(qa, request.product, request.platform, request.version)
            except HTTPException as exc:
                await websocket.send_json({'type': 'error', 'error': str(exc.detail)})
                return
            sid = str(request.session_id or uuid.uuid4());
            start = time.monotonic();
            source_items = []
            await websocket.send_json({'type': 'start', 'session_id': sid})
            greeting = await run_in_threadpool(fixed_answer, qa, request.query, sid, request.source_filter)
            if greeting:
                await websocket.send_json({'type': 'token', 'token': greeting})
            else:
                generator = qa.query(request.query, source_filter=request.source_filter, session_id=sid,
                                     source_callback=lambda values: source_items.extend(values),
                                     cancel_event=cancel_event, product=request.product,
                                     platform=request.platform, version=request.version)
                iterator = iter(generator)
                def next_item():
                    try: return next(iterator)
                    except StopIteration: return None
                async def watch_disconnect():
                    try:
                        while True:
                            message = await websocket.receive()
                            if message.get('type') == 'websocket.disconnect': return
                    except (WebSocketDisconnect, RuntimeError):
                        return
                disconnect_task = asyncio.create_task(watch_disconnect())
                while True:
                    next_task = asyncio.create_task(run_in_threadpool(next_item))
                    done, _ = await asyncio.wait({next_task, disconnect_task},
                                                 return_when=asyncio.FIRST_COMPLETED)
                    if disconnect_task in done:
                        raise CancelledError('生成已取消')
                    item = await next_task
                    next_task = None
                    if item is None: break
                    token, complete = item
                    if token: await websocket.send_json({'type': 'token', 'token': token})
                    if complete: break
            await websocket.send_json({'type': 'end', 'is_complete': True, 'processing_time': time.monotonic() - start,
                                       'sources': source_items})
        except (WebSocketDisconnect, CancelledError):
            pass
        except RetrievalError as exc:
            if not cancel_event.is_set():
                logger.warning('流式知识库检索失败', exc_info=True)
                try:
                    await websocket.send_json({'type': 'error', 'error': str(exc)})
                except (RuntimeError, WebSocketDisconnect):
                    pass
        except Exception:
            if cancel_event.is_set():
                logger.info('流式问答已取消')
            else:
                logger.exception('流式问答失败')
                try:
                    await websocket.send_json({'type': 'error',
                                               'error': '回答生成失败，请检查模型配置或稍后重试。已生成的部分不会作为完整回答保存。'})
                except (RuntimeError, WebSocketDisconnect):
                    pass
        finally:
            cancel_event.set()
            if disconnect_task is not None and not disconnect_task.done():
                disconnect_task.cancel()
                try:
                    await disconnect_task
                except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    pass
            if generator is not None:
                # 取消 asyncio 任务不会停止已启动的工作线程。先等 next() 退出，
                # 再关闭生成器，避免并发调用 close() 导致会话锁无法及时释放。
                if next_task is not None:
                    await asyncio.gather(next_task, return_exceptions=True)
                try:
                    await run_in_threadpool(generator.close)
                except Exception:
                    logger.debug('关闭流式生成器时资源已释放', exc_info=True)
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    app.mount('/static', StaticFiles(directory=str(ROOT / 'static')), name='static')
    return app


app = create_app()
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', 8081)),
                reload=False, access_log=False)
