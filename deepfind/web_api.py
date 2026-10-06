from __future__ import annotations

import argparse
import asyncio
import json
import logging
import mimetypes
import os
import secrets
from queue import Empty
from pathlib import Path
import torch

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .chat_store import repo_root
from .web_models import (
    ChatDetailResponse,
    ChatListResponse,
    CreateChatRequest,
    CreateChatResponse,
    HealthResponse,
    ProgressEvent,
    SendMessageRequest,
)
from .config import SettingsError, _load_dotenv
from .web_service import DeepFindWebService
from .workspace import MAX_PDF_BYTES, MAX_TEXT_BYTES, WorkspaceError

# Load .env file at module import time
_load_dotenv()

logger = logging.getLogger("uvicorn.error")


def _configured_token() -> str:
    return (os.environ.get("DEEPFIND_WEB_TOKEN") or "").strip()


def require_auth(authorization: str = Header(default="")) -> None:
    token = _configured_token()
    if not token:
        return
    provided = authorization.removeprefix("Bearer ").strip()
    if not provided or not secrets.compare_digest(provided, token):
        raise HTTPException(status_code=401, detail="unauthorized")


def _workspace_http_error(exc: WorkspaceError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": exc.message},
    )


def _require_websocket_auth(websocket: WebSocket) -> bool:
    token = _configured_token()
    if not token:
        return True
    provided = websocket.query_params.get("token", "")
    return bool(provided and secrets.compare_digest(provided, token))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        # SAMEORIGIN allows the app's own iframe artifacts to load
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=()"
        return response


def _allowed_origins() -> list[str]:
    raw = (os.environ.get("DEEPFIND_CORS_ORIGINS") or "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    if _configured_token():
        return []
    return ["*"]


def _encode_sse(event: ProgressEvent) -> str:
    payload = json.dumps(
        {
            "timestamp": event.timestamp,
            "data": event.data,
        },
        ensure_ascii=False,
    )
    return f"event: {event.type}\ndata: {payload}\n\n"


def build_app(service: DeepFindWebService | None = None) -> FastAPI:
    app = FastAPI(title="DeepFind Web")

    origins = _allowed_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "Authorization", "Accept"],
        )

    app.add_middleware(SecurityHeadersMiddleware)
    app.state.service = service or DeepFindWebService(enable_workspace=True)

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        requires_token = bool(_configured_token())
        return HealthResponse(
            ok=True,
            service="deepfind-web",
            local_model=app.state.service.local_model_info(),
            requires_token=requires_token,
        )

    @app.get("/api/chats", response_model=ChatListResponse)
    def list_chats(_auth: None = Depends(require_auth)) -> ChatListResponse:
        return ChatListResponse(
            chats=app.state.service.list_chats(),
            local_model=app.state.service.local_model_info(),
            tools=app.state.service.tool_options(),
        )

    @app.post("/api/chats", response_model=CreateChatResponse)
    def create_chat(payload: CreateChatRequest | None = None, _auth: None = Depends(require_auth)) -> CreateChatResponse:
        chat = app.state.service.create_chat(title=payload.title if payload else None)
        return CreateChatResponse(chat=chat)

    @app.get("/api/chats/{chat_id}", response_model=ChatDetailResponse)
    def get_chat(chat_id: str, _auth: None = Depends(require_auth)) -> ChatDetailResponse:
        try:
            chat = app.state.service.get_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}") from exc
        return ChatDetailResponse(chat=chat)

    @app.delete("/api/chats/{chat_id}", status_code=204)
    def delete_chat(chat_id: str, _auth: None = Depends(require_auth)) -> Response:
        try:
            app.state.service.delete_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}") from exc
        return Response(status_code=204)

    @app.get("/api/chats/{chat_id}/workspace")
    def workspace_status(chat_id: str, _auth: None = Depends(require_auth)) -> dict:
        try:
            app.state.service.get_chat(chat_id)
            return app.state.service.workspace_manager.status(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/api/chats/{chat_id}/workspace/files")
    def workspace_files(
        chat_id: str,
        path: str = Query("."),
        _auth: None = Depends(require_auth),
    ) -> dict:
        try:
            app.state.service.get_chat(chat_id)
            return app.state.service.workspace_manager.list_files(chat_id, path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/api/chats/{chat_id}/workspace/metadata")
    def workspace_metadata(
        chat_id: str,
        path: str = Query(...),
        _auth: None = Depends(require_auth),
    ) -> dict:
        try:
            app.state.service.get_chat(chat_id)
            return app.state.service.workspace_manager.metadata(chat_id, path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/api/chats/{chat_id}/workspace/content")
    def workspace_content(
        chat_id: str,
        request: Request,
        path: str = Query(...),
        _auth: None = Depends(require_auth),
    ) -> Response:
        try:
            app.state.service.get_chat(chat_id)
            metadata = app.state.service.workspace_manager.metadata(chat_id, path)
            media_type = metadata.get("mime_type") or mimetypes.guess_type(path)[0] or "application/octet-stream"
            limit = MAX_PDF_BYTES if path.lower().endswith(".pdf") else MAX_TEXT_BYTES
            range_header = request.headers.get("range", "")
            offset = 0
            length = None
            status_code = 200
            headers = {
                "Content-Disposition": "inline",
                "Cache-Control": "no-store",
                "Accept-Ranges": "bytes",
            }
            if range_header.startswith("bytes="):
                start_text, _, end_text = range_header.removeprefix("bytes=").partition("-")
                if not start_text.isdigit() or (end_text and not end_text.isdigit()):
                    raise HTTPException(status_code=416, detail="invalid byte range")
                offset = int(start_text)
                end = min(int(end_text) if end_text else metadata["size"] - 1, metadata["size"] - 1)
                if offset > end:
                    raise HTTPException(status_code=416, detail="invalid byte range")
                length = end - offset + 1
                status_code = 206
                headers["Content-Range"] = f"bytes {offset}-{end}/{metadata['size']}"
            content = app.state.service.workspace_manager.read_file(
                chat_id,
                path,
                limit=limit,
                offset=offset,
                length=length,
            )
            return Response(
                content=content,
                media_type=media_type,
                status_code=status_code,
                headers=headers,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/api/chats/{chat_id}/workspace/documents/pdf")
    def workspace_pdf(
        chat_id: str,
        request: Request,
        path: str = Query(...),
        _auth: None = Depends(require_auth),
    ) -> Response:
        if not path.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=415,
                detail={"code": "unsupported_format", "message": "The file is not a PDF"},
            )
        try:
            app.state.service.get_chat(chat_id)
            metadata = app.state.service.workspace_manager.metadata(chat_id, path)
            range_header = request.headers.get("range", "")
            offset = 0
            length = None
            status_code = 200
            headers = {"Cache-Control": "no-store", "Accept-Ranges": "bytes"}
            if range_header.startswith("bytes="):
                start_text, _, end_text = range_header.removeprefix("bytes=").partition("-")
                if not start_text.isdigit() or (end_text and not end_text.isdigit()):
                    raise HTTPException(status_code=416, detail="invalid byte range")
                offset = int(start_text)
                end = min(int(end_text) if end_text else metadata["size"] - 1, metadata["size"] - 1)
                if offset > end:
                    raise HTTPException(status_code=416, detail="invalid byte range")
                length = end - offset + 1
                status_code = 206
                headers["Content-Range"] = f"bytes {offset}-{end}/{metadata['size']}"
            content = app.state.service.workspace_manager.read_file(
                chat_id,
                path,
                limit=MAX_PDF_BYTES,
                offset=offset,
                length=length,
            )
            return Response(content=content, media_type="application/pdf", status_code=status_code, headers=headers)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/api/chats/{chat_id}/workspace/documents/workbook")
    def workspace_workbook(
        chat_id: str,
        path: str = Query(...),
        _auth: None = Depends(require_auth),
    ) -> dict:
        try:
            app.state.service.get_chat(chat_id)
            return app.state.service.workspace_manager.workbook(chat_id, path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/api/chats/{chat_id}/workspace/documents/sheet")
    def workspace_sheet(
        chat_id: str,
        path: str = Query(...),
        sheet_id: str = Query(...),
        cell_range: str = Query("A1:Z100"),
        _auth: None = Depends(require_auth),
    ) -> dict:
        try:
            app.state.service.get_chat(chat_id)
            return app.state.service.workspace_manager.sheet(chat_id, path, sheet_id, cell_range)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/api/chats/{chat_id}/workspace/documents/word")
    def workspace_word(
        chat_id: str,
        path: str = Query(...),
        _auth: None = Depends(require_auth),
    ) -> dict:
        try:
            app.state.service.get_chat(chat_id)
            return app.state.service.workspace_manager.word(chat_id, path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.post("/api/chats/{chat_id}/workspace/terminals")
    def create_workspace_terminal(chat_id: str, _auth: None = Depends(require_auth)) -> dict:
        try:
            app.state.service.get_chat(chat_id)
            return app.state.service.workspace_manager.create_terminal(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.get("/api/chats/{chat_id}/workspace/terminals")
    def list_workspace_terminals(chat_id: str, _auth: None = Depends(require_auth)) -> dict:
        try:
            app.state.service.get_chat(chat_id)
            return {"terminals": app.state.service.workspace_manager.list_terminals(chat_id)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.delete("/api/chats/{chat_id}/workspace/terminals/{terminal_id}", status_code=204)
    def close_workspace_terminal(
        chat_id: str,
        terminal_id: str,
        _auth: None = Depends(require_auth),
    ) -> Response:
        try:
            app.state.service.get_chat(chat_id)
            app.state.service.workspace_manager.close_terminal(chat_id, terminal_id)
            return Response(status_code=204)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="chat not found") from exc
        except WorkspaceError as exc:
            raise _workspace_http_error(exc) from exc

    @app.websocket("/api/chats/{chat_id}/workspace/terminals/{terminal_id}/stream")
    async def workspace_terminal_stream(
        websocket: WebSocket,
        chat_id: str,
        terminal_id: str,
    ) -> None:
        if not _require_websocket_auth(websocket):
            await websocket.close(code=4401)
            return
        try:
            app.state.service.get_chat(chat_id)
            terminal = app.state.service.workspace_manager.get_terminal(chat_id, terminal_id)
        except (FileNotFoundError, WorkspaceError):
            await websocket.close(code=4404)
            return
        await websocket.accept()
        subscriber, replay = terminal.subscribe(int(websocket.query_params.get("after", "0")))

        async def send_events() -> None:
            for event in replay:
                await websocket.send_json(event)
            while True:
                try:
                    event = await asyncio.to_thread(subscriber.get, True, 0.5)
                except Empty:
                    continue
                await websocket.send_json(event)
                if event.get("type") == "exit":
                    return
                if event.get("code") == "terminal_backpressure":
                    await websocket.close(code=4408)
                    return

        sender = asyncio.create_task(send_events())
        try:
            while True:
                terminal.send(await websocket.receive_json())
        except WebSocketDisconnect:
            pass
        except WorkspaceError as exc:
            await websocket.send_json({"type": "error", "code": exc.code, "message": exc.message})
        finally:
            sender.cancel()
            terminal.unsubscribe(subscriber)

    @app.post("/api/chats/{chat_id}/messages/stream")
    def stream_message(chat_id: str, payload: SendMessageRequest, _auth: None = Depends(require_auth)) -> StreamingResponse:
        try:
            stream = app.state.service.stream_message(
                chat_id,
                payload.content,
                payload.mode,
                payload.model_target,
                deep_mode=payload.deep_mode,
                research_mode=payload.research_mode,
                selected_tools=payload.selected_tools,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SettingsError as exc:
            return PlainTextResponse(str(exc), status_code=400)
        return StreamingResponse(
            (_encode_sse(event) for event in stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/files")
    def get_file(path: str = Query(...)) -> FileResponse:
        try:
            resolved = app.state.service.resolve_file_path(path)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        root = repo_root().resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="path is outside the repository") from exc
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(resolved)

    @app.get("/api/rag/files")
    def get_rag_file(citation: str = Query(...)) -> FileResponse:
        try:
            resolved = app.state.service.resolve_rag_document(citation)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="RAG document not found")
        return FileResponse(
            resolved,
            filename=resolved.name,
            content_disposition_type="attachment",
        )

    dist_dir = repo_root() / "web" / "dist"
    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="web")
    else:
        @app.get("/", include_in_schema=False)
        def landing() -> PlainTextResponse:
            return PlainTextResponse("DeepFind Web API is running. Build ./web to serve the UI here.")

    return app


def create_app() -> FastAPI:
    return build_app()


def main(argv: list[str] | None = None) -> int:
    logger.info("PyTorch CUDA available: %s", torch.cuda.is_available())

    parser = argparse.ArgumentParser(prog="deepfind-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "deepfind.web_api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
