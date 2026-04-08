from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

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
from .web_service import DeepFindWebService


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.service = service or DeepFindWebService()

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(ok=True, service="deepfind-web", local_model=app.state.service.local_model_info())

    @app.get("/api/chats", response_model=ChatListResponse)
    def list_chats() -> ChatListResponse:
        return ChatListResponse(
            chats=app.state.service.list_chats(),
            local_model=app.state.service.local_model_info(),
        )

    @app.post("/api/chats", response_model=CreateChatResponse)
    def create_chat(payload: CreateChatRequest | None = None) -> CreateChatResponse:
        chat = app.state.service.create_chat(title=payload.title if payload else None)
        return CreateChatResponse(chat=chat)

    @app.get("/api/chats/{chat_id}", response_model=ChatDetailResponse)
    def get_chat(chat_id: str) -> ChatDetailResponse:
        try:
            chat = app.state.service.get_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}") from exc
        return ChatDetailResponse(chat=chat)

    @app.delete("/api/chats/{chat_id}", status_code=204)
    def delete_chat(chat_id: str) -> Response:
        try:
            app.state.service.delete_chat(chat_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}") from exc
        return Response(status_code=204)

    @app.post("/api/chats/{chat_id}/messages/stream")
    def stream_message(chat_id: str, payload: SendMessageRequest) -> StreamingResponse:
        try:
            stream = app.state.service.stream_message(
                chat_id,
                payload.content,
                payload.mode,
                payload.model_target,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"chat not found: {chat_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        resolved = app.state.service.resolve_file_path(path)
        root = repo_root().resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="path is outside the repository") from exc
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(resolved)

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
