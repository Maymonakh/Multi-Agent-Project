import json
import socket
import threading
import time

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ui_client import ApiClient, ApiClientError

DOCS = {"a.txt": b"x"}
SEEN = {}


async def guard(request):
    if request.headers.get("x-api-key") != "secret":
        return JSONResponse({"detail": "Invalid or missing API key."}, status_code=401)


def route(fn):
    async def wrapper(request):
        return await guard(request) or await fn(request)
    return wrapper


@route
async def health(request):
    return JSONResponse({"status": "ok", "documents": len(DOCS), "ocr_engine": "tesseract"})


@route
async def documents(request):
    return JSONResponse({"documents": sorted(DOCS)})


@route
async def upload_document(request):
    form = await request.form()
    data = await form["file"].read()
    DOCS[form["file"].filename] = data
    return JSONResponse({"document": form["file"].filename, "message": "ok"}, status_code=201)


@route
async def upload_image(request):
    form = await request.form()
    return JSONResponse({"document": "i_ocr.txt", "message": "ok", "ocr_text": (await form["image"].read()).decode()})


@route
async def delete_document(request):
    name = request.path_params["name"]
    if name not in DOCS:
        return JSONResponse({"detail": f"Document not found: {name}"}, status_code=404)
    del DOCS[name]
    return Response(status_code=204)


@route
async def ask(request):
    body = await request.json()
    SEEN["ask"] = body
    if len(body["question"]) > 20:
        return JSONResponse({"detail": [{"loc": ["body", "question"], "msg": "too long"}]}, status_code=422)
    return JSONResponse({"answer": "hi", "cached": False, "analysis": {}, "elapsed_ms": 5, "ocr_text": None})


@route
async def ask_image(request):
    form = await request.form()
    SEEN["ask_image"] = {"q": form["question"], "history": form["history"], "img": (await form["image"].read())}
    return JSONResponse({"answer": "img", "cached": False, "analysis": {}, "elapsed_ms": 5, "ocr_text": "t"})


@route
async def clear_cache(request):
    return Response(status_code=204)


app = Starlette(routes=[
    Route("/health", health), Route("/documents", documents), Route("/documents", upload_document, methods=["POST"]),
    Route("/documents/image", upload_image, methods=["POST"]),
    Route("/documents/{name}", delete_document, methods=["DELETE"]),
    Route("/ask", ask, methods=["POST"]), Route("/ask/image", ask_image, methods=["POST"]),
    Route("/cache", clear_cache, methods=["DELETE"]),
])


def start_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}"


def expect_error(status, fn, *args):
    try:
        fn(*args)
    except ApiClientError as error:
        assert error.status_code == status, (error.status_code, error.detail)
        return error.detail
    raise AssertionError(f"expected ApiClientError {status}")


if __name__ == "__main__":
    server, url = start_server()
    c = ApiClient(url, api_key="secret")

    assert c.health()["ocr_engine"] == "tesseract"
    assert c.documents() == ["a.txt"]
    assert c.upload_document("b.txt", b"hello")["document"] == "b.txt" and c.documents() == ["a.txt", "b.txt"]
    assert c.upload_image("s.png", "نص عربي".encode())["ocr_text"] == "نص عربي"
    c.delete_document("b.txt")
    assert "not found" in expect_error(404, c.delete_document, "b.txt")
    c.delete_document("a.txt")                                   # 204 -> None
    assert c.documents() == []
    c.clear_cache()

    hist = [{"role": "user", "content": str(i)} for i in range(30)]
    assert c.ask("What is RAG?", hist)["answer"] == "hi" and len(SEEN["ask"]["history"]) == 20
    assert "question: too long" in expect_error(422, c.ask, "x" * 30)           # validation list -> readable text
    r = c.ask_image("q.png", b"\x89PNG", "Explain", hist[:2])
    assert r["ocr_text"] == "t" and SEEN["ask_image"]["q"] == "Explain" and SEEN["ask_image"]["img"] == b"\x89PNG"
    assert json.loads(SEEN["ask_image"]["history"])[0]["content"] == "0"

    assert "Invalid" in expect_error(401, ApiClient(url, api_key="wrong").health)
    assert "Cannot reach" in expect_error(0, ApiClient("http://127.0.0.1:1", api_key="x", timeout=2).health)
    server.should_exit = True
    print("ui_client OK")
