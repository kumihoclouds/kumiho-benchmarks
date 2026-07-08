"""
Local OpenAI-compatible proxy backed by ChatGPT/Codex OAuth — NO API key.

Why this exists
---------------
The benchmark harness (``common.py``) reaches every LLM through the
``openai`` SDK: ``AsyncOpenAI().chat.completions.create(...)`` for answering
(``generate_answer``), judging (``llm_judge``), query reformulation, and the
kumiho-memory ``MemorySummarizer`` used during ingestion/consolidation. The
SDK honours the ``OPENAI_BASE_URL`` environment variable, so pointing that at
this proxy routes *all* of those calls through a user's ChatGPT subscription
(the OAuth token the ``codex`` CLI already stored in ``~/.codex/auth.json``)
instead of a paid API key — without touching a single line of ``common.py`` or
the kumiho-memory SDK.

    POST /v1/chat/completions  ->  https://chatgpt.com/backend-api/codex/responses

Usage
-----
    # 0. one-shot check the OAuth token works
    python -m kumiho_eval.codex_proxy --self-test

    # 1. start the proxy (foreground)
    python -m kumiho_eval.codex_proxy --port 8123

    # 2. in another shell, point the benchmark at it
    export OPENAI_BASE_URL=http://127.0.0.1:8123/v1
    export OPENAI_API_KEY=codex-oauth          # any non-empty string
    python -m kumiho_eval.run_benchmarks --benchmark locomo ...

Model mapping
-------------
The harness hard-codes ``gpt-4o`` / ``gpt-4o-mini`` in several places. Any
non-reasoning model name is transparently mapped to ``CODEX_MODEL`` (default
``gpt-5``); names that already look like reasoning models (gpt-5*, codex*,
o1/o3/o4*) pass through unchanged.

Limits
------
* ChatGPT-subscription Codex has plan rate limits (Plus < Pro). Long runs will
  hit 429s — the harness already wraps calls in exponential backoff.
* The Responses backend serves reasoning models only; ``temperature`` from the
  caller is ignored and ``reasoning.effort`` (default ``minimal``) is used.
* There is no embeddings endpoint — ``/v1/embeddings`` returns 501. Run with
  ``--sibling-similarity-threshold 0`` and no ``--two-pass-rerank`` so the
  harness never needs client-side embeddings (core recall embeds server-side).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger("kumiho_eval.codex_proxy")

# --- OAuth / endpoint constants (match the openai/codex CLI) ----------------
OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
DEFAULT_CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"

DEFAULT_CODEX_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.5")
DEFAULT_REASONING_EFFORT = os.environ.get("CODEX_REASONING_EFFORT", "low")
DEFAULT_ORIGINATOR = os.environ.get("CODEX_ORIGINATOR", "codex_cli_rs")
# Reasoning models spend part of the output budget on hidden reasoning tokens,
# so a tiny caller cap (e.g. judge max_tokens=10) would starve the visible
# answer. We omit max_output_tokens by default and rely on concise prompts;
# set CODEX_FORWARD_MAX_TOKENS=1 to forward the caller's cap instead.
FORWARD_MAX_TOKENS = os.environ.get("CODEX_FORWARD_MAX_TOKENS", "") == "1"

_REASONING_PREFIXES = ("gpt-5", "codex", "o1", "o3", "o4")


def _b64url_json(segment: str) -> dict:
    """Decode a base64url JWT segment into a dict (no signature check)."""
    pad = "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(segment + pad))


def _jwt_claims(token: str) -> dict:
    try:
        return _b64url_json(token.split(".")[1])
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# OAuth token source: ~/.codex/auth.json  (written by the `codex` CLI login)
# ---------------------------------------------------------------------------


class CodexAuth:
    """Reads and refreshes the ChatGPT/Codex OAuth token from disk.

    The token file format matches the ``codex`` CLI::

        {"auth_mode": "chatgpt",
         "tokens": {"id_token": ..., "access_token": ...,
                    "refresh_token": ..., "account_id": ...},
         "last_refresh": "..."}
    """

    def __init__(self, codex_home: Optional[str] = None) -> None:
        home = codex_home or os.environ.get("CODEX_HOME") or str(Path.home() / ".codex")
        self.auth_path = Path(home) / "auth.json"
        self._lock = threading.Lock()
        self._access_token: str = ""
        self._account_id: str = ""
        self._refresh_token: str = ""
        self._exp: int = 0
        self._load()

    def _load(self) -> None:
        if not self.auth_path.exists():
            raise FileNotFoundError(
                f"Codex auth file not found at {self.auth_path}. "
                "Log in with the `codex` CLI first (codex login)."
            )
        data = json.loads(self.auth_path.read_text(encoding="utf-8"))
        tokens = data.get("tokens") or {}
        self._access_token = tokens.get("access_token", "")
        self._refresh_token = tokens.get("refresh_token", "")
        self._account_id = tokens.get("account_id", "") or self._account_id_from_jwt()
        if not self._access_token:
            raise ValueError(f"No access_token in {self.auth_path}")
        self._exp = int(_jwt_claims(self._access_token).get("exp", 0))

    def _account_id_from_jwt(self) -> str:
        claims = _jwt_claims(self._access_token)
        auth = claims.get("https://api.openai.com/auth", {})
        if isinstance(auth, dict) and auth.get("chatgpt_account_id"):
            return str(auth["chatgpt_account_id"])
        return str(claims.get("account_id", ""))

    def account_id(self) -> str:
        return self._account_id

    def access_token(self) -> str:
        """Return a valid access token, refreshing if it expires within 120s."""
        with self._lock:
            if self._exp and time.time() < self._exp - 120:
                return self._access_token
            self._refresh_locked()
            return self._access_token

    def _refresh_locked(self) -> None:
        if not self._refresh_token:
            # No refresh token — hope the current one is still valid.
            logger.warning("No refresh_token available; using existing access token")
            return
        logger.info("Refreshing Codex OAuth access token")
        resp = requests.post(
            OPENAI_OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": OPENAI_OAUTH_CLIENT_ID,
            },
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        self._access_token = body.get("access_token", self._access_token)
        if body.get("refresh_token"):
            self._refresh_token = body["refresh_token"]
        self._exp = int(_jwt_claims(self._access_token).get("exp", 0))
        self._persist(body)

    def _persist(self, body: dict) -> None:
        """Write refreshed tokens back to auth.json, preserving the CLI format."""
        try:
            data = json.loads(self.auth_path.read_text(encoding="utf-8"))
            tokens = data.setdefault("tokens", {})
            tokens["access_token"] = self._access_token
            if body.get("refresh_token"):
                tokens["refresh_token"] = body["refresh_token"]
            if body.get("id_token"):
                tokens["id_token"] = body["id_token"]
            data["last_refresh"] = (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
            )
            self.auth_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # non-fatal: token still usable in-memory
            logger.warning("Could not persist refreshed token: %s", exc)


# ---------------------------------------------------------------------------
# chat.completions  ->  Responses API translation
# ---------------------------------------------------------------------------


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(_REASONING_PREFIXES)


def resolve_model(requested: str) -> str:
    """Map a requested model to a Codex-served model.

    Reasoning-model names pass through; everything else (gpt-4o, gpt-4o-mini,
    unknown) maps to ``CODEX_MODEL``.
    """
    if requested and _is_reasoning_model(requested):
        return requested
    return DEFAULT_CODEX_MODEL


def _text_format_from_response_format(response_format: Optional[dict]) -> Optional[dict]:
    """Translate an OpenAI chat ``response_format`` into a Responses ``text.format``.

    Supports ``{"type": "json_object"}`` and
    ``{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}``
    — the structured-output modes the kumiho-memory summarizer uses.
    """
    if not isinstance(response_format, dict):
        return None
    rf_type = response_format.get("type")
    if rf_type == "json_object":
        return {"type": "json_object"}
    if rf_type == "json_schema":
        js = response_format.get("json_schema", {}) or {}
        fmt: dict[str, Any] = {"type": "json_schema"}
        if js.get("name"):
            fmt["name"] = js["name"]
        if js.get("schema") is not None:
            fmt["schema"] = js["schema"]
        if "strict" in js:
            fmt["strict"] = js["strict"]
        return fmt
    return None


def build_responses_payload(
    messages: list[dict[str, Any]],
    *,
    model: str,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    response_format: Optional[dict] = None,
) -> dict[str, Any]:
    """Convert OpenAI chat messages into a Codex Responses API request body."""
    system_parts: list[str] = []
    inp: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Already in Responses content-part form or multimodal — flatten text.
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in ("text", "input_text")
            )
        if role == "system":
            if content:
                system_parts.append(content)
        elif role == "assistant":
            inp.append({
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            })
        else:  # user / tool / anything else -> user turn
            inp.append({
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            })

    payload: dict[str, Any] = {
        "model": model,
        "instructions": "\n\n".join(system_parts) if system_parts else "You are a helpful assistant.",
        "input": inp,
        "store": False,
        "stream": True,
    }
    if _is_reasoning_model(model):
        # gpt-5.4/5.5 accept none|low|medium|high|xhigh (no "minimal").
        effort = "low" if reasoning_effort == "minimal" else reasoning_effort
        payload["reasoning"] = {"effort": effort}
    if max_output_tokens and FORWARD_MAX_TOKENS:
        payload["max_output_tokens"] = int(max_output_tokens)
    text_format = _text_format_from_response_format(response_format)
    if text_format:
        payload["text"] = {"format": text_format}
    return payload


def _parse_sse_response(resp: requests.Response) -> dict[str, Any]:
    """Consume the Codex SSE stream, return {text, usage, finish_reason}."""
    text_parts: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    finish_reason = "stop"

    # text/event-stream has no charset, so requests' decode_unicode yields
    # bytes; force UTF-8 and decode defensively.
    resp.encoding = resp.encoding or "utf-8"
    for raw in resp.iter_lines(decode_unicode=True):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            evt = json.loads(data)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type", "")
        if etype == "response.output_text.delta":
            text_parts.append(evt.get("delta", ""))
        elif etype in ("response.completed", "response.done"):
            r = evt.get("response", {}) or {}
            u = r.get("usage") or {}
            usage = {
                "prompt_tokens": int(u.get("input_tokens", 0) or 0),
                "completion_tokens": int(u.get("output_tokens", 0) or 0),
                "total_tokens": int(u.get("total_tokens", 0) or 0),
            }
            if not text_parts:  # no deltas were streamed — pull from final object
                ot = r.get("output_text")
                if isinstance(ot, str) and ot:
                    text_parts.append(ot)
                else:
                    for item in r.get("output", []) or []:
                        for part in item.get("content", []) or []:
                            if part.get("type") == "output_text":
                                text_parts.append(part.get("text", ""))
        elif etype in ("error", "response.failed"):
            err = evt.get("error") or evt.get("response", {}).get("error") or evt
            raise RuntimeError(f"Codex Responses error: {json.dumps(err)[:500]}")

    return {
        "text": "".join(text_parts),
        "usage": usage,
        "finish_reason": finish_reason,
    }


def _post_and_parse(
    auth: CodexAuth,
    payload: dict[str, Any],
    *,
    timeout: float,
    responses_url: str,
) -> dict[str, Any]:
    """POST a Responses payload to the Codex backend and parse the SSE stream."""
    headers = {
        "Authorization": f"Bearer {auth.access_token()}",
        "chatgpt-account-id": auth.account_id(),
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "session_id": str(uuid.uuid4()),
        "accept": "text/event-stream",
        "content-type": "application/json",
        "user-agent": DEFAULT_ORIGINATOR,
    }
    resp = requests.post(
        os.environ.get("CODEX_RESPONSES_URL", responses_url),
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Codex HTTP {resp.status_code}: {resp.text[:1000]}")
    return _parse_sse_response(resp)


def codex_complete(
    auth: CodexAuth,
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_CODEX_MODEL,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    response_format: Optional[dict] = None,
    timeout: float = 300.0,
    responses_url: str = DEFAULT_CODEX_RESPONSES_URL,
) -> dict[str, Any]:
    """One chat completion via the Codex Responses backend. Returns {text, usage}."""
    resolved = resolve_model(model)
    payload = build_responses_payload(
        messages,
        model=resolved,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        response_format=response_format,
    )
    out = _post_and_parse(auth, payload, timeout=timeout, responses_url=responses_url)
    out["model"] = resolved
    return out


def codex_responses(
    auth: CodexAuth,
    req: dict[str, Any],
    *,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: float = 300.0,
    responses_url: str = DEFAULT_CODEX_RESPONSES_URL,
) -> dict[str, Any]:
    """Handle a native Responses API request (``client.responses.create``).

    The incoming body is already Responses-shaped; we remap the model, clamp
    the reasoning effort, normalise ``input`` (string or message list), force
    streaming, and return {text, usage, model}.
    """
    resolved = resolve_model(req.get("model", DEFAULT_CODEX_MODEL))
    raw_input = req.get("input", "")
    if isinstance(raw_input, str):
        inp: list[dict[str, Any]] = [{
            "role": "user",
            "content": [{"type": "input_text", "text": raw_input}],
        }]
    else:
        inp = raw_input  # assume already a valid Responses input array

    payload: dict[str, Any] = {
        "model": resolved,
        "input": inp,
        "store": False,
        "stream": True,
    }
    if req.get("instructions"):
        payload["instructions"] = req["instructions"]
    if _is_reasoning_model(resolved):
        effort = req.get("reasoning", {}).get("effort") if isinstance(req.get("reasoning"), dict) else None
        effort = effort or reasoning_effort
        payload["reasoning"] = {"effort": "low" if effort == "minimal" else effort}
    if req.get("text"):
        payload["text"] = req["text"]
    if req.get("max_output_tokens") and FORWARD_MAX_TOKENS:
        payload["max_output_tokens"] = int(req["max_output_tokens"])

    out = _post_and_parse(auth, payload, timeout=timeout, responses_url=responses_url)
    out["model"] = resolved
    return out


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP server
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    auth: CodexAuth = None  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(200, {
                "object": "list",
                "data": [{"id": DEFAULT_CODEX_MODEL, "object": "model", "owned_by": "openai"}],
            })
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid JSON body"}})
            return

        if self.path.endswith("/embeddings"):
            # Codex OAuth has no embeddings surface. If a real OpenAI embeddings
            # key is provided (OPENAI_EMBEDDINGS_API_KEY), forward there so the
            # harness's client-side rerank/sibling-filter keeps full quality;
            # otherwise 501 and let those optional features degrade gracefully.
            emb_key = os.environ.get("OPENAI_EMBEDDINGS_API_KEY", "")
            if not emb_key:
                self._send_json(501, {"error": {
                    "message": "No embeddings backend. Set OPENAI_EMBEDDINGS_API_KEY "
                               "to forward /v1/embeddings to OpenAI, or disable "
                               "client-side embedding features.",
                    "type": "not_implemented",
                }})
                return
            emb_base = os.environ.get("OPENAI_EMBEDDINGS_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            try:
                up = requests.post(
                    f"{emb_base}/embeddings",
                    headers={"Authorization": f"Bearer {emb_key}", "Content-Type": "application/json"},
                    json=req,
                    timeout=120,
                )
                self.send_response(up.status_code)
                self.send_header("Content-Type", "application/json")
                body = up.content
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                logger.warning("embeddings forward failed: %s", exc)
                self._send_json(502, {"error": {"message": str(exc), "type": "upstream_error"}})
            return

        # --- Native Responses API (client.responses.create) ---
        if self.path.endswith("/responses"):
            try:
                result = codex_responses(self.auth, req)
            except Exception as exc:
                logger.warning("codex_responses failed: %s", exc)
                self._send_json(502, {"error": {"message": str(exc), "type": "upstream_error"}})
                return
            usage = result["usage"]
            text = result["text"]
            self._send_json(200, {
                "id": f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "model": result.get("model", req.get("model", DEFAULT_CODEX_MODEL)),
                "output": [{
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }],
                "output_text": text,
                "usage": {
                    "input_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                },
            })
            return

        if not self.path.endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"unhandled path {self.path}"}})
            return

        messages = req.get("messages", [])
        model = req.get("model", DEFAULT_CODEX_MODEL)
        max_tokens = req.get("max_tokens") or req.get("max_completion_tokens")
        try:
            result = codex_complete(
                self.auth,
                messages,
                model=model,
                max_output_tokens=max_tokens,
                response_format=req.get("response_format"),
            )
        except Exception as exc:
            logger.warning("codex_complete failed: %s", exc)
            self._send_json(502, {"error": {"message": str(exc), "type": "upstream_error"}})
            return

        usage = result["usage"]
        self._send_json(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.get("model", model),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result.get("finish_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
            },
        })


def make_server(host: str, port: int, auth: CodexAuth) -> ThreadingHTTPServer:
    """Build (but do not start) the proxy HTTP server bound to host:port."""
    _Handler.auth = auth
    return ThreadingHTTPServer((host, port), _Handler)


def serve(host: str, port: int, auth: CodexAuth) -> None:
    httpd = make_server(host, port, auth)
    logger.info(
        "Codex OAuth proxy on http://%s:%d/v1  (model=%s, effort=%s, account=%s)",
        host, port, DEFAULT_CODEX_MODEL, DEFAULT_REASONING_EFFORT,
        auth.account_id()[:8] + "…",
    )
    print(f"Codex OAuth proxy listening on http://{host}:{port}/v1  (model={DEFAULT_CODEX_MODEL})")
    print("  export OPENAI_BASE_URL=http://%s:%d/v1" % (host, port))
    print("  export OPENAI_API_KEY=codex-oauth")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def _self_test(auth: CodexAuth) -> int:
    print(f"account_id: {auth.account_id()}")
    print(f"model:      {DEFAULT_CODEX_MODEL}  (effort={DEFAULT_REASONING_EFFORT})")
    t0 = time.perf_counter()
    result = codex_complete(
        auth,
        [
            {"role": "system", "content": "You answer with a single word."},
            {"role": "user", "content": "What is the capital of France? One word."},
        ],
        model=DEFAULT_CODEX_MODEL,
    )
    dt = (time.perf_counter() - t0) * 1000
    print(f"answer:     {result['text']!r}")
    print(f"usage:      {result['usage']}")
    print(f"latency:    {dt:.0f} ms")
    ok = "paris" in result["text"].lower()
    print("RESULT:     " + ("PASS ✅" if ok else "CHECK ⚠️ (unexpected answer)"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenAI-compatible proxy backed by Codex OAuth")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--codex-home", default=None, help="override ~/.codex")
    ap.add_argument("--self-test", action="store_true", help="one round-trip and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    auth = CodexAuth(codex_home=args.codex_home)
    if args.self_test:
        return _self_test(auth)
    serve(args.host, args.port, auth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
