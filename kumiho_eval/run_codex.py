"""
One-command benchmark runner backed by ChatGPT/Codex OAuth — NO API key.

Starts the Codex OAuth proxy in-process (a daemon thread), points the openai
SDK at it via ``OPENAI_BASE_URL``, then delegates to ``run_benchmarks``. Every
``run_benchmarks`` flag passes straight through.

    python -m kumiho_eval.run_codex --locomo --max-samples 1 \
        --recall-mode summarized --recall-limit 3 --graph-augmented

Notes
-----
* Answer, judge, query-reformulation, and the kumiho-memory summarizer all
  route through the proxy — i.e. the whole pipeline runs on the ChatGPT
  subscription. Configure with ``CODEX_MODEL`` (default gpt-5.5) and
  ``CODEX_REASONING_EFFORT`` (default low).
* Codex has no embeddings endpoint, so client-side embedding features are
  disabled here (sibling similarity threshold forced to 0, two-pass rerank
  off). Core recall still embeds server-side on the Kumiho tenant.
* Requires a reachable Kumiho endpoint + token (``KUMIHO_ENDPOINT`` /
  ``KUMIHO_AUTH_TOKEN``), exactly like the normal harness.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

from kumiho_eval.codex_proxy import CodexAuth, make_server, DEFAULT_CODEX_MODEL


def start_proxy_thread(host: str = "127.0.0.1", port: int = 8123) -> None:
    """Start the Codex OAuth proxy on a daemon thread and export env vars."""
    auth = CodexAuth()
    httpd = make_server(host, port, auth)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    base_url = f"http://{host}:{port}/v1"
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", "codex-oauth")
    # Give the kumiho-memory summarizer a non-empty key so it uses the OpenAI
    # adapter (which honours OPENAI_BASE_URL) instead of the unreachable
    # no-key fallback endpoint.
    os.environ.setdefault("KUMIHO_LLM_API_KEY", "codex-oauth")

    emb = "→OpenAI" if os.environ.get("OPENAI_EMBEDDINGS_API_KEY") else "off (501)"
    print(
        f"[run_codex] Codex OAuth proxy on {base_url} "
        f"(model={DEFAULT_CODEX_MODEL}, account={auth.account_id()[:8]}…, embeddings {emb})",
        file=sys.stderr,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.environ.get("CODEX_PROXY_HOST", "127.0.0.1")
    port = int(os.environ.get("CODEX_PROXY_PORT", "8123"))
    start_proxy_thread(host, port)

    from kumiho_eval.run_benchmarks import main as run_benchmarks_main

    return run_benchmarks_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
