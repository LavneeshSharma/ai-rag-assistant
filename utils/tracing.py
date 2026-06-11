import contextvars
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BASE_DIR = Path(__file__).resolve().parents[1]
TRACE_FILE = Path(
    os.getenv("RAG_TRACE_FILE", BASE_DIR / "evaluation" / "traces.jsonl")
)
TRACE_CHUNK_CHAR_LIMIT = int(os.getenv("RAG_TRACE_CHUNK_CHAR_LIMIT", "1200"))
TRACE_PROMPT_CHAR_LIMIT = int(os.getenv("RAG_TRACE_PROMPT_CHAR_LIMIT", "12000"))
GROQ_INPUT_COST_PER_MILLION = float(
    os.getenv("GROQ_INPUT_COST_PER_MILLION", "0.59")
)
GROQ_OUTPUT_COST_PER_MILLION = float(
    os.getenv("GROQ_OUTPUT_COST_PER_MILLION", "0.79")
)

_CURRENT_TRACE = contextvars.ContextVar("rag_current_trace", default=None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(value: Any, limit: int = TRACE_CHUNK_CHAR_LIMIT) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return f"{value[:limit]}... [truncated {len(value) - limit} chars]"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _langsmith_enabled() -> bool:
    tracing_flag = (
        os.getenv("LANGSMITH_TRACING")
        or os.getenv("LANGCHAIN_TRACING_V2")
        or ""
    ).lower()
    return tracing_flag in {"1", "true", "yes"} and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def _usage_from_response(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or {}

    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or token_usage.get("completion_tokens")
    )
    total_tokens = (
        usage.get("total_tokens")
        or token_usage.get("total_tokens")
        or (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def estimate_usage_from_text(prompt: str, response: str) -> Dict[str, int]:
    input_tokens = max(1, round(len(prompt) / 4))
    output_tokens = max(1, round(len(response) / 4))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def estimate_cost(usage: Dict[str, Optional[int]]) -> Optional[float]:
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None

    input_cost = input_tokens * GROQ_INPUT_COST_PER_MILLION / 1_000_000
    output_cost = output_tokens * GROQ_OUTPUT_COST_PER_MILLION / 1_000_000
    return round(input_cost + output_cost, 8)


def extract_usage_and_cost(
    response: Any,
    prompt: str,
    response_text: str,
) -> Dict[str, Any]:
    usage = _usage_from_response(response)
    estimated = False

    if usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        usage = estimate_usage_from_text(prompt, response_text)
        estimated = True

    return {
        "token_usage": usage,
        "token_usage_estimated": estimated,
        "estimated_cost_usd": estimate_cost(usage),
        "pricing": {
            "provider": "groq",
            "input_cost_per_million_tokens": GROQ_INPUT_COST_PER_MILLION,
            "output_cost_per_million_tokens": GROQ_OUTPUT_COST_PER_MILLION,
        },
    }


def serialize_document(doc: Any, index: int) -> Dict[str, Any]:
    metadata = getattr(doc, "metadata", {}) or {}
    content = getattr(doc, "page_content", "") or ""
    return {
        "rank": index + 1,
        "source": metadata.get("source"),
        "file_name": metadata.get("file_name")
        or os.path.basename(metadata.get("source", "")),
        "page": metadata.get("page"),
        "page_label": metadata.get("page_label"),
        "chunk_id": metadata.get("chunk_id"),
        "chunk_index": metadata.get("chunk_index"),
        "retrieval_source": metadata.get("retrieval_source"),
        "vector_score": metadata.get("vector_score"),
        "bm25_score": metadata.get("bm25_score"),
        "rerank_score": metadata.get("rerank_score"),
        "content": _truncate(content),
        "content_length": len(content),
    }


def serialize_documents(docs: Iterable[Any]) -> List[Dict[str, Any]]:
    return [serialize_document(doc, index) for index, doc in enumerate(docs)]


class RAGTrace:
    def __init__(
        self,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.trace_id = uuid.uuid4().hex
        self.name = name
        self.inputs = inputs or {}
        self.metadata = metadata or {}
        self.started_at = time.perf_counter()
        self.langsmith_run = None

    def __enter__(self):
        self._token = _CURRENT_TRACE.set(self)
        self._start_langsmith_run()
        self.record_step(
            "trace_start",
            inputs=self.inputs,
            outputs={"trace_id": self.trace_id},
            metadata=self.metadata,
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        latency = time.perf_counter() - self.started_at
        outputs = {
            "trace_id": self.trace_id,
            "latency_seconds": round(latency, 3),
            "status": "error" if exc else "ok",
        }
        if exc:
            outputs["error"] = str(exc)

        self.record_step("trace_end", outputs=outputs)
        if self.langsmith_run:
            try:
                self.langsmith_run.add_outputs(outputs)
                if exc:
                    self.langsmith_run.end(error=str(exc))
                else:
                    self.langsmith_run.end()
                self._langsmith_context.__exit__(exc_type, exc, traceback)
            except Exception:
                pass

        _CURRENT_TRACE.reset(self._token)

    def _start_langsmith_run(self) -> None:
        if not _langsmith_enabled():
            return
        try:
            from langsmith.run_helpers import trace

            project_name = os.getenv("LANGSMITH_PROJECT", "rag-evaluation")
            self._langsmith_context = trace(
                self.name,
                run_type="chain",
                inputs=_json_safe(self.inputs),
                metadata=_json_safe(
                    {
                        **self.metadata,
                        "trace_id": self.trace_id,
                    }
                ),
                project_name=project_name,
                tags=["rag", "evaluation"],
            )
            self.langsmith_run = self._langsmith_context.__enter__()
        except Exception as exc:
            self.langsmith_run = None
            self.record_step(
                "langsmith_trace_error",
                outputs={"error": str(exc)},
            )

    def record_step(
        self,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_type: str = "chain",
        latency_seconds: Optional[float] = None,
    ) -> None:
        record = {
            "trace_id": self.trace_id,
            "step": name,
            "timestamp": _utc_now(),
            "run_type": run_type,
            "inputs": _json_safe(inputs or {}),
            "outputs": _json_safe(outputs or {}),
            "metadata": _json_safe(metadata or {}),
        }
        if latency_seconds is not None:
            record["latency_seconds"] = round(latency_seconds, 3)

        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")

        if self.langsmith_run:
            try:
                child = self.langsmith_run.create_child(
                    name=name,
                    run_type=run_type,
                    inputs=record["inputs"],
                    metadata=record["metadata"],
                )
                child.add_outputs(record["outputs"])
                if latency_seconds is not None:
                    child.add_metadata({"latency_seconds": latency_seconds})
                child.end()
            except Exception:
                pass


@contextmanager
def trace_step(
    name: str,
    inputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_type: str = "chain",
):
    trace = _CURRENT_TRACE.get()
    started_at = time.perf_counter()
    try:
        yield
    except Exception as exc:
        if trace:
            trace.record_step(
                name,
                inputs=inputs,
                outputs={"error": str(exc)},
                metadata=metadata,
                run_type=run_type,
                latency_seconds=time.perf_counter() - started_at,
            )
        raise


def record_trace_step(
    name: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_type: str = "chain",
    latency_seconds: Optional[float] = None,
) -> None:
    trace = _CURRENT_TRACE.get()
    if trace:
        trace.record_step(
            name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            run_type=run_type,
            latency_seconds=latency_seconds,
        )


def truncate_prompt(prompt: str) -> str:
    return _truncate(prompt, TRACE_PROMPT_CHAR_LIMIT)
