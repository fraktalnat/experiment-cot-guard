import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from gen import generate
from gen.generate import (
    GenerationConfig,
    format_response,
    openrouter_payload,
    post_openrouter,
    rollout_task_ids,
    run_openrouter_generation,
)

OR_CFG = {
    "url": "https://openrouter.ai/api/v1/chat/completions",
    "provider": "google-ai-studio",
    "reasoning_effort": "medium",
    "concurrency": 2,
    "max_retries": 2,
    "timeout_s": 5,
}
MESSAGES = [{"role": "user", "content": "Is this code suspicious?"}]


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def body(reasoning: str | None, content: str, cost: float = 0.01) -> FakeResponse:
    return FakeResponse(
        json.dumps({
            "choices": [{"message": {"reasoning": reasoning, "content": content}}],
            "usage": {"cost": cost},
        }).encode()
    )


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(OR_CFG["url"], code, "error", {}, None)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate.time, "sleep", lambda _: None)


@pytest.fixture
def requests(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def urlopen(request: Any, timeout: float) -> FakeResponse:
        sent.append(json.loads(request.data))
        return body("weighing verify=False", "<answer>C</answer>")

    monkeypatch.setattr(generate.urllib.request, "urlopen", urlopen)
    return sent


def write_input(path: Path, task_ids: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"task_id": t, "messages": MESSAGES, "metadata": {"main_task_id": t}}) + "\n"
            for t in task_ids
        )
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_rollout_task_ids_add_draw_suffix_only_for_several_draws() -> None:
    assert rollout_task_ids("t", 1) == ["t"]
    assert rollout_task_ids("t", 3) == ["t:0", "t:1", "t:2"]


def test_format_response_matches_batch_api_format() -> None:
    assert format_response("R", "<answer>C</answer>") == "<think>R</think><answer>C</answer>"


def test_payload_pins_provider_effort_and_temperature() -> None:
    payload = openrouter_payload(MESSAGES, "google/gemini-3-flash", 0.6, OR_CFG)
    assert payload == {
        "model": "google/gemini-3-flash",
        "messages": MESSAGES,
        "temperature": 0.6,
        "reasoning": {"effort": "medium"},
        "usage": {"include": True},
        "provider": {"order": ["google-ai-studio"], "allow_fallbacks": False},
    }


def test_post_returns_reasoning_content_and_cost(requests: list) -> None:
    assert post_openrouter({}, OR_CFG, "key") == ("weighing verify=False", "<answer>C</answer>", 0.01)


def test_post_treats_missing_reasoning_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate.urllib.request, "urlopen", lambda *a, **k: body(None, "<answer>A</answer>"))
    assert post_openrouter({}, OR_CFG, "key")[0] == ""


def test_post_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter([http_error(503), http_error(429), body("r", "<answer>A</answer>")])

    def urlopen(*args: Any, **kwargs: Any) -> FakeResponse:
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(generate.urllib.request, "urlopen", urlopen)
    assert post_openrouter({}, OR_CFG, "key")[1] == "<answer>A</answer>"


def test_post_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(*args: Any, **kwargs: Any) -> FakeResponse:
        raise http_error(503)

    monkeypatch.setattr(generate.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        post_openrouter({}, OR_CFG, "key")


def test_post_raises_on_non_retryable_error_without_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def urlopen(*args: Any, **kwargs: Any) -> FakeResponse:
        calls.append(1)
        raise http_error(401)

    monkeypatch.setattr(generate.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        post_openrouter({}, OR_CFG, "key")
    assert len(calls) == 1


def test_post_raises_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate.urllib.request, "urlopen", lambda *a, **k: body("r", ""))
    with pytest.raises(ValueError):
        post_openrouter({}, OR_CFG, "key")


def test_run_writes_n_rows_per_record_with_draw_ids(
    tmp_path: Path, requests: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    write_input(tmp_path / "in.jsonl", ["a", "b"])
    config = GenerationConfig(model="google/gemini-3-flash", backend="openrouter", temperature=0.6, n=4)

    run_openrouter_generation(
        str(tmp_path / "in.jsonl"), str(tmp_path / "out.jsonl"), config, OR_CFG, str(tmp_path / "ckpt.jsonl")
    )

    rows = read_rows(tmp_path / "out.jsonl")
    assert sorted(r["task_id"] for r in rows) == [f"{t}:{i}" for t in "ab" for i in range(4)]
    assert all(r["messages"][:-1] == MESSAGES for r in rows)
    assert all(r["messages"][-1] == {"role": "assistant", "content": "<think>weighing verify=False</think><answer>C</answer>"} for r in rows)
    assert all(r["metadata"]["main_task_id"] == r["task_id"].split(":")[0] for r in rows)
    assert len(requests) == 8


def test_run_resumes_from_checkpoint(
    tmp_path: Path, requests: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    write_input(tmp_path / "in.jsonl", ["a"])
    done = {"task_id": "a:0", "messages": MESSAGES + [{"role": "assistant", "content": "<think></think>done"}], "metadata": {}}
    (tmp_path / "ckpt.jsonl").write_text(json.dumps(done) + "\n")
    config = GenerationConfig(model="google/gemini-3-flash", backend="openrouter", n=2)

    run_openrouter_generation(
        str(tmp_path / "in.jsonl"), str(tmp_path / "out.jsonl"), config, OR_CFG, str(tmp_path / "ckpt.jsonl")
    )

    assert len(requests) == 1
    assert sorted(r["task_id"] for r in read_rows(tmp_path / "out.jsonl")) == ["a:0", "a:1"]


def test_run_keeps_finished_rows_when_a_request_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    write_input(tmp_path / "in.jsonl", ["a"])
    replies = iter([body("r", "<answer>C</answer>")])

    def urlopen(*args: Any, **kwargs: Any) -> FakeResponse:
        reply = next(replies, None)
        if reply is None:
            raise http_error(401)
        return reply

    monkeypatch.setattr(generate.urllib.request, "urlopen", urlopen)
    config = GenerationConfig(model="google/gemini-3-flash", backend="openrouter", n=2)
    sequential = {**OR_CFG, "concurrency": 1}

    with pytest.raises(urllib.error.HTTPError):
        run_openrouter_generation(
            str(tmp_path / "in.jsonl"), str(tmp_path / "out.jsonl"), config, sequential, str(tmp_path / "ckpt.jsonl")
        )

    assert len(read_rows(tmp_path / "ckpt.jsonl")) == 1
    assert not (tmp_path / "out.jsonl").exists()
