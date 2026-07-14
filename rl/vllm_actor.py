from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import time
import traceback
from typing import Any, Dict, Optional


@dataclass
class ActorCompletion:
    token_ids: list[int]
    text: str = ""


@dataclass
class ActorRequestOutput:
    outputs: list[ActorCompletion]


def _shutdown_engine(llm: Any) -> None:
    if llm is None:
        return
    for obj in (llm, getattr(llm, "llm_engine", None)):
        for attr in ("shutdown", "close"):
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass


def _handle_actor_request(state: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    command = str(request.get("command", ""))
    if command == "ping":
        return {"ready": True, "engine_loaded": state.get("llm") is not None}
    if command == "load_engine":
        _shutdown_engine(state.get("llm"))
        from vllm import LLM  # type: ignore

        start = time.perf_counter()
        state["llm"] = LLM(**dict(request["llm_kwargs"]))
        return {"load_sec": max(0.0, time.perf_counter() - start)}
    if command == "unload_engine":
        _shutdown_engine(state.get("llm"))
        state["llm"] = None
        return {"unloaded": True}
    if command == "generate":
        llm = state.get("llm")
        if llm is None:
            raise RuntimeError("vLLM actor engine is not loaded")
        from vllm import SamplingParams  # type: ignore

        sampling_params = SamplingParams(**dict(request["sampling_kwargs"]))
        prompt_token_ids = [list(map(int, row)) for row in request["prompt_token_ids"]]
        try:
            outputs = llm.generate(
                prompts=[{"prompt_token_ids": ids} for ids in prompt_token_ids],
                sampling_params=sampling_params,
            )
        except TypeError:
            outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        serialized = []
        for output in outputs:
            completions = list(getattr(output, "outputs", []) or [])
            serialized.append(
                [
                    {
                        "token_ids": [int(token) for token in list(getattr(completion, "token_ids", []) or [])],
                        "text": str(getattr(completion, "text", "") or ""),
                    }
                    for completion in completions
                ]
            )
        return {"outputs": serialized}
    if command == "sleep":
        llm = state.get("llm")
        if llm is None:
            return {"slept": False, "reason": "engine_not_loaded"}
        sleep_fn = getattr(llm, "sleep", None)
        if not callable(sleep_fn):
            raise RuntimeError("vLLM actor engine does not expose sleep()")
        sleep_fn(level=int(request.get("level", 1)))
        return {"slept": True, "level": int(request.get("level", 1))}
    if command == "wake_up":
        llm = state.get("llm")
        if llm is None:
            return {"woke": False, "reason": "engine_not_loaded"}
        wake_fn = getattr(llm, "wake_up", None)
        if not callable(wake_fn):
            raise RuntimeError("vLLM actor engine does not expose wake_up()")
        tags = request.get("tags")
        wake_fn(tags=None if tags is None else list(tags))
        return {"woke": True, "tags": tags}
    if command == "close":
        _shutdown_engine(state.get("llm"))
        state["llm"] = None
        state["closed"] = True
        return {"closed": True}
    raise ValueError(f"Unsupported vLLM actor command {command!r}")


def _actor_main(connection) -> None:
    state: Dict[str, Any] = {"llm": None, "closed": False}
    connection.send({"kind": "ready"})
    try:
        while not state["closed"]:
            request = connection.recv()
            request_id = request.get("request_id")
            try:
                payload = _handle_actor_request(state, request)
                connection.send({"request_id": request_id, "ok": True, "payload": payload})
            except Exception as exc:
                connection.send(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    }
                )
    except EOFError:
        pass
    finally:
        _shutdown_engine(state.get("llm"))
        connection.close()


class VLLMActorClient:
    def __init__(
        self,
        *,
        start_method: str = "spawn",
        request_timeout_sec: float = 600.0,
        shutdown_timeout_sec: float = 30.0,
    ) -> None:
        self.start_method = str(start_method)
        self.request_timeout_sec = float(request_timeout_sec)
        self.shutdown_timeout_sec = float(shutdown_timeout_sec)
        self._connection = None
        self._process = None
        self._request_id = 0

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        if self.is_alive:
            return
        context = mp.get_context(self.start_method)
        parent, child = context.Pipe(duplex=True)
        # vLLM may create its own worker processes; a daemon multiprocessing
        # process is not allowed to create children.
        process = context.Process(target=_actor_main, args=(child,), name="fitmotn-vllm-actor", daemon=False)
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        if not parent.poll(self.request_timeout_sec):
            self.close(force=True)
            raise TimeoutError("Timed out waiting for the vLLM actor process to start")
        try:
            message = parent.recv()
        except EOFError as exc:
            exitcode = process.exitcode
            self.close(force=True)
            raise RuntimeError(f"vLLM actor exited during startup with code {exitcode}") from exc
        if message.get("kind") != "ready":
            self.close(force=True)
            raise RuntimeError(f"Unexpected vLLM actor startup response: {message}")

    def _request(self, command: str, **payload: Any) -> Dict[str, Any]:
        self.start()
        if self._connection is None or self._process is None:
            raise RuntimeError("vLLM actor process is unavailable")
        if not self._process.is_alive():
            raise RuntimeError(f"vLLM actor process exited unexpectedly with code {self._process.exitcode}")
        self._request_id += 1
        request_id = self._request_id
        self._connection.send({"request_id": request_id, "command": command, **payload})
        if not self._connection.poll(self.request_timeout_sec):
            self.close(force=True)
            raise TimeoutError(f"Timed out waiting for vLLM actor command={command!r}")
        try:
            response = self._connection.recv()
        except EOFError as exc:
            exitcode = self._process.exitcode
            self.close(force=True)
            raise RuntimeError(
                f"vLLM actor exited during command={command!r} with code {exitcode}"
            ) from exc
        if response.get("request_id") != request_id:
            self.close(force=True)
            raise RuntimeError(f"Mismatched vLLM actor response id: expected={request_id}, response={response}")
        if not response.get("ok"):
            raise RuntimeError(
                f"vLLM actor command={command!r} failed: {response.get('error_type')}: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return dict(response.get("payload") or {})

    def load_engine(self, llm_kwargs: Dict[str, Any]) -> float:
        result = self._request("load_engine", llm_kwargs=dict(llm_kwargs))
        return float(result.get("load_sec", 0.0))

    def ping(self) -> Dict[str, Any]:
        return self._request("ping")

    def unload_engine(self) -> None:
        if self.is_alive:
            self._request("unload_engine")

    def sleep(self, *, level: int = 1) -> Dict[str, Any]:
        return self._request("sleep", level=int(level))

    def wake_up(self, *, tags: Optional[list[str]] = None) -> Dict[str, Any]:
        return self._request("wake_up", tags=tags)

    def generate(self, *, prompt_token_ids: list[list[int]], sampling_kwargs: Dict[str, Any]) -> list[ActorRequestOutput]:
        result = self._request(
            "generate",
            prompt_token_ids=prompt_token_ids,
            sampling_kwargs=dict(sampling_kwargs),
        )
        return [
            ActorRequestOutput(
                outputs=[
                    ActorCompletion(token_ids=list(item.get("token_ids") or []), text=str(item.get("text") or ""))
                    for item in completions
                ]
            )
            for completions in result.get("outputs", [])
        ]

    def close(self, *, force: bool = False) -> None:
        process = self._process
        connection = self._connection
        if process is None:
            return
        if process.is_alive() and not force:
            try:
                self._request("close")
            except Exception:
                force = True
        process.join(timeout=self.shutdown_timeout_sec)
        if process.is_alive():
            process.terminate()
            process.join(timeout=self.shutdown_timeout_sec)
        if connection is not None:
            connection.close()
        self._connection = None
        self._process = None
