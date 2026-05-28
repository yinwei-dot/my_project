from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import requests
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")
EXPERTS = ("temporal", "discourse", "precondition", "commonsense")
# Round 1: temporal+precondition 自由提案（开放搜索）；
# Round 2+: discourse+commonsense 对上轮因果边做 causal/reject 二分类；
# 提前停止条件：discourse 与 commonsense 的因果边集完全一致；
# Judge: 对最后一轮因果并集做 causal/temporal 二分类。
DEFAULT_ROUND_EXPERTS: list[list[str]] = [
    ["temporal", "precondition"],
    ["discourse", "commonsense"],
    ["discourse", "commonsense"],
]
PIPELINE_SIGNATURE_VERSION = "timeline_reorder_v1_expert_refine_v3_binary_classify_v1_judge_binary_v1_strict_reason_v1_strong_filter_v1_temporal_builder_v1_staged_rounds_v3"
TOKEN_ESTIMATE_BYTES_PER_TOKEN = 3.2
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "api_url": "https://api.siliconflow.cn/v1/chat/completions",
    "expert_model": "Pro/deepseek-ai/DeepSeek-V3.2",
    "judge_model": "Pro/deepseek-ai/DeepSeek-V3.2",
    "expert_max_tokens": 1400,
    "judge_max_tokens": 3600,
    "rounds": 2,          # 专家讨论轮数：Round 1（独立提案）+ Round 2+（复核）；Judge 单独运行，不计入此数
    "round_experts": DEFAULT_ROUND_EXPERTS,
    "resume": True,
    "workers": 1,         # 第三方代理有 Cloudflare 限流，串行避免 403
    "expert_workers": 1,  # 同上，专家级也串行
    "expert_rpm_limit": 500,
    "expert_tpm_limit": 2000000,
    "judge_rpm_limit": 500,
    "judge_tpm_limit": 2000000,
    "rate_limit_window_seconds": 60.0,
    "request_timeout": 300,
    "judge_timeout": 300,
    "request_max_retries": 5,      # 503/500 服务端拥塞时需要更多重试机会
    "retry_backoff_factor": 1.0,   # 快速重试（V3.2 响应稳定）
    "temperature": 0.0,
    "seed": None,  # 不传 seed：避免禁用 SiliconFlow 服务端前缀 KV-Cache，维持正常速度
    "enable_thinking": False,  # Qwen3 思维链开关；False = 关闭 <think>，减少 token 消耗
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = ENV_LINE_RE.match(raw_line)
        if not match:
            continue
        key, value = match.groups()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def first_config_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


# 进度条最后一次 _render 是否未以 \n 结尾（inline 模式）
_progress_line_dirty: bool = False


class _ProgressAwareStreamHandler(logging.StreamHandler):
    """若进度条已原地渲染但行尾没有 \\n，则先写 \\n 再输出日志，避免日志追在进度条同一行。"""

    def emit(self, record: logging.LogRecord) -> None:
        global _progress_line_dirty
        if _progress_line_dirty:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except Exception:  # noqa: BLE001
                pass
            _progress_line_dirty = False
        super().emit(record)


def configure_logging(log_dir: Path, log_name: str = "pipeline") -> Path:
    """创建日志文件；log_name 同时作为子目录名（不同业务隔离）。"""
    sub_dir = log_dir / log_name
    sub_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = sub_dir / f"{log_name}_{timestamp}.log"

    if logging.getLogger().handlers:
        return log_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[
            _ProgressAwareStreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def log_file_only(logger: logging.Logger, level: int, message: str, *args: Any) -> None:
    if not logger.isEnabledFor(level):
        return

    record = logger.makeRecord(
        logger.name,
        level,
        fn="",
        lno=0,
        msg=message,
        args=args,
        exc_info=None,
    )

    current: logging.Logger | None = logger
    file_handlers: list[logging.FileHandler] = []
    seen: set[int] = set()
    while current is not None:
        for handler in current.handlers:
            if isinstance(handler, logging.FileHandler) and id(handler) not in seen:
                file_handlers.append(handler)
                seen.add(id(handler))
        if not current.propagate:
            break
        current = current.parent

    emitted = False
    for handler in file_handlers:
        if handler.level in {logging.NOTSET, 0} or record.levelno >= handler.level:
            handler.handle(record)
            emitted = True

    if not emitted:
        logger.log(level, message, *args)


def read_json_utf8(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json_utf8(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")


def write_text_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_dumps(payload: Any, *, indent: int = 2) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def render_prompt_template(template: str, context: dict[str, str]) -> str:
    rendered = template
    for key, value in context.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _extract_balanced_json_fragment(text: str, start_index: int) -> str | None:
    pairs = {"{": "}", "[": "]"}
    if start_index < 0 or start_index >= len(text) or text[start_index] not in pairs:
        return None

    stack: list[str] = [text[start_index]]
    in_string = False
    escaping = False
    for index in range(start_index + 1, len(text)):
        char = text[index]
        if in_string:
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in pairs:
            stack.append(char)
            continue
        if char in {"}", "]"}:
            if not stack:
                return None
            open_char = stack.pop()
            if pairs[open_char] != char:
                return None
            if not stack:
                return text[start_index : index + 1]
    return None


def smart_parse_json(text: str) -> Any:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None

    candidates: list[str] = [raw_text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    without_think = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    if without_think and without_think != raw_text:
        candidates.append(without_think)

    seen_candidates: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        for start_index, char in enumerate(candidate):
            if char not in "[{":
                continue
            fragment = _extract_balanced_json_fragment(candidate, start_index)
            if not fragment or fragment in seen_candidates:
                continue
            seen_candidates.add(fragment)
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                continue
    return None


def extract_doc_id(text: str) -> str:
    match = re.match(r"^([A-Z]+\d+)", str(text or "").strip())
    if match:
        return match.group(1)
    raise ValueError(f"无法从名称中提取 doc_id: {text}")


def event_id_num(event_id: str) -> int:
    match = re.fullmatch(r"E(\d+)", str(event_id or "").strip())
    if not match:
        return 10**9
    return int(match.group(1))


def doc_stem_sort_key(stem: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([A-Z]+)(\d+)", stem)
    if match:
        return (match.group(1), int(match.group(2)), stem)
    return (stem, 10**9, stem)


def _normalize_api_url(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized.rstrip("/") + "/chat/completions"


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path
    data_dir: Path
    output_dir: Path
    log_dir: Path
    api_key: str
    api_url: str
    expert_model: str
    judge_model: str
    expert_max_tokens: int
    judge_max_tokens: int
    rounds: int
    resume: bool
    workers: int
    expert_workers: int
    expert_rpm_limit: int
    expert_tpm_limit: int
    judge_rpm_limit: int
    judge_tpm_limit: int
    rate_limit_window_seconds: float
    request_timeout: int
    judge_timeout: int
    request_max_retries: int
    retry_backoff_factor: float
    temperature: float
    seed: int | None
    enable_thinking: bool
    limit: int | None
    doc_ids: tuple[str, ...]
    prompt_dir: Path
    experts: tuple[str, ...] = EXPERTS
    round_experts: tuple[tuple[str, ...], ...] = ()

    def experts_for_round(self, round_index: int) -> tuple[str, ...]:
        """返回指定轮次应参与的专家列表。

        - 若 round_experts 已配置：round_index 在范围内取对应组；
          超出范围时沿用最后一组配置（避免回退到全部专家，
          确保时序/前提专家只出现在第 1 轮）。
        - 若未配置：使用 self.experts。
        """
        if self.round_experts:
            if round_index < len(self.round_experts):
                return self.round_experts[round_index]
            return self.round_experts[-1]
        return self.experts

    @property
    def run_signature(self) -> str:
        payload = {
            "pipeline_signature_version": PIPELINE_SIGNATURE_VERSION,
            "expert_model": self.expert_model,
            "judge_model": self.judge_model,
            "rounds": self.rounds,
            "experts": list(self.experts),
            "round_experts": [list(g) for g in self.round_experts],
            "prompt_dir": str(self.prompt_dir),
        }
        return sha256_text(str(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "log_dir": str(self.log_dir),
            "api_url": self.api_url,
            "expert_model": self.expert_model,
            "judge_model": self.judge_model,
            "expert_max_tokens": self.expert_max_tokens,
            "judge_max_tokens": self.judge_max_tokens,
            "rounds": self.rounds,
            "resume": self.resume,
            "workers": self.workers,
            "expert_workers": self.expert_workers,
            "expert_rpm_limit": self.expert_rpm_limit,
            "expert_tpm_limit": self.expert_tpm_limit,
            "judge_rpm_limit": self.judge_rpm_limit,
            "judge_tpm_limit": self.judge_tpm_limit,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
            "request_timeout": self.request_timeout,
            "judge_timeout": self.judge_timeout,
            "request_max_retries": self.request_max_retries,
            "retry_backoff_factor": self.retry_backoff_factor,
            "temperature": self.temperature,
            "seed": self.seed,
            "enable_thinking": self.enable_thinking,
            "limit": self.limit,
            "doc_ids": list(self.doc_ids),
            "prompt_dir": str(self.prompt_dir),
            "experts": list(self.experts),
            "round_experts": [list(g) for g in self.round_experts],
            "pipeline_signature_version": PIPELINE_SIGNATURE_VERSION,
            "run_signature": self.run_signature,
        }


def build_config(args: Any, root_dir: Path) -> AppConfig:
    load_env_file(root_dir / ".env")
    data_dir = Path(getattr(args, "data_dir", "") or (root_dir / "data" / "call_llm")).resolve()
    output_dir = Path(getattr(args, "output_dir", "") or (root_dir / "output")).resolve()
    log_dir = Path(getattr(args, "log_dir", "") or (root_dir / "log")).resolve()
    prompt_dir_default = root_dir / "prompts"

    api_key = getattr(args, "api_key", None) or os.getenv("API_KEY", "")
    api_url = _normalize_api_url(
        getattr(args, "api_url", None)
        or getattr(args, "base_url", None)
        or os.getenv("API_URL", "")
        or DEFAULT_PIPELINE_CONFIG["api_url"]
    )
    expert_model = (
        getattr(args, "expert_model", None)
        or getattr(args, "model", None)
        or os.getenv("CAUSAL_EXPERT_MODEL", "")
        or DEFAULT_PIPELINE_CONFIG["expert_model"]
    )
    judge_model = (
        getattr(args, "judge_model", None)
        or getattr(args, "model", None)
        or os.getenv("CAUSAL_JUDGE_MODEL", "")
        or DEFAULT_PIPELINE_CONFIG["judge_model"]
        or expert_model
    )
    expert_max_tokens = max(
        1,
        int(
            getattr(args, "expert_max_tokens", None)
            or os.getenv("CAUSAL_EXPERT_MAX_TOKENS", "")
            or DEFAULT_PIPELINE_CONFIG["expert_max_tokens"]
        ),
    )
    judge_max_tokens = max(
        1,
        int(
            getattr(args, "judge_max_tokens", None)
            or os.getenv("CAUSAL_JUDGE_MAX_TOKENS", "")
            or DEFAULT_PIPELINE_CONFIG["judge_max_tokens"]
        ),
    )

    doc_ids = tuple(filter(None, [item.strip() for item in getattr(args, "doc_ids", []) or []]))
    limit = getattr(args, "limit", None)
    limit = int(limit) if limit is not None else None
    rounds = max(
        1,
        int(
            getattr(args, "rounds", None)
            or os.getenv("CAUSAL_DISCUSSION_ROUNDS", "")
            or DEFAULT_PIPELINE_CONFIG["rounds"]
        ),
    )
    resume = (
        bool(getattr(args, "resume", None))
        if getattr(args, "resume", None) is not None
        else env_bool("CAUSAL_RESUME", DEFAULT_PIPELINE_CONFIG["resume"])
    )
    workers = max(
        1,
        int(
            getattr(args, "workers", None)
            or os.getenv("CAUSAL_WORKERS", "")
            or DEFAULT_PIPELINE_CONFIG["workers"]
        ),
    )
    expert_workers = max(
        1,
        int(
            getattr(args, "expert_workers", None)
            or os.getenv("CAUSAL_EXPERT_WORKERS", "")
            or DEFAULT_PIPELINE_CONFIG["expert_workers"]
        ),
    )
    expert_rpm_limit = max(
        0,
        int(
            first_config_value(
                getattr(args, "expert_rpm_limit", None),
                getattr(args, "rpm_limit", None),
                os.getenv("CAUSAL_EXPERT_RPM", ""),
                DEFAULT_PIPELINE_CONFIG["expert_rpm_limit"],
            )
        ),
    )
    expert_tpm_limit = max(
        0,
        int(
            first_config_value(
                getattr(args, "expert_tpm_limit", None),
                getattr(args, "tpm_limit", None),
                os.getenv("CAUSAL_EXPERT_TPM", ""),
                DEFAULT_PIPELINE_CONFIG["expert_tpm_limit"],
            )
        ),
    )
    judge_rpm_limit = max(
        0,
        int(
            first_config_value(
                getattr(args, "judge_rpm_limit", None),
                getattr(args, "rpm_limit", None),
                os.getenv("CAUSAL_JUDGE_RPM", ""),
                DEFAULT_PIPELINE_CONFIG["judge_rpm_limit"],
            )
        ),
    )
    judge_tpm_limit = max(
        0,
        int(
            first_config_value(
                getattr(args, "judge_tpm_limit", None),
                getattr(args, "tpm_limit", None),
                os.getenv("CAUSAL_JUDGE_TPM", ""),
                DEFAULT_PIPELINE_CONFIG["judge_tpm_limit"],
            )
        ),
    )
    rate_limit_window_seconds = float(
        getattr(args, "rate_limit_window_seconds", None)
        or os.getenv("CAUSAL_RATE_LIMIT_WINDOW_SECONDS", "")
        or DEFAULT_PIPELINE_CONFIG["rate_limit_window_seconds"]
    )
    request_timeout = int(
        getattr(args, "timeout", None)
        or os.getenv("CAUSAL_REQUEST_TIMEOUT", "")
        or DEFAULT_PIPELINE_CONFIG["request_timeout"]
    )
    judge_timeout = max(
        request_timeout,
        int(
            getattr(args, "judge_timeout", None)
            or os.getenv("CAUSAL_JUDGE_TIMEOUT", "")
            or DEFAULT_PIPELINE_CONFIG["judge_timeout"]
        ),
    )
    request_max_retries = max(
        0,
        int(
            getattr(args, "max_retries", None)
            or os.getenv("CAUSAL_REQUEST_MAX_RETRIES", "")
            or DEFAULT_PIPELINE_CONFIG["request_max_retries"]
        ),
    )
    retry_backoff_factor = max(
        0.0,
        float(
            getattr(args, "retry_backoff_factor", None)
            or os.getenv("CAUSAL_RETRY_BACKOFF_FACTOR", "")
            or DEFAULT_PIPELINE_CONFIG["retry_backoff_factor"]
        ),
    )
    temperature = float(
        getattr(args, "temperature", None)
        or os.getenv("CAUSAL_TEMPERATURE", "")
        or DEFAULT_PIPELINE_CONFIG["temperature"]
    )
    _enable_thinking_raw = getattr(args, "enable_thinking", None)
    if _enable_thinking_raw is not None:
        enable_thinking = bool(_enable_thinking_raw)
    else:
        _env_et = os.getenv("CAUSAL_ENABLE_THINKING", "")
        if _env_et:
            enable_thinking = _env_et.strip().lower() in {"1", "true", "yes", "on"}
        else:
            enable_thinking = bool(DEFAULT_PIPELINE_CONFIG["enable_thinking"])
    _seed_raw = (
        getattr(args, "seed", None)
        or os.getenv("CAUSAL_SEED", "")
    )
    if _seed_raw is None or _seed_raw == "":
        _seed_raw = DEFAULT_PIPELINE_CONFIG.get("seed")
    seed: int | None = int(_seed_raw) if _seed_raw is not None and str(_seed_raw) != "" else None
    prompt_dir = Path(
        getattr(args, "prompt_dir", None)
        or os.getenv("CAUSAL_PROMPT_DIR", "")
        or str(prompt_dir_default)
    ).resolve()

    missing: list[str] = []
    if not api_key:
        missing.append("API_KEY / --api-key")
    if not api_url:
        missing.append("API_URL / --api-url")
    if not expert_model:
        missing.append("CAUSAL_EXPERT_MODEL / --expert-model")
    if not judge_model:
        missing.append("CAUSAL_JUDGE_MODEL / --judge-model")
    if missing:
        raise ValueError("当前版本仅保留 LLM 调用，请补充配置: " + ", ".join(missing))

    return AppConfig(
        root_dir=root_dir.resolve(),
        data_dir=data_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        api_key=api_key,
        api_url=api_url,
        expert_model=expert_model,
        judge_model=judge_model,
        expert_max_tokens=expert_max_tokens,
        judge_max_tokens=judge_max_tokens,
        rounds=rounds,
        resume=resume,
        workers=workers,
        expert_workers=expert_workers,
        expert_rpm_limit=expert_rpm_limit,
        expert_tpm_limit=expert_tpm_limit,
        judge_rpm_limit=judge_rpm_limit,
        judge_tpm_limit=judge_tpm_limit,
        rate_limit_window_seconds=rate_limit_window_seconds,
        request_timeout=request_timeout,
        judge_timeout=judge_timeout,
        request_max_retries=request_max_retries,
        retry_backoff_factor=retry_backoff_factor,
        temperature=temperature,
        seed=seed,
        enable_thinking=enable_thinking,
        limit=limit,
        doc_ids=doc_ids,
        prompt_dir=prompt_dir,
        round_experts=tuple(
            tuple(str(e) for e in g)
            for g in (DEFAULT_PIPELINE_CONFIG.get("round_experts") or [])
        ),
    )


def discover_input_files(config: AppConfig) -> list[Path]:
    if not config.data_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {config.data_dir}")
    files = [
        path
        for path in config.data_dir.glob("*.json")
        if path.name not in {"build_summary.json"}
    ]
    files = sorted(files, key=lambda path: doc_stem_sort_key(path.stem))
    if config.doc_ids:
        wanted = set(config.doc_ids)
        files = [path for path in files if path.stem in wanted]
    if config.limit is not None:
        files = files[: config.limit]
    return files


@dataclass
class LLMCallResult:
    success: bool
    raw_text: str
    parsed_json: object | None
    error: str | None = None
    total_tokens: int | None = None


@dataclass
class TokenReservation:
    timestamp: float
    tokens: int
    active: bool = True


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / TOKEN_ESTIMATE_BYTES_PER_TOKEN))


def estimate_chat_tokens(messages: list[dict[str, str]]) -> int:
    total = 12
    for message in messages:
        total += 6
        total += estimate_text_tokens(message.get("role", ""))
        total += estimate_text_tokens(str(message.get("content", "")))
    return max(1, total)


def extract_usage_total_tokens(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    total_tokens = usage.get("total_tokens")
    if isinstance(total_tokens, int) and total_tokens > 0:
        return total_tokens
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, int) and prompt_tokens >= 0 and isinstance(completion_tokens, int) and completion_tokens >= 0:
        combined = prompt_tokens + completion_tokens
        return combined if combined > 0 else None
    return None


class SlidingWindowRateLimiter:
    def __init__(
        self,
        *,
        rpm_limit: int,
        tpm_limit: int,
        window_seconds: float,
        logger: logging.Logger | None = None,
    ) -> None:
        if rpm_limit < 0:
            raise ValueError("rpm_limit 不能小于 0")
        if tpm_limit < 0:
            raise ValueError("tpm_limit 不能小于 0")
        if rpm_limit <= 0 and tpm_limit <= 0:
            raise ValueError("rpm_limit 和 tpm_limit 不能同时关闭")
        if window_seconds <= 0:
            raise ValueError("window_seconds 必须大于 0")

        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        self.logger = logger
        self.lock = threading.Lock()
        self.request_timestamps: deque[float] = deque()
        self.token_entries: deque[TokenReservation] = deque()
        self.token_total = 0

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.request_timestamps and self.request_timestamps[0] <= cutoff:
            self.request_timestamps.popleft()
        while self.token_entries and self.token_entries[0].timestamp <= cutoff:
            reservation = self.token_entries.popleft()
            reservation.active = False
            self.token_total -= reservation.tokens

    def _compute_wait_locked(self, now: float, requested_tokens: int) -> tuple[float, list[str]]:
        wait_until = now
        reasons: list[str] = []

        if self.rpm_limit > 0 and len(self.request_timestamps) >= self.rpm_limit:
            reasons.append("RPM")
            wait_until = max(wait_until, self.request_timestamps[0] + self.window_seconds)

        if self.tpm_limit > 0 and self.token_total + requested_tokens > self.tpm_limit:
            reasons.append("TPM")
            excess = self.token_total + requested_tokens - self.tpm_limit
            released = 0
            token_wait_until = now
            for reservation in self.token_entries:
                released += reservation.tokens
                token_wait_until = reservation.timestamp + self.window_seconds
                if released >= excess:
                    break
            wait_until = max(wait_until, token_wait_until)

        return max(0.0, wait_until - now), reasons

    def acquire(self, *, requested_tokens: int, request_label: str = "") -> TokenReservation | None:
        reserved_tokens = max(1, min(requested_tokens, self.tpm_limit)) if self.tpm_limit > 0 else 0
        has_logged_wait = False
        while True:
            with self.lock:
                now = time.monotonic()
                self._prune_locked(now)
                wait_seconds, reasons = self._compute_wait_locked(now, reserved_tokens)
                if wait_seconds <= 0:
                    reservation = TokenReservation(timestamp=now, tokens=reserved_tokens) if self.tpm_limit > 0 else None
                    if self.rpm_limit > 0:
                        self.request_timestamps.append(now)
                    if reservation is not None:
                        self.token_entries.append(reservation)
                        self.token_total += reserved_tokens
                    return reservation

            if not has_logged_wait and wait_seconds >= 1.0 and self.logger is not None:
                reason_text = "+".join(reasons) if reasons else "rate-limit"
                label_text = f" [{request_label}]" if request_label else ""
                log_file_only(self.logger, logging.INFO, "触发 %s 预等待 %.1fs%s", reason_text, wait_seconds, label_text)
                has_logged_wait = True
            time.sleep(wait_seconds if wait_seconds > 0 else 0.01)

    def reconcile_usage(self, reservation: TokenReservation | None, actual_tokens: int | None) -> None:
        if reservation is None or actual_tokens is None or self.tpm_limit <= 0:
            return

        normalized_tokens = max(1, min(actual_tokens, self.tpm_limit))
        with self.lock:
            now = time.monotonic()
            self._prune_locked(now)
            if not reservation.active:
                return
            delta = normalized_tokens - reservation.tokens
            if delta == 0:
                return
            reservation.tokens = normalized_tokens
            self.token_total += delta


class OpenAICompatibleClient:
    def __init__(
        self,
        config: AppConfig,
        logger: logging.Logger,
        *,
        rate_limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.rate_limiter = rate_limiter
        self._session = requests.Session()
        self._session.trust_env = False  # 跳过系统代理，直连 API

    def _retry_delay_seconds(self, attempt_index: int) -> float:
        factor = max(0.0, self.config.retry_backoff_factor)
        base = factor * (2 ** attempt_index)
        # 加 ±50% 随机抖动，防止并发重试的雷群效应
        jitter = base * random.uniform(-0.5, 0.5)
        return max(0.0, base + jitter)

    def chat_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        request_label: str = "",
        max_tokens: int | None = None,
        timeout: int | None = None,
    ) -> LLMCallResult:
        if not self.config.api_url or not self.config.api_key or not model:
            return LLMCallResult(False, "", None, "缺少 api_url / api_key / model 配置")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        estimated_tokens = estimate_chat_tokens(messages)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        # Qwen3 思维链：受 enable_thinking 配置控制
        if "qwen3" in model.lower():
            payload["enable_thinking"] = self.config.enable_thinking
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        body = json.dumps(payload).encode("utf-8")
        max_attempts = max(1, self.config.request_max_retries + 1)
        effective_timeout = timeout if timeout is not None and timeout > 0 else self.config.request_timeout
        label = request_label or model
        last_raw_text = ""
        last_error: str | None = None

        for attempt_index in range(max_attempts):
            reservation: TokenReservation | None = None
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            try:
                if self.rate_limiter is not None:
                    reservation = self.rate_limiter.acquire(requested_tokens=estimated_tokens, request_label=label)
                resp = self._session.post(
                    self.config.api_url,
                    data=body,
                    headers=headers,
                    timeout=effective_timeout,
                )
                resp.raise_for_status()
                response_payload = resp.json()
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else 0
                detail = exc.response.text if exc.response is not None else ""
                last_raw_text = detail
                last_error = f"HTTP {status_code}"
                should_retry = status_code in RETRYABLE_HTTP_STATUS_CODES and attempt_index < max_attempts - 1
                if should_retry:
                    delay = self._retry_delay_seconds(attempt_index)
                    self.logger.warning(
                        "LLM HTTP 错误，将在 %.1fs 后重试 (%s/%s) [%s]: %s %s",
                        delay,
                        attempt_index + 1,
                        self.config.request_max_retries,
                        label,
                        status_code,
                        detail[:200],
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue
                self.logger.warning("LLM HTTP 错误 [%s]: %s %s", label, status_code, detail[:400])
                return LLMCallResult(False, detail, None, last_error)
            except Exception as exc:
                last_error = str(exc)
                should_retry = attempt_index < max_attempts - 1
                if should_retry:
                    delay = self._retry_delay_seconds(attempt_index)
                    self.logger.warning(
                        "LLM 调用失败，将在 %.1fs 后重试 (%s/%s) [%s]: %s",
                        delay,
                        attempt_index + 1,
                        self.config.request_max_retries,
                        label,
                        exc,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    continue
                self.logger.warning("LLM 调用失败 [%s]: %s", label, exc)
                return LLMCallResult(False, last_raw_text, None, last_error)

            total_tokens = extract_usage_total_tokens(response_payload)
            if self.rate_limiter is not None:
                self.rate_limiter.reconcile_usage(reservation, total_tokens)

            choices = response_payload.get("choices", []) if isinstance(response_payload, dict) else []
            raw_text = ""
            if choices and isinstance(choices[0], dict):
                raw_text = str(choices[0].get("message", {}).get("content", ""))
            parsed_json = smart_parse_json(raw_text)
            if parsed_json is None:
                last_raw_text = raw_text
                last_error = "LLM 输出不是合法 JSON。"
                should_retry = attempt_index < max_attempts - 1
                preview = raw_text[:240].replace("\r", " ").replace("\n", " ")
                if should_retry:
                    delay = self._retry_delay_seconds(attempt_index)
                    self.logger.warning(
                        "LLM 返回无法解析的 JSON，将在 %.1fs 后重试 (%s/%s) [%s]: %s",
                        delay,
                        attempt_index + 1,
                        self.config.request_max_retries,
                        label,
                        preview,
                    )
                    # 修复型多轮请求：把损坏的输出 + 修复指令追加进对话，
                    # 告知模型上次哪里格式有问题，对小模型效果明显优于原样重发。
                    repair_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": raw_text[:2000]},
                        {"role": "user", "content": "你的输出无法被解析为合法 JSON。请重新只输出一个完整合法的 JSON 对象，不要附加任何说明、思考或 Markdown 代码块。"},
                    ]
                    body = json.dumps({**payload, "messages": repair_messages}).encode("utf-8")
                    if delay > 0:
                        time.sleep(delay)
                    continue
                self.logger.warning("LLM 返回无法解析的 JSON [%s]: %s", label, preview)
                return LLMCallResult(False, raw_text, None, last_error, total_tokens=total_tokens)
            return LLMCallResult(True, raw_text, parsed_json, total_tokens=total_tokens)

        return LLMCallResult(False, last_raw_text, None, last_error or "LLM 调用失败")


def create_checkpoint(*, source_hash: str, run_signature: str) -> dict[str, Any]:
    return {
        "source_hash": source_hash,
        "run_signature": run_signature,
        "last_completed_round": -1,
        "completed_experts": {},
        "judge_done": False,
    }


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json_utf8(path)
    return payload if isinstance(payload, dict) else None


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    write_json_utf8(path, payload)


def checkpoint_matches(payload: dict[str, Any] | None, *, source_hash: str) -> bool:
    """Resume/cache 命中仅基于输入文档内容（source_hash）匹配。"""
    if not payload:
        return False
    return payload.get("source_hash") == source_hash


def mark_expert_completed(payload: dict[str, Any], round_index: int, expert_name: str) -> None:
    completed = payload.setdefault("completed_experts", {})
    round_key = str(round_index)
    current = completed.setdefault(round_key, [])
    if expert_name not in current:
        current.append(expert_name)
    payload["last_completed_round"] = max(int(payload.get("last_completed_round", -1)), round_index)


def mark_judge_completed(payload: dict[str, Any]) -> None:
    payload["judge_done"] = True


@dataclass
class ActiveDocProgress:
    total_steps: int
    completed_steps: int = 0
    stage_label: str = "准备中"


class BatchProgressDisplay:
    def __init__(self, total_docs: int, total_steps: int):
        self.total_docs = max(1, total_docs)
        self.total_steps = max(1, total_steps)
        self.completed_docs = 0
        self.completed_steps = 0
        self.start_time: float | None = None  # 首次 start_doc 时才开始计时
        self.active_doc_name = "-"
        self.active = ActiveDocProgress(total_steps=0, completed_steps=0, stage_label="等待中")
        self.active_counted_in_total = True
        self.rendered = False
        self._lock = threading.Lock()

    def _visible_completed_steps(self) -> int:
        if self.active_counted_in_total:
            return self.completed_steps
        return min(self.total_steps, self.completed_steps + self.active.completed_steps)

    def _estimate_remaining_seconds(self) -> float | None:
        visible_completed_steps = self._visible_completed_steps()
        if visible_completed_steps <= 0:
            return None
        remaining_steps = max(0, self.total_steps - visible_completed_steps)
        if remaining_steps == 0:
            return 0.0
        elapsed = max(time.monotonic() - self.start_time, 1e-6) if self.start_time else 1e-6
        rate = visible_completed_steps / elapsed
        if rate <= 0:
            return None
        return remaining_steps / rate

    def _format_remaining_duration(self) -> str:
        remaining_seconds = self._estimate_remaining_seconds()
        if remaining_seconds is None:
            return "--:--:--"
        total_seconds = max(0, int(round(remaining_seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _format_bar(self, ratio: float, width: int = 14) -> tuple[str, int]:
        bounded = min(1.0, max(0.0, ratio))
        filled = min(width, int(round(bounded * width)))
        bar = "#" * filled + "-" * (width - filled)
        percent = int(round(bounded * 100))
        return bar, percent

    def _display_name(self, name: str, max_width: int) -> str:
        if len(name) <= max_width:
            return name
        return name[: max_width - 1] + "…"

    def _build_line(self) -> str:
        visible_completed_steps = self._visible_completed_steps()
        total_ratio = visible_completed_steps / self.total_steps
        total_bar, total_percent = self._format_bar(total_ratio)

        sent_done = self.active.completed_steps
        sent_total = self.active.total_steps

        elapsed = int(time.monotonic() - self.start_time) if self.start_time else 0
        eh, erem = divmod(elapsed, 3600)
        em, es = divmod(erem, 60)
        elapsed_str = f"{eh:02d}:{em:02d}:{es:02d}"
        eta_str = self._format_remaining_duration()

        return (
            f"总进度[{total_bar}] {total_percent}% | "
            f"文档 {self.completed_docs}/{self.total_docs} | "
            f"步骤 {sent_done}/{sent_total} | "
            f"已用 {elapsed_str} 剩余 {eta_str}"
        )

    def _render(self, *, final: bool = False) -> None:
        global _progress_line_dirty
        line = self._build_line()
        prefix = "\r\x1b[2K" if self.rendered else ""
        suffix = "\n" if final else ""
        sys.stdout.write(prefix + line + suffix)
        sys.stdout.flush()
        _progress_line_dirty = not final
        self.rendered = True

    def start_doc(self, doc_name: str, total_steps: int, completed_steps: int = 0, stage_label: str = "准备中") -> None:
        with self._lock:
            if self.start_time is None:
                self.start_time = time.monotonic()
            self.active_doc_name = doc_name
            self.active = ActiveDocProgress(
                total_steps=max(1, total_steps),
                completed_steps=min(max(0, completed_steps), max(1, total_steps)),
                stage_label=stage_label,
            )
            self.active_counted_in_total = False
            self._render()

    def set_stage(self, stage_label: str) -> None:
        with self._lock:
            self.active.stage_label = stage_label

    def advance_step(self, stage_label: str) -> None:
        with self._lock:
            self.active.completed_steps = min(self.active.total_steps, self.active.completed_steps + 1)
            self.active.stage_label = stage_label
            self._render()

    def skip_doc(self, doc_name: str, total_steps: int, reason: str = "缓存命中") -> None:
        with self._lock:
            self.active_doc_name = doc_name
            self.active = ActiveDocProgress(total_steps=max(1, total_steps), completed_steps=max(1, total_steps), stage_label=reason)
            self.completed_steps = min(self.total_steps, self.completed_steps + max(1, total_steps))
            self.completed_docs += 1
            self.active_counted_in_total = True
            self._render(final=self.completed_docs >= self.total_docs)

    def finish_doc(self, stage_label: str = "完成") -> None:
        with self._lock:
            self.active.completed_steps = self.active.total_steps
            self.active.stage_label = stage_label
            self.completed_steps = min(self.total_steps, self.completed_steps + self.active.total_steps)
            self.completed_docs += 1
        self.active_counted_in_total = True
        self._render(final=self.completed_docs >= self.total_docs)


def _format_edges(edges: list[dict[str, Any]]) -> list[str]:
    if not edges:
        return ["- （无）"]
    return [f"- {edge.get('cause', '?')} -> {edge.get('effect', '?')}  {edge.get('reason', '')}" for edge in edges]


class TraceExporter:
    def __init__(self, output_dir: Path) -> None:
        self.trace_dir = output_dir / "discussion_traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, doc_slug: str, trace_bundle: dict[str, Any]) -> Path:
        path = self.trace_dir / f"{doc_slug}_discussion.json"
        # 导出时清理冗余字段，round 改为 1-indexed（内部仍 0-indexed）
        rounds = []
        for entry in trace_bundle.get("rounds", []):
            e: dict[str, Any] = {"round": entry.get("round", 0) + 1}
            # 保留 expert_traces，清理其中的 prompt_metadata，同时将 round 转为 1-indexed
            expert_traces = {}
            for name, tr in entry.get("expert_traces", {}).items():
                clean_tr = {k: v for k, v in tr.items() if k != "prompt_metadata"}
                if "round" in clean_tr:
                    clean_tr["round"] = clean_tr["round"] + 1
                expert_traces[name] = clean_tr
            if expert_traces:
                e["expert_traces"] = expert_traces
            rounds.append(e)
        # 按时间顺序组织输出：元数据 → rounds → judge
        display_bundle: dict[str, Any] = {}
        for k, v in trace_bundle.items():
            if k not in ("rounds", "judge"):
                display_bundle[k] = v
        display_bundle["rounds"] = rounds
        # 清理 judge 中的 rejected 字段
        judge = trace_bundle.get("judge", {})
        if judge:
            clean_judge = dict(judge)
            parsed = clean_judge.get("parsed_result", {})
            if isinstance(parsed, dict) and "rejected" in parsed:
                clean_judge["parsed_result"] = {k: v for k, v in parsed.items() if k != "rejected"}
            display_bundle["judge"] = clean_judge
        write_json_utf8(path, display_bundle)
        return path

    def export_text(self, doc_slug: str, trace_bundle: dict[str, Any]) -> Path:
        path = self.trace_dir / f"{doc_slug}_discussion.txt"
        lines: list[str] = [f"文档: {trace_bundle.get('doc_name', doc_slug)} [规则注入]", ""]
        for round_entry in trace_bundle.get("rounds", []):
            round_num = round_entry.get("round", 0)
            lines.extend([f"===== Round {round_num + 1} ===== [规则注入]", ""])
            traces = round_entry.get("expert_traces", {})
            for expert_name, trace in traces.items():
                lines.extend(
                    [
                        f"--- {expert_name} --- [规则注入]",
                        "[system_prompt] [规则注入]",
                        trace.get("system_prompt", ""),
                        "",
                        "[user_prompt] [规则注入]",
                        trace.get("user_prompt", ""),
                        "",
                        "[raw_response] [规则注入]",
                        trace.get("raw_response", ""),
                        "",
                    ]
                )
        judge_trace = trace_bundle.get("judge", {})
        if judge_trace.get("system_prompt") or judge_trace.get("user_prompt"):
            lines.extend(
                [
                    "===== Judge ===== [规则注入]",
                    "[system_prompt] [规则注入]",
                    judge_trace.get("system_prompt", ""),
                    "",
                    "[user_prompt] [规则注入]",
                    judge_trace.get("user_prompt", ""),
                    "",
                    "[raw_response] [规则注入]",
                    judge_trace.get("raw_response", ""),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "===== Judge ===== [规则注入]",
                    "[skipped] " + judge_trace.get("note", "候选池为空，跳过 Judge。") + " [规则注入]",
                    "",
                ]
            )
        write_text_utf8(path, "\n".join(lines))
        return path

    def export_summary(self, doc_slug: str, trace_bundle: dict[str, Any]) -> Path:
        path = self.trace_dir / f"{doc_slug}_discussion_summary.txt"
        lines: list[str] = [f"文档: {trace_bundle.get('doc_name', doc_slug)}"]
        doc_text = trace_bundle.get("doc_text", "")
        if doc_text:
            lines.extend(["", "===== 全文 =====", "", doc_text.strip()])
        for round_entry in trace_bundle.get("rounds", []):
            round_num = round_entry.get("round", 0)
            lines.extend(["", f"Round {round_num + 1}"])
            for expert_name, result in round_entry.get("expert_results", {}).items():
                causal = result.get("causal_edges", [])
                temporal = result.get("temporal_edges", [])
                lines.append(f"[{expert_name}] 因果边")
                lines.extend(_format_edges(causal))
                if round_num >= 1:
                    lines.append(f"[{expert_name}] 时序边")
                    lines.extend(_format_edges(temporal))
        judge_result = trace_bundle.get("judge", {}).get("parsed_result", {})
        lines.append("Judge 因果边")
        lines.extend(_format_edges(judge_result.get("causal", [])))
        lines.append("Judge 时序边")
        lines.extend(_format_edges(judge_result.get("temporal", [])))
        write_text_utf8(path, "\n".join(lines))
        return path


def resolve_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def build_pipeline_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run causal_inference_v4 pipeline.")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-ids", nargs="*", default=[])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--expert-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--expert-max-tokens", type=int, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt-dir", default=None)
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=None,
        help="同时设置 expert/judge 的 RPM 上限；若提供更细粒度参数，则以更细粒度参数为准。",
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=None,
        help="同时设置 expert/judge 的 TPM 上限；若提供更细粒度参数，则以更细粒度参数为准。",
    )
    parser.add_argument("--expert-rpm-limit", type=int, default=None)
    parser.add_argument("--expert-tpm-limit", type=int, default=None)
    parser.add_argument("--judge-rpm-limit", type=int, default=None)
    parser.add_argument("--judge-tpm-limit", type=int, default=None)
    parser.add_argument("--rate-limit-window-seconds", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--judge-timeout", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-backoff-factor", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--enable-thinking", action="store_true", default=None, dest="enable_thinking", help="为 Qwen3 开启 <think> 思维链（默认关闭）")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None, help="并发文档数（默认1，串行）")
    parser.add_argument("--expert-workers", type=int, default=None, help="同轮内并发专家数（默认1，串行）")
    return parser


def create_runtime(parsed_args: argparse.Namespace) -> tuple[AppConfig, logging.Logger, Path]:
    args = parsed_args
    if getattr(args, "no_resume", False):
        args.resume = False
    root_dir = resolve_root_dir()
    config = build_config(args, root_dir)
    log_path = configure_logging(config.log_dir)
    logger = logging.getLogger("causal_inference_v4")
    logger.info("日志文件: %s", log_path)
    log_file_only(logger, logging.INFO, "运行配置: %s", json_dumps(config.to_dict()))
    return config, logger, log_path


__all__ = [
    "AppConfig",
    "BatchProgressDisplay",
    "DEFAULT_PIPELINE_CONFIG",
    "EXPERTS",
    "LLMCallResult",
    "OpenAICompatibleClient",
    "PIPELINE_SIGNATURE_VERSION",
    "SlidingWindowRateLimiter",
    "TokenReservation",
    "TraceExporter",
    "build_config",
    "build_pipeline_arg_parser",
    "checkpoint_matches",
    "configure_logging",
    "create_checkpoint",
    "create_runtime",
    "discover_input_files",
    "doc_stem_sort_key",
    "env_bool",
    "event_id_num",
    "extract_doc_id",
    "json_dumps",
    "load_checkpoint",
    "load_env_file",
    "log_file_only",
    "mark_expert_completed",
    "mark_judge_completed",
    "read_json_utf8",
    "read_text_utf8",
    "render_prompt_template",
    "resolve_root_dir",
    "save_checkpoint",
    "sha256_file",
    "sha256_text",
    "smart_parse_json",
    "write_json_utf8",
    "write_text_utf8",
]