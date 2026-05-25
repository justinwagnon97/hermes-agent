#!/usr/bin/env python3
"""File Tools Module - LLM agent file manipulation tools."""

import errno
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from agent.file_safety import get_read_block_error
from tools.binary_extensions import has_binary_extension
from tools.file_operations import (
    ShellFileOperations,
    normalize_read_pagination,
    normalize_search_pagination,
)
from tools import file_state
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


_EXPECTED_WRITE_ERRNOS = {errno.EACCES, errno.EPERM, errno.EROFS}

# ---------------------------------------------------------------------------
# Read-size guard: cap the character count returned to the model.
# We're model-agnostic so we can't count tokens; characters are a safe proxy.
# 100K chars ≈ 25–35K tokens across typical tokenisers.  Files larger than
# this in a single read are a context-window hazard — the model should use
# offset+limit to read the relevant section.
#
# Configurable via config.yaml:  file_read_max_chars: 200000
# ---------------------------------------------------------------------------
_DEFAULT_MAX_READ_CHARS = 100_000
_max_read_chars_cached: int | None = None


def _get_max_read_chars() -> int:
    """Return the configured max characters per file read.

    Reads ``file_read_max_chars`` from config.yaml on first call, caches
    the result for the lifetime of the process.  Falls back to the
    built-in default if the config is missing or invalid.
    """
    global _max_read_chars_cached
    if _max_read_chars_cached is not None:
        return _max_read_chars_cached
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        val = cfg.get("file_read_max_chars")
        if isinstance(val, (int, float)) and val > 0:
            _max_read_chars_cached = int(val)
            return _max_read_chars_cached
    except Exception:
        pass
    _max_read_chars_cached = _DEFAULT_MAX_READ_CHARS
    return _max_read_chars_cached

# If the total file size exceeds this AND the caller didn't specify a narrow
# range (limit <= 200), we include a hint encouraging targeted reads.
_LARGE_FILE_HINT_BYTES = 512_000  # 512 KB

# ---------------------------------------------------------------------------
# Device path blocklist — reading these hangs the process (infinite output
# or blocking on input).  Checked by path only (no I/O).
# ---------------------------------------------------------------------------
_BLOCKED_DEVICE_PATHS = frozenset({
    # Infinite output — never reach EOF
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    # Blocks waiting for input
    "/dev/stdin", "/dev/tty", "/dev/console",
    # Nonsensical to read
    "/dev/stdout", "/dev/stderr",
    # fd aliases
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


def _resolve_path(filepath: str, task_id: str = "default") -> Path:
    """Resolve a path relative to TERMINAL_CWD (the worktree base directory)
    instead of the main repository root.
    """
    return _resolve_path_for_task(filepath, task_id)


def _get_live_tracking_cwd(task_id: str = "default") -> str | None:
    """Return the task's live terminal cwd for bookkeeping when available."""
    try:
        from tools.terminal_tool import _resolve_container_task_id
        container_key = _resolve_container_task_id(task_id)
    except Exception:
        container_key = task_id

    with _file_ops_lock:
        cached = _file_ops_cache.get(container_key) or _file_ops_cache.get(task_id)
    if cached is not None:
        live_cwd = getattr(getattr(cached, "env", None), "cwd", None) or getattr(
            cached, "cwd", None
        )
        if live_cwd:
            return live_cwd

    try:
        from tools.terminal_tool import _active_environments, _env_lock

        with _env_lock:
            env = _active_environments.get(task_id)
        if env is not None:
            live_cwd = getattr(getattr(env, "env", None), "cwd", None) or getattr(
                env, "cwd", None
            )
            if live_cwd:
                return live_cwd
    except Exception:
        pass

    return None


_RESOLVED_CWD_CACHE: dict[str, tuple[float, Path]] = {}


def _resolve_cwd_for_task(task_id: str = "default") -> Path:
    """Resolve the current working directory for a task.

    Priority:
      1. Live tracking cwd (from terminal env) — captures ``cd`` effects.
      2. Task-level override from ``_task_env_overrides``.
      3. Global cwd from the environment config.

    Results are cached up to 5 seconds to avoid excessive filesystem I/O.
    """
    # Live cwd is cheap (just an attribute read) but we still cache the
    # resolved result so the Path creation + normpath is shared.
    from tools.terminal_tool import _task_env_overrides

    now = time.time()
    cache_key = task_id
    cached = _RESOLVED_CWD_CACHE.get(cache_key)
    if cached and (now - cached[0]) < 5.0:
        return cached[1]

    live = _get_live_tracking_cwd(task_id)
    if live:
        resolved = Path(live).resolve()
        _RESOLVED_CWD_CACHE[cache_key] = (now, resolved)
        return resolved

    overrides = _task_env_overrides.get(task_id, {})
    if overrides and overrides.get("cwd"):
        resolved = Path(overrides["cwd"]).resolve()
        _RESOLVED_CWD_CACHE[cache_key] = (now, resolved)
        return resolved

    config = _get_env_config()
    resolved = Path(config["cwd"]).resolve()
    _RESOLVED_CWD_CACHE[cache_key] = (now, resolved)
    return resolved


# ── Cached env config ──────────────────────────────────────────────────


_ENV_CONFIG_CACHE: tuple[float, dict] | None = None


def _get_env_config() -> dict:
    """Return the environment config, cached for 30 seconds."""
    global _ENV_CONFIG_CACHE
    now = time.time()
    if _ENV_CONFIG_CACHE and (now - _ENV_CONFIG_CACHE[0]) < 30.0:
        return _ENV_CONFIG_CACHE[1]

    from tools.terminal_tool import _get_or_create_task_env
    config = _get_or_create_task_env("__env_config__")
    _ENV_CONFIG_CACHE = (now, config)
    return config


# ── Per-task terminal environment management ──────────────────────────

_file_ops_cache: dict[str, "ShellFileOperations"] = {}
_file_ops_lock = threading.Lock()
_env_lock = threading.Lock()
_active_environments: dict[str, object] = {}
_last_activity: dict[str, float] = {}
_creation_locks: dict[str, threading.Lock] = {}
_creation_locks_lock = threading.Lock()
_cleanup_started = False
_cleanup_lock = threading.Lock()
CLEANUP_INTERVAL = 600  # 10 minutes
ENV_TTL = 3600  # 1 hour


def _start_cleanup_thread():
    """Start a background thread to clean up stale environments."""
    global _cleanup_started
    with _cleanup_lock:
        if _cleanup_started:
            return
        _cleanup_started = True

    def _cleanup_loop():
        while True:
            time.sleep(CLEANUP_INTERVAL)
            now = time.time()
            stale_ids = []
            with _env_lock:
                for tid, last_active in list(_last_activity.items()):
                    if tid == "__env_config__":
                        continue
                    if now - last_active > ENV_TTL:
                        stale_ids.append(tid)
                for tid in stale_ids:
                    _active_environments.pop(tid, None)
                    _last_activity.pop(tid, None)

            if stale_ids:
                with _file_ops_lock:
                    for tid in stale_ids:
                        _file_ops_cache.pop(tid, None)
                        _file_ops_cache.pop(f"__env_{tid}", None)
                logger.info(
                    "Cleaned up %d stale environments", len(stale_ids)
                )

    thread = threading.Thread(target=_cleanup_loop, daemon=True)
    thread.start()


def _get_file_ops(task_id: str = "default") -> "ShellFileOperations":
    """Get or create ShellFileOperations for the given task."""
    with _file_ops_lock:
        cached = _file_ops_cache.get(task_id)
        if cached is not None:
            return cached

    # Use a per-task creation lock to prevent duplicate environment creation
    # when multiple threads try to get file_ops for the same task_id.
    with _creation_locks_lock:
        if task_id not in _creation_locks:
            _creation_locks[task_id] = threading.Lock()
        task_lock = _creation_locks[task_id]

    with task_lock:
        # Double-check: another thread may have created it while we waited
        with _env_lock:
            if task_id in _active_environments:
                _last_activity[task_id] = time.time()
                terminal_env = _active_environments[task_id]
            else:
                terminal_env = None

        if terminal_env is None:
            from tools.terminal_tool import _task_env_overrides

            config = _get_env_config()
            env_type = config["env_type"]
            overrides = _task_env_overrides.get(task_id, {})

            if env_type == "docker":
                image = overrides.get("docker_image") or config["docker_image"]
            elif env_type == "singularity":
                image = overrides.get("singularity_image") or config["singularity_image"]
            elif env_type == "modal":
                image = overrides.get("modal_image") or config["modal_image"]
            elif env_type == "daytona":
                image = overrides.get("daytona_image") or config["daytona_image"]
            else:
                image = ""

            cwd = overrides.get("cwd") or config["cwd"]
            logger.info("Creating new %s environment for task %s...", env_type, task_id[:8])

            container_config = None
            if env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}:
                container_config = {
                    "container_cpu": config.get("container_cpu", 1),
                    "container_memory": config.get("container_memory", 5120),
                    "container_disk": config.get("container_disk", 51200),
                    "container_persistent": config.get("container_persistent", True),
                    "vercel_runtime": config.get("vercel_runtime", ""),
                    "docker_volumes": config.get("docker_volumes", []),
                    "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
                    "docker_forward_env": config.get("docker_forward_env", []),
                    "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
                }

            ssh_config = None
            if env_type == "ssh":
                ssh_config = {
                    "host": config.get("ssh_host", ""),
                    "user": config.get("ssh_user", ""),
                    "port": config.get("ssh_port", 22),
                    "key": config.get("ssh_key", ""),
                    "persistent": config.get("ssh_persistent", False),
                }

            local_config = None
            if env_type == "local":
                local_config = {
                    "persistent": config.get("local_persistent", False),
                }

            terminal_env = _create_environment(
                env_type=env_type,
                image=image,
                cwd=cwd,
                timeout=config["timeout"],
                ssh_config=ssh_config,
                container_config=container_config,
                local_config=local_config,
                task_id=task_id,
                host_cwd=config.get("host_cwd"),
            )

            with _env_lock:
                _active_environments[task_id] = terminal_env
                _last_activity[task_id] = time.time()

            _start_cleanup_thread()
            logger.info("%s environment ready for task %s", env_type, task_id[:8])

    # Build file_ops from the (guaranteed live) environment and cache it
    file_ops = ShellFileOperations(terminal_env)
    with _file_ops_lock:
        _file_ops_cache[task_id] = file_ops
    return file_ops


def clear_file_ops_cache(task_id: str = None):
    """Clear the file operations cache."""
    with _file_ops_lock:
        if task_id:
            _file_ops_cache.pop(task_id, None)
        else:
            _file_ops_cache.clear()


def read_file_tool(path: str, offset: int = 1, limit: int = 500, task_id: str = "default") -> str:
    """Read a file with pagination and line numbers."""
    try:
        offset, limit = normalize_read_pagination(offset, limit)

        # ── Device path guard ─────────────────────────────────────────
        # Block paths that would hang the process (infinite output,
        # blocking on input).  Pure path check — no I/O.
        if _is_blocked_device(path):
            return json.dumps({
                "error": (
                    f"Cannot read '{path}': this is a device file that would "
                    "block or produce infinite output."
                ),
            })

        _resolved = _resolve_path_for_task(path, task_id)

        # ── Binary file guard ─────────────────────────────────────────
        # Block binary files by extension (no I/O).
        if has_binary_extension(str(_resolved)):
            _ext = _resolved.suffix.lower()
            return json.dumps({
                "error": (
                    f"Cannot read binary file '{path}' ({_ext}). "
                    "Use vision_analyze for images, or terminal to inspect binary files."
                ),
            })

        # ── Hermes internal path guard ────────────────────────────────
        # Prevent prompt injection via catalog or hub metadata files.
        block_error = get_read_block_error(path)
        if block_error:
            return json.dumps({"error": block_error})

        # ── Dedup check ───────────────────────────────────────────────
        # If we already read this exact (path, offset, limit) and the
        # file hasn't been modified since, return a lightweight stub
        # instead of re-sending the same content.  Saves context tokens.
        resolved_str = str(_resolved)
        dedup_key = (resolved_str, offset, limit)
        with _read_tracker_lock:
            task_data = _read_tracker.setdefault(task_id, {
                "last_key": None, "consecutive": 0,
                "read_history": set(), "dedup": {},
                "dedup_hits": {}, "read_timestamps": {},
            })
            # Backward-compat for pre-existing tracker entries that predate
            # dedup_hits/read_timestamps (long-lived task or crossed an
            # upgrade boundary).
            if "dedup_hits" not in task_data:
                task_data["dedup_hits"] = {}
            if "read_timestamps" not in task_data:
                task_data["read_timestamps"] = {}
            cached_mtime = task_data.get("dedup", {}).get(dedup_key)

        if cached_mtime is not None:
            try:
                current_mtime = os.path.getmtime(resolved_str)
                if current_mtime == cached_mtime:
                    # Count repeated stub returns so weak tool-followers that
                    # ignore the "refer to earlier result" hint don't burn
                    # their iteration budget in an infinite read loop.  After
                    # 2 stubs for the same key we escalate to a hard block
                    # mirroring the count>=4 path on real reads.
                    with _read_tracker_lock:
                        hits = task_data["dedup_hits"].get(dedup_key, 0) + 1
                        task_data["dedup_hits"][dedup_key] = hits
                        _cap_read_tracker_data(task_data)

                    if hits >= 2:
                        return json.dumps({
                            "error": (
                                f"BLOCKED: You have called read_file on this "
                                f"exact region {hits + 1} times and the file "
                                "has NOT changed. STOP calling read_file for "
                                "this path — the content from your earlier "
                                "read_file result in this conversation is "
                                "still current. Proceed with your task using "
                                "the information you already have."
                            ),
                            "path": path,
                            "already_read": hits + 1,
                        }, ensure_ascii=False)

                    return json.dumps({
                        "status": "unchanged",
                        "message": _READ_DEDUP_STATUS_MESSAGE,
                        "path": path,
                        "dedup": True,
                        "content_returned": False,
                    }, ensure_ascii=False)
            except OSError:
                pass  # stat failed — fall through to full read

        # ── Perform the read ──────────────────────────────────────────
        file_ops = _get_file_ops(task_id)
        result = file_ops.read_file(path, offset, limit)
        result_dict = result.to_dict()

        # ── Character-count guard ─────────────────────────────────────
        # We're model-agnostic so we can't count tokens; characters are
        # the best proxy we have.  If the read produced an unreasonable
        # amount of content, reject it and tell the model to narrow down.
        # Note: we check the formatted content (with line-number prefixes),
        # not the raw file size, because that's what actually enters context.
        # Check BEFORE redaction to avoid expensive regex on huge content.
        content_len = len(result.content or "")
        file_size = result_dict.get("file_size", 0)
        if result_dict.get("truncated") or content_len > _get_max_read_chars():
            return json.dumps({
                "error": (
                    f"File content exceeds maximum read size "
                    f"({_get_max_read_chars():,} chars). "
                    "Use offset and limit to read specific sections."
                ),
            })

        if not result_dict.get("error"):
            result_dict["content"] = redact_sensitive_text(
                result.content, code_file=True
            )

        # Add large-file hint when the file is big AND the caller didn't
        # ask for a narrow range.  Avoids the model feeling lectured when
        # it deliberately read a reasonable subset.
        if (file_size > _LARGE_FILE_HINT_BYTES
                and not result_dict.get("error")
                and result_dict.get("truncated")):
            result_dict.setdefault("_hint", (
                f"This file is large ({file_size:,} bytes). "
                "Consider reading only the section you need with offset and limit "
                "to keep context usage efficient."
            ))

        # ── Track for consecutive-loop detection ──────────────────────
        read_key = ("read", path, offset, limit)
        with _read_tracker_lock:
            # Ensure "dedup" / "dedup_hits" keys exist (backward compat with
            # old tracker state from pre-dedup-guard sessions).
            if "dedup" not in task_data:
                task_data["dedup"] = {}
            if "dedup_hits" not in task_data:
                task_data["dedup_hits"] = {}
            # Real read succeeded — this key is no longer in a stub-loop, so
            # reset its hit counter.  (File either changed or stat failed
            # earlier and we fell through.)
            task_data["dedup_hits"].pop(dedup_key, None)
            task_data["read_history"].add((path, offset, limit))
            if task_data["last_key"] == read_key:
                task_data["consecutive"] += 1
            else:
                task_data["last_key"] = read_key
                task_data["consecutive"] = 1
            count = task_data["consecutive"]

            # Store mtime at read time for two purposes:
            # 1. Dedup: skip identical re-reads of unchanged files.
            # 2. Staleness: warn on write/patch if the file changed since
            #    the agent last read it (external edit, concurrent agent, etc.).
            try:
                _mtime_now = os.path.getmtime(resolved_str)
                task_data["dedup"][dedup_key] = _mtime_now
                task_data.setdefault("read_timestamps", {})[resolved_str] = _mtime_now
            except OSError:
                pass  # Can't stat — skip tracking for this entry

            # Bound the per-task containers so a long CLI session doesn't
            # accumulate megabytes of dict/set state.  See _cap_read_tracker_data.
            _cap_read_tracker_data(task_data)

        # Cross-agent file-state registry (separate from per-task read
        # tracker above): records that THIS agent has read this path so
        # write/patch can detect sibling-subagent writes that happened
        # after our read.  Partial read when offset>1 or the read was
        # truncated (large file with more content than limit covered).
        # Outside the _read_tracker_lock so the registry's own locking
        # isn't nested under ours.
        try:
            _partial = (offset > 1) or bool(result_dict.get("truncated"))
            file_state.record_read(task_id, resolved_str, partial=_partial)
        except Exception:
            logger.debug("file_state.record_read failed", exc_info=True)

        if count >= 4:
            # Hard block: stop returning content to break the loop
            return json.dumps({
                "error": (
                    f"BLOCKED: You have read this exact file region {count} times in a row. "
                    "The content has NOT changed. You already have this information. "
                    "STOP re-reading and proceed with your task."
                ),
                "path": path,
                "already_read": count,
            }, ensure_ascii=False)
        elif count >= 3:
            result_dict["_warning"] = (
                f"You have read this exact file region {count} times consecutively. "
                "The content has not changed since your last read. Use the information you already have. "
                "If you are stuck in a loop, stop reading and proceed with writing or responding."
            )

        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))




def reset_file_dedup(task_id: str = None):
    """Clear the deduplication cache for file reads.

    Called after context compression — the original read content has been
    summarised away, so the model needs the full content if it reads the
    same file again.  Without this, reads after compression would return
    a "file unchanged" stub pointing at content that no longer exists in
    context.

    Call with a task_id to clear just that task, or without to clear all.
    """
    with _read_tracker_lock:
        if task_id:
            task_data = _read_tracker.get(task_id)
            if task_data:
                if "dedup" in task_data:
                    task_data["dedup"].clear()
                if "dedup_hits" in task_data:
                    task_data["dedup_hits"].clear()
        else:
            for task_data in _read_tracker.values():
                if "dedup" in task_data:
                    task_data["dedup"].clear()
                if "dedup_hits" in task_data:
                    task_data["dedup_hits"].clear()


def notify_other_tool_call(task_id: str = "default"):
    """Reset consecutive read/search counter for a task.

    Called by the tool dispatcher (model_tools.py) whenever a tool OTHER
    than read_file / search_files is executed.  This ensures we only warn
    or block on *truly consecutive* repeated reads — if the agent does
    anything else in between (write, patch, terminal, etc.) the counter
    resets and the next read is treated as fresh.
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data:
            task_data["last_key"] = None
            task_data["consecutive"] = 0
            # An intervening non-read tool call breaks any stub-loop in
            # progress, so clear per-key dedup hit counters too.
            if "dedup_hits" in task_data:
                task_data["dedup_hits"].clear()


def _invalidate_dedup_for_path(filepath: str, task_id: str) -> None:
    """Remove all dedup cache entries whose resolved path matches *filepath*.

    Called after write_file and patch so that a subsequent read_file on
    the same path always returns fresh content instead of a stale
    "File unchanged" stub.  The dedup cache keys are tuples of
    ``(resolved_path, offset, limit)``; we must evict **all** offset/limit
    combinations for the written path because any cached range could now
    be stale.

    Must be called with ``_read_tracker_lock`` **not** held — acquires it
    internally.
    """
    try:
        resolved = str(_resolve_path(filepath))
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is None:
            return
        dedup = task_data.get("dedup")
        if not dedup:
            return
        # Collect keys to remove (can't mutate dict during iteration).
        stale_keys = [k for k in dedup if k[0] == resolved]
        for k in stale_keys:
            del dedup[k]


def _update_read_timestamp(filepath: str, task_id: str) -> None:
    """Record the file's current modification time after a successful write.

    Called after write_file and patch so that consecutive edits by the
    same task don't trigger false staleness warnings — each write
    refreshes the stored timestamp to match the file's new state.

    Also invalidates the dedup cache for the written path so that
    subsequent reads return fresh content (fixes #13144).
    """
    # Invalidate dedup first (before acquiring lock for timestamp update).
    _invalidate_dedup_for_path(filepath, task_id)
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
        current_mtime = os.path.getmtime(resolved)
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is not None:
            task_data.setdefault("read_timestamps", {})[resolved] = current_mtime
            _cap_read_tracker_data(task_data)


def _check_file_staleness(filepath: str, task_id: str) -> str | None:
    """Check whether a file was modified since the agent last read it.

    Returns a warning string if the file is stale (mtime changed since
    the last read_file call for this task), or None if the file is fresh
    or was never read.  Does not block — the write still proceeds.
    """
    try:
        resolved = str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return None
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if not task_data:
            return None
        read_mtime = task_data.get("read_timestamps", {}).get(resolved)
    if read_mtime is None:
        return None  # File was never read — nothing to compare against
    try:
        current_mtime = os.path.getmtime(resolved)
    except OSError:
        return None
    if current_mtime != read_mtime:
        return (
            f"Warning: File '{filepath}' was modified since you last read it "
            f"(old: {read_mtime}, new: {current_mtime}). "
            "The content you read may be "
            "stale. Consider re-reading the file to verify before writing."
        )
    return None


def write_file_tool(path: str, content: str, task_id: str = "default") -> str:
    """Write content to a file."""
    sensitive_err = _check_sensitive_path(path, task_id)
    if sensitive_err:
        return tool_error(sensitive_err)
    if _is_internal_file_status_text(content):
        return tool_error(
            "Refusing to write internal read_file status text as file content. "
            "Re-read the file or reconstruct the intended file contents before writing."
        )
    try:
        # Resolve once for the registry lock + stale check.  Failures here
        # fall back to the legacy path — write proceeds, per-task staleness
        # check below still runs.
        try:
            _resolved = str(_resolve_path_for_task(path, task_id))
        except Exception:
            _resolved = None

        if _resolved is None:
            stale_warning = _check_file_staleness(path, task_id)
            file_ops = _get_file_ops(task_id)
            result = file_ops.write_file(path, content)
            result_dict = result.to_dict()
            if stale_warning:
                result_dict["_warning"] = stale_warning
            _update_read_timestamp(path, task_id)
            # Slim return: just bytes + sha256 instead of full result to save context tokens
            if result_dict.get("error"):
                return json.dumps({"error": result_dict["error"]}, ensure_ascii=False)
            return json.dumps({
                "written_bytes": result_dict.get("bytes_written", 0),
                "sha256": hashlib.sha256(content.encode('utf-8')).hexdigest(),
            }, ensure_ascii=False)

        # Serialize the read→modify→write region per-path so concurrent
        # subagents can't interleave on the same file.  Different paths
        # remain fully parallel.
        with file_state.lock_path(_resolved):
            # Cross-agent staleness wins over per-task warning when both
            # fire — its message names the sibling subagent.
            cross_warning = file_state.check_stale(task_id, _resolved)
            stale_warning = _check_file_staleness(path, task_id)
            file_ops = _get_file_ops(task_id)
            result = file_ops.write_file(path, content)
            result_dict = result.to_dict()
            effective_warning = cross_warning or stale_warning
            if effective_warning:
                result_dict["_warning"] = effective_warning
            # Refresh stamps after the successful write so consecutive
            # writes by this task don't trigger false staleness warnings.
            _update_read_timestamp(path, task_id)
            if not result_dict.get("error"):
                file_state.note_write(task_id, _resolved)
        # Slim return: just bytes + sha256 instead of full result to save context tokens
        if result_dict.get("error"):
            return json.dumps({"error": result_dict["error"]}, ensure_ascii=False)
        return json.dumps({
            "written_bytes": result_dict.get("bytes_written", 0),
            "sha256": hashlib.sha256(content.encode('utf-8')).hexdigest(),
        }, ensure_ascii=False)
    except Exception as e:
        if _is_expected_write_exception(e):
            logger.debug("write_file expected denial: %s: %s", type(e).__name__, e)
        else:
            logger.error("write_file error: %s: %s", type(e).__name__, e, exc_info=True)
        return tool_error(str(e))


def patch_tool(mode: str = "replace", path: str = None, old_string: str = None,
               new_string: str = None, replace_all: bool = False, patch: str = None,
               task_id: str = "default") -> str:
    """Patch a file using replace mode or V4A patch format."""
    # Check sensitive paths for both replace (explicit path) and V4A patch (extract paths)
    _paths_to_check = []
    if path:
        _paths_to_check.append(path)
    if mode == "patch" and patch:
        import re as _re
        for _m in _re.finditer(r'^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+)$', patch, _re.MULTILINE):
            _paths_to_check.append(_m.group(1).strip())
    for _p in _paths_to_check:
        sensitive_err = _check_sensitive_path(_p, task_id)
        if sensitive_err:
            return tool_error(sensitive_err)
    try:
        # Resolve paths for locking.  Ordered + deduplicated so concurrent
        # callers lock in the same order — prevents deadlock on overlapping
        # multi-file V4A patches.
        _resolved_paths: list[str] = []
        _seen: set[str] = set()
        for _p in _paths_to_check:
            try:
                _r = str(_resolve_path_for_task(_p, task_id))
            except Exception:
                _r = None
            if _r and _r not in _seen:
                _resolved_paths.append(_r)
                _seen.add(_r)
        _resolved_paths.sort()

        # Acquire per-path locks in sorted order via ExitStack.  On single
        # path this degenerates to one lock; on empty list (unresolvable)
        # it's a no-op and execution falls through unchanged.
        from contextlib import ExitStack
        with ExitStack() as _locks:
            for _r in _resolved_paths:
                _locks.enter_context(file_state.lock_path(_r))

            # Collect warnings — cross-agent registry first (names sibling),
            # then per-task tracker as a fallback.
            stale_warnings: list[str] = []
            _path_to_resolved: dict[str, str] = {}
            for _p in _paths_to_check:
                try:
                    _r = str(_resolve_path_for_task(_p, task_id))
                except Exception:
                    _r = None
                _path_to_resolved[_p] = _r
                _cross = file_state.check_stale(task_id, _r) if _r else None
                _sw = _cross or _check_file_staleness(_p, task_id)
                if _sw:
                    stale_warnings.append(_sw)

            file_ops = _get_file_ops(task_id)

            if mode == "replace":
                if not path:
                    return tool_error("path required")
                if old_string is None or new_string is None:
                    return tool_error("old_string and new_string required")
                result = file_ops.patch_replace(path, old_string, new_string, replace_all)
            elif mode == "patch":
                if not patch:
                    return tool_error("patch content required")
                result = file_ops.patch_v4a(patch)
            else:
                return tool_error(f"Unknown mode: {mode}")

            result_dict = result.to_dict()
            if stale_warnings:
                result_dict["_warning"] = stale_warnings[0] if len(stale_warnings) == 1 else " | ".join(stale_warnings)
            # Refresh stored timestamps for all successfully-patched paths so
            # consecutive edits by this task don't trigger false warnings.
            if not result_dict.get("error"):
                for _p in _paths_to_check:
                    _update_read_timestamp(_p, task_id)
                    _r = _path_to_resolved.get(_p)
                    if _r:
                        file_state.note_write(task_id, _r)
        # Hint when old_string not found — saves iterations where the agent
        # retries with stale content instead of re-reading the file.
        # Suppressed when patch_replace already attached a rich "Did you mean?"
        # snippet (which is strictly more useful than the generic hint).
        if result_dict.get("error") and "Could not find" in str(result_dict["error"]):
            if "Did you mean one of these sections?" not in str(result_dict["error"]):
                result_dict["_hint"] = (
                    "old_string not found. Use read_file to verify the current "
                    "content, or search_files to locate the text."
                )
        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))


def search_tool(pattern: str, target: str = "content", path: str = ".",
                file_glob: str = None, limit: int = 50, offset: int = 0,
                output_mode: str = "content", context: int = 0,
                task_id: str = "default") -> str:
    """Search for content or files."""
    try:
        offset, limit = normalize_search_pagination(offset, limit)

        # Track searches to detect *consecutive* repeated search loops.
        # Include pagination args so users can page through truncated
        # results without tripping the repeated-search guard.
        search_key = (
            "search",
            pattern,
            target,
            str(path),
            file_glob or "",
            limit,
            offset,
        )
        with _read_tracker_lock:
            task_data = _read_tracker.setdefault(task_id, {
                "last_key": None, "consecutive": 0, "read_history": set(),
            })
            if task_data["last_key"] == search_key:
                task_data["consecutive"] += 1
            else:
                task_data["last_key"] = search_key
                task_data["consecutive"] = 1
            count = task_data["consecutive"]

        if count >= 4:
            return json.dumps({
                "error": (
                    f"BLOCKED: You have run this exact search {count} times in a row. "
                    "The results have NOT changed. You already have this information. "
                    "STOP re-searching and proceed with your task."
                ),
                "pattern": pattern,
                "already_searched": count,
            }, ensure_ascii=False)

        file_ops = _get_file_ops(task_id)
        result = file_ops.search(
            pattern=pattern, path=path, target=target, file_glob=file_glob,
            limit=limit, offset=offset, output_mode=output_mode, context=context
        )
        if hasattr(result, 'matches'):
            for m in result.matches:
                if hasattr(m, 'content') and m.content:
                    m.content = redact_sensitive_text(m.content, code_file=True)
        result_dict = result.to_dict()

        if count >= 3:
            result_dict["_warning"] = (
                f"You have run this exact search {count} times consecutively. "
                "The results have not changed. Use the information you already have. "
                "If you are stuck in a loop, stop searching and proceed with writing or responding."
            )

        return json.dumps(result_dict, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e))


# ── Helpers ───────────────────────────────────────────────────────────

def tool_error(msg: str) -> str:
    """Return a JSON error response."""
    return json.dumps({"error": msg}, ensure_ascii=False)


_READ_DEDUP_STATUS_MESSAGE = (
    "Content not returned (file unchanged since last read). "
    "Proceed using the content from your earlier read_file result."
)

# Per-task read tracker — stores mtime snapshots, dedup keys, and
# consecutive-read counters.  Protected by _read_tracker_lock.
_read_tracker: dict[str, dict] = {}
_read_tracker_lock = threading.Lock()

# Max entries per-task before pruning (see _cap_read_tracker_data).
_MAX_READ_TRACKER_DEDUP_ENTRIES = 100
_MAX_READ_TRACKER_HISTORY_ENTRIES = 500


def _cap_read_tracker_data(task_data: dict) -> None:
    """Evict oldest entries when per-task tracker containers grow too large.

    Called after each read_file, write_file, and patch — the three operations
    that produce timer / read-history entries.  Without this cap, a
    long-running CLI session could accumulate millions of dedup keys over
    months of uptime (different offset+limit combinations on thousands of
    files).
    """
    dedup = task_data.get("dedup")
    if dedup is not None and len(dedup) > _MAX_READ_TRACKER_DEDUP_ENTRIES:
        # Evict oldest 25% by insertion order (Python 3.7+ dict).
        excess = len(dedup) - _MAX_READ_TRACKER_DEDUP_ENTRIES
        for k in list(dedup)[:excess]:
            del dedup[k]

    history = task_data.get("read_history")
    if history is not None and len(history) > _MAX_READ_TRACKER_HISTORY_ENTRIES:
        # Convert to list, slice oldest 25%, convert back.
        excess = len(history) - _MAX_READ_TRACKER_HISTORY_ENTRIES
        for item in list(history)[:excess]:
            history.discard(item)


def _resolve_path_for_task(filepath: str, task_id: str = "default") -> Path:
    """Resolve *filepath* relative to the task's working directory.

    If *filepath* is absolute, use it as-is (after resolving symlinks).
    If relative, resolve against the task's cwd so that ``cd`` commands
    issued via the terminal tool are reflected here.

    Accepts ``None`` as a no-op — returns ``None`` unchanged.
    """
    if filepath is None:
        return None  # type: ignore[return-value]

    # Always try to resolve via task cwd — even absolute paths benefit
    # from symlink resolution (e.g. /tmp → /private/tmp on macOS).
    cwd = _resolve_cwd_for_task(task_id)
    try:
        resolved = (cwd / filepath).resolve()
    except (OSError, ValueError):
        # If path resolution fails (e.g. invalid characters), fall back
        # to the filepath as a raw Path for error reporting.
        resolved = Path(filepath)
    return resolved


# ── Environment creation ──────────────────────────────────────────────

def _create_environment(
    env_type: str,
    image: str = "",
    cwd: str = "",
    timeout: int = 180,
    ssh_config: dict = None,
    container_config: dict = None,
    local_config: dict = None,
    task_id: str = "default",
    host_cwd: str = None,
) -> object:
    """Create a terminal environment of the specified type."""
    from tools.terminal_tool import _create_environment as _terminal_create
    return _terminal_create(
        env_type=env_type,
        image=image,
        cwd=cwd,
        timeout=timeout,
        ssh_config=ssh_config,
        container_config=container_config,
        local_config=local_config,
        task_id=task_id,
        host_cwd=host_cwd,
    )


# ── Path sensitivity checks ───────────────────────────────────────────

_SENSITIVE_PATH_PREFIXES: list[str] = []
_SENSITIVE_PATHS_LOADED = False


def _load_sensitive_paths():
    """Load sensitive path prefixes from config."""
    global _SENSITIVE_PATH_PREFIXES, _SENSITIVE_PATHS_LOADED
    if _SENSITIVE_PATHS_LOADED:
        return
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        paths = cfg.get("sensitive_path_prefixes", [])
        if isinstance(paths, list):
            _SENSITIVE_PATH_PREFIXES = [os.path.normpath(p) for p in paths]
    except Exception:
        pass
    _SENSITIVE_PATHS_LOADED = True


def _check_sensitive_path(path: str, task_id: str) -> str | None:
    """Check if path is sensitive and block write/patch operations.

    Returns an error message if the path is sensitive, or None if safe.
    """
    if not path:
        return None
    _load_sensitive_paths()
    if not _SENSITIVE_PATH_PREFIXES:
        return None
    try:
        resolved = str(_resolve_path_for_task(path, task_id))
    except Exception:
        resolved = path
    normalized = os.path.normpath(resolved)
    for prefix in _SENSITIVE_PATH_PREFIXES:
        if normalized.startswith(prefix):
            return (
                f"Write denied: '{path}' matches sensitive path prefix '{prefix}'. "
                "This path is protected from accidental modification."
            )
    return None


def _is_blocked_device(path: str) -> bool:
    """Check if path is a device file that would hang the process.

    Pure path check — no I/O.  Compare against the known blocklist.
    """
    if not path:
        return False
    # Normalize to prevent trivial bypasses like //dev/zero or /DEV/zero.
    normalized = os.path.normpath(path).lower()
    return normalized in _BLOCKED_DEVICE_PATHS


# ── Internal status text guard ─────────────────────────────────────────

def _is_internal_file_status_text(content: str) -> bool:
    """Detect if the content looks like internal status text (dedup stub).

    Prevents the model from accidentally writing a read_file "unchanged"
    stub as actual file content, which would silently corrupt the file.
    """
    if not content or not isinstance(content, str):
        return False
    stripped = content.strip()
    if not stripped:
        return False
    # The dedup stub is a JSON payload with specific keys.
    if stripped.startswith('{"') and '"status"' in stripped and '"dedup"' in stripped:
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and obj.get("status") == "unchanged":
                return True
        except (json.JSONDecodeError, ValueError):
            pass
    return False


# ── Schemas ───────────────────────────────────────────────────────────

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a text file with line numbers and pagination. Use this instead of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. Suggests similar filenames if not found. Use offset and limit for large files. Reads exceeding ~100K characters are rejected; use offset and limit to read specific sections of large files. NOTE: Cannot read images or binary files — use vision_analyze for images.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (absolute, relative, or ~/path)"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed, default: 1)", "default": 1, "minimum": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to return (default: 500, max: 2000)", "default": 500, "maximum": 2000}
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write content to a file, completely replacing existing content. Use this instead of echo/cat heredoc in terminal. Creates parent directories automatically. OVERWRITES the entire file — use 'patch' for targeted edits. Auto-runs syntax checks on .py/.json/.yaml/.toml and other linted languages; only NEW errors introduced by this write are surfaced (pre-existing errors are filtered out).",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"},
            "content": {"type": "string", "description": "Complete content to write to the file"}
        },
        "required": ["path", "content"]
    }
}

PATCH_SCHEMA = {
    "name": "patch",
    "description": (
        "Targeted find-and-replace edits in files. Use this instead of sed/awk in terminal. "
        "Uses fuzzy matching (9 strategies) so minor whitespace/indentation differences won't break it. "
        "Returns a unified diff. Auto-runs syntax checks after editing.\n\n"
        "REPLACE MODE (mode='replace', default): find a unique string and replace it. "
        "REQUIRED PARAMETERS: mode, path, old_string, new_string.\n"
        "PATCH MODE (mode='patch'): apply V4A multi-file patches for bulk changes. "
        "REQUIRED PARAMETERS: mode, patch."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["replace", "patch"],
                "description": "Edit mode. 'replace' (default): requires path + old_string + new_string. 'patch': requires patch content only.",
                "default": "replace",
            },
            "path": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. File path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. Exact text to find and replace. Must be unique in the file unless replace_all=true. Include surrounding context lines to ensure uniqueness.",
            },
            "new_string": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. Replacement text. Pass empty string '' to delete the matched text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                "default": False,
            },
            "patch": {
                "type": "string",
                "description": "REQUIRED when mode='patch'. V4A format patch content. Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch",
            },
        },
        "required": ["mode"],
    },
}

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": "Search file contents or find files by name. Use this instead of grep/rg/find/ls in terminal. Ripgrep-backed, faster than shell equivalents.\n\nContent search (target='content'): Regex search inside files. Output modes: full matches with line numbers, file paths only, or match counts.\n\nFile search (target='files'): Find files by glob pattern (e.g., '*.py', '*config*'). Also use this instead of ls — results sorted by modification time.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern for content search, or glob pattern (e.g., '*.py') for file search"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "'content' searches inside file contents, 'files' searches for files by name", "default": "content"},
            "path": {"type": "string", "description": "Directory or file to search in (default: current working directory)", "default": "."},
            "file_glob": {"type": "string", "description": "Filter files by pattern in grep mode (e.g., '*.py' to only search Python files)"},
            "limit": {"type": "integer", "description": "Maximum number of results to return (default: 50)", "default": 50},
            "offset": {"type": "integer", "description": "Skip first N results for pagination (default: 0)", "default": 0},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "description": "Output format for grep mode: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file", "default": "content"},
            "context": {"type": "integer", "description": "Number of context lines before and after each match (grep mode only)", "default": 0}
        },
        "required": ["pattern"]
    }
}


def _handle_read_file(args, **kw):
    tid = kw.get("task_id") or "default"
    return read_file_tool(path=args.get("path", ""), offset=args.get("offset", 1), limit=args.get("limit", 500), task_id=tid)


def _handle_write_file(args, **kw):
    tid = kw.get("task_id") or "default"
    if not args.get("path") or not isinstance(args.get("path"), str):
        return tool_error(
            "write_file: missing required field 'path'. Re-emit the tool call with "
            "both 'path' and 'content' set."
        )
    if "content" not in args:
        return tool_error(
            "write_file: missing required field 'content'. The tool call included a "
            "path but no content argument — this is almost always a dropped-arg bug "
            "under context pressure. Re-emit the tool call with the full content "
            "payload, or use execute_code with hermes_tools.write_file() for very "
            "large files."
        )
    if not isinstance(args["content"], str):
        return tool_error(
            f"write_file: 'content' must be a string, got "
            f"{type(args['content']).__name__}."
        )
    return write_file_tool(path=args["path"], content=args["content"], task_id=tid)


def _handle_patch(args, **kw):
    tid = kw.get("task_id") or "default"
    return patch_tool(
        mode=args.get("mode", "replace"), path=args.get("path"),
        old_string=args.get("old_string"), new_string=args.get("new_string"),
        replace_all=args.get("replace_all", False), patch=args.get("patch"), task_id=tid)


def _handle_search_files(args, **kw):
    tid = kw.get("task_id") or "default"
    target_map = {"grep": "content", "find": "files"}
    raw_target = args.get("target", "content")
    target = target_map.get(raw_target, raw_target)
    return search_tool(
        pattern=args.get("pattern", ""), target=target, path=args.get("path", "."),
        file_glob=args.get("file_glob"), limit=args.get("limit", 50), offset=args.get("offset", 0),
        output_mode=args.get("output_mode", "content"), context=args.get("context", 0), task_id=tid)


def _check_file_reqs(**kw):
    """Check that file operations are available."""
    return True
