"""Strata Engine for Flint 40B (Recursive MoE 256 experts + 2 shared experts).

Implements the Strata Engine Line Protocol over stdin/stdout:
  - Startup: Outputs 'READY <max_context> stop'
  - Prompts: Reads 'GEN <max_new> [temp=F] [top_p=F] <token_ids...>'
  - Tokens: Outputs 'T <token_id>'
  - Completion: Outputs 'DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> stop'
  - Cancellation: Reads 'STOP' and aborts active generation

Bridges to the native CUDA sm_120 Flint MoE backend with Tiered CPU Expert offload.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

CANDIDATE_BINS = [
    REPO_ROOT / "bin" / "llama-server.exe",
    Path(r"E:\AI-models\llm-server-build\src-bee-v046\build-flint\bin\llama-server.exe"),
    REPO_ROOT.parent / "llm-server-build" / "src-bee-v046" / "build-flint" / "bin" / "llama-server.exe",
]

CANDIDATE_MODELS = [
    REPO_ROOT / "models" / "flint" / "flint-UD-Q3_K_XL.gguf",
    REPO_ROOT / "models" / "flint-UD-Q3_K_XL.gguf",
    Path(r"E:\AI-models\models\flint\flint-UD-Q3_K_XL.gguf"),
    REPO_ROOT.parent / "models" / "flint" / "flint-UD-Q3_K_XL.gguf",
]

DEFAULT_BIN = next((b for b in CANDIDATE_BINS if b.exists()), CANDIDATE_BINS[0])
DEFAULT_MODEL = next((m for m in CANDIDATE_MODELS if m.exists()), CANDIDATE_MODELS[0])


def log_err(msg: str):
    sys.stderr.write(f"[strata-flint] {msg}\n")
    sys.stderr.flush()


def wait_for_server(port: int, timeout_s: float = 120.0, proc: subprocess.Popen | None = None) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if proc and proc.poll() is not None:
            log_err(f"Underlying server died with exit code {proc.returncode}")
            return False
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") in ("ok", "no slot available"):
                        return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    # Ensure stdout/stdin are line-buffered and binary/utf-8 safe
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description="Strata Flint-40B Engine")
    ap.add_argument("--native", help="Path to Flint GGUF model")
    ap.add_argument("--pack", help="Path to Flint pack directory")
    ap.add_argument("--max-context", type=int, default=32768, help="Context size")
    ap.add_argument("--gpu", type=str, default="0", help="GPU id")
    ap.add_argument("--port", type=int, default=8199, help="Internal server port")
    ap.add_argument("--threads", type=int, default=12, help="CPU threads")
    ap.add_argument("--n-cpu-moe", type=int, default=None, help="MoE layers offloaded to CPU RAM")
    ap.add_argument("--cache-type-k", type=str, default="q8_0", help="KV cache K precision")
    ap.add_argument("--cache-type-v", type=str, default="q8_0", help="KV cache V precision")
    ap.add_argument("--ctx-checkpoints", type=int, default=None, help="Context checkpoints")
    ap.add_argument("--cache-ram", type=int, default=None, help="Host RAM cache in MiB")
    ap.add_argument("--batch-size", type=int, default=2048, help="Logical batch size")
    ap.add_argument("--ubatch-size", type=int, default=1024, help="Physical micro-batch size")
    ap.add_argument("--kvarn-window-chunk", type=int, default=None, help="KVarN window chunk")
    ap.add_argument("--server-bin", type=str, default=None, help="llama-server binary path")
    ap.add_argument("--serve", action="store_true", help="Run in resident serve mode (Strata protocol)")
    ap.add_argument("--tokens", type=str, default=None, help="One-shot tokens list (test mode)")

    args, unknown = ap.parse_known_args()

    model_path = args.native
    if not model_path and args.pack:
        p = Path(args.pack)
        cand = p / "flint-UD-Q3_K_XL.gguf"
        if cand.exists():
            model_path = str(cand)
    if not model_path:
        model_path = str(DEFAULT_MODEL)

    server_bin = Path(args.server_bin) if args.server_bin else DEFAULT_BIN
    if not server_bin.exists():
        log_err(f"Binary not found: {server_bin}")
        return 1

    internal_port = args.port
    max_context = args.max_context

    # Tiered Context & Memory Architecture (Strata-style):
    # In Strata, increasing context to 200K/256K does NOT eject model layers to CPU!
    # Instead, the active KV window resides in VRAM, while the bulk context / checkpoints
    # are stored in Host RAM (--ctx-checkpoints + --cache-ram).
    # Thus ncmoe stays at 11 (36 layers in VRAM), preserving ~80-88 t/s decode speed!
    if args.n_cpu_moe is not None:
        ncmoe = args.n_cpu_moe
    else:
        ncmoe = 11  # Always keep 36 layers on GPU; context is tiered into RAM

    # Default KV cache types for long contexts:
    # At >= 65K tokens, we use KVarN / Host RAM checkpointing so KV stays compact in VRAM (~1 GB)
    ctk = args.cache_type_k
    ctv = args.cache_type_v


    cmd = [
        str(server_bin),
        "-m", str(model_path),
        "--ctx-size", str(max_context),
        "--n-gpu-layers", "999",
        "--n-cpu-moe", str(ncmoe),
        "--flash-attn", "on",
        "--fit", "off",
        "--cache-type-k", str(ctk),
        "--cache-type-v", str(ctv),
        "--threads", str(args.threads),
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--parallel", "1",
        "--host", "127.0.0.1",
        "--port", str(internal_port),
        "--alias", "flint",
        "--log-disable"
    ]

    if max_context >= 65536 or args.ctx_checkpoints is not None:
        checkpoints = args.ctx_checkpoints if args.ctx_checkpoints is not None else 16
        cache_ram = args.cache_ram if args.cache_ram is not None else 8192
        cmd.extend(["--ctx-checkpoints", str(checkpoints), "--cache-ram", str(cache_ram)])
    if args.kvarn_window_chunk is not None:
        cmd.extend(["--kvarn-window-chunk", str(args.kvarn_window_chunk)])

    log_err(f"Launching Flint backend: {' '.join(cmd[:6])} ... on port {internal_port}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )

    def cleanup():
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    atexit.register(cleanup)

    if not wait_for_server(internal_port, timeout_s=90.0, proc=proc):
        log_err("Flint backend failed to initialize")
        cleanup()
        return 1

    log_err(f"Flint backend ready on port {internal_port}")

    # Emit Strata READY handshake
    print(f"READY {max_context} stop", flush=True)

    # Main resident loop reading Strata commands from stdin
    active_req = None

    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            break
        if not line:
            break  # EOF / parent exited

        line = line.strip()
        if not line:
            continue

        if line == "STOP":
            # Cancellation requested
            try:
                stop_url = f"http://127.0.0.1:{internal_port}/slots/0?action=stop"
                req = urllib.request.Request(stop_url, method="POST")
                urllib.request.urlopen(req, timeout=1.0)
            except Exception:
                pass
            print("DONE 0 0 0.0 0.0 stop", flush=True)
            continue

        if line.startswith("GEN "):
            parts = line[4:].split()
            max_new = 128
            temp = 0.7
            top_p = 0.95
            token_ids_str = ""

            for p in parts:
                if p.startswith("temperature="):
                    try:
                        temp = float(p.split("=")[1])
                    except ValueError:
                        pass
                elif p.startswith("top_p="):
                    try:
                        top_p = float(p.split("=")[1])
                    except ValueError:
                        pass
                elif p.isdigit() and len(p) < 6 and not token_ids_str:
                    max_new = int(p)
                else:
                    token_ids_str = p

            # Parse prompt token IDs
            try:
                prompt_ids = [int(x) for x in re.findall(r"\d+", token_ids_str)]
            except Exception:
                prompt_ids = []

            if not prompt_ids:
                print("ERR empty prompt token list", flush=True)
                continue

            payload = {
                "prompt": prompt_ids,
                "n_predict": max_new,
                "temperature": temp,
                "top_p": top_p,
                "stream": True,
                "cache_prompt": True
            }

            req_url = f"http://127.0.0.1:{internal_port}/completion"
            data = json.dumps(payload).encode("utf-8")
            http_req = urllib.request.Request(
                req_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            t_start = time.monotonic()
            prompt_ms = 0.0
            decode_ms = 0.0
            gen_count = 0
            prompt_count = len(prompt_ids)
            finish_reason = "stop"

            try:
                with urllib.request.urlopen(http_req, timeout=300.0) as resp:
                    for sse_line in resp:
                        sse_line = sse_line.decode("utf-8", errors="replace").strip()
                        if not sse_line.startswith("data:"):
                            continue
                        chunk_str = sse_line[5:].strip()
                        if not chunk_str:
                            continue
                        try:
                            chunk = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            continue

                        # Check for prompt progress
                        if "prompt_progress" in chunk:
                            prog = chunk["prompt_progress"]
                            cur_p = prog.get("progress", 0)
                            tot_p = prog.get("total", 0)
                            el_ms = (time.monotonic() - t_start) * 1000.0
                            print(f"PP {cur_p} {tot_p} {el_ms:.1f}", flush=True)

                        # Check for streamed tokens
                        tokens = chunk.get("tokens", [])
                        for t in tokens:
                            print(f"T {t}", flush=True)
                            gen_count += 1

                        if chunk.get("stop", False):
                            finish_reason = chunk.get("stopping_word") or "stop"
                            timings = chunk.get("timings", {})
                            prompt_ms = timings.get("prompt_ms", 0.0)
                            decode_ms = timings.get("predicted_ms", 0.0)
                            if timings.get("prompt_n"):
                                prompt_count = timings["prompt_n"]
                            break

                total_time = (time.monotonic() - t_start) * 1000.0
                if decode_ms <= 0:
                    decode_ms = max(1.0, total_time - prompt_ms)

                print(f"DONE {gen_count} {prompt_count} {prompt_ms:.1f} {decode_ms:.1f} {finish_reason}", flush=True)

            except Exception as e:
                log_err(f"Error during completion generation: {e}")
                print(f"ERR generation failure: {e}", flush=True)

        elif line.startswith("VRAM "):
            # Strata VRAM query
            print("VRAM ok", flush=True)

    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
