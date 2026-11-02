# -*- coding: utf-8 -*-
"""Needle In A Haystack (NIAH) Benchmark for Flint 40B MoE (Strata Tiered 256K).
Tests retrieval of a secret needle placed deep in long context.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

FORK_DIR = Path(r"E:\AI-models\strata-fork")
CONFIG_PATH = FORK_DIR / "strata-flint-256k.json"
PORT = 8081
API_URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HEALTH_URL = f"http://127.0.0.1:{PORT}/health"

NEEDLE = "ВАЖНОЕ ПРИМЕЧАНИЕ ДЛЯ ОПЕРАТОРА: Секретный мастер-ключ доступа к квантовому шлюзу — КВАРК-7749-ДЕЛЬТА."
SECRET_EXPECTED = "КВАРК-7749-ДЕЛЬТА"
QUESTION = "Ответь строго по тексту выше: Какой секретный мастер-ключ доступа к квантовому шлюзу указан в примечании для оператора? Укажи точный ключ."


def get_vram_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        used, total = map(int, out.split(","))
        return used, total
    except Exception:
        return 0, 0


def wait_for_server(timeout_s=120):
    print(f"[*] Ожидание готовности Strata Server на http://127.0.0.1:{PORT}...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            req = urllib.request.Request(HEALTH_URL, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") in ("ok", "no slot available"):
                        print(f"[+] Сервер готов за {time.monotonic() - t0:.1f} сек!")
                        return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def build_haystack(target_chars=250000, needle_depth=0.5):
    """Builds a haystack document with the secret needle at needle_depth (0.0 to 1.0)."""
    src_file = Path(r"E:\AI-models\ctx-200k-prompt.txt")
    if src_file.exists():
        raw_text = src_file.read_text(encoding="utf-8", errors="ignore")
    else:
        # Fallback text generator
        block = (
            "Архитектура современных параллельных вычислений требует строгого разделения "
            "потоков инструкций и данных. При оптимизации работы графического процессора "
            "важно учитывать выравнивание памяти по границам варпов и коалесцированный доступ.\n"
        )
        raw_text = block * 1000

    # Repeat or slice to get desired character length
    while len(raw_text) < target_chars:
        raw_text += raw_text

    raw_text = raw_text[:target_chars]

    # Split into two halves according to needle_depth
    split_pos = int(len(raw_text) * needle_depth)
    part1 = raw_text[:split_pos]
    part2 = raw_text[split_pos:]

    full_text = part1 + f"\n\n[--- СИСТЕМНАЯ ВСТАВКА ---]\n{NEEDLE}\n[--- КОНЕЦ ВСТАВКИ ---]\n\n" + part2
    return full_text


def run_benchmark(target_chars=250000, needle_depth=0.5):
    print("=" * 65)
    print("  Needle In A Haystack Benchmark (Flint 40B @ Strata 256K Tiered)")
    print("=" * 65)

    used, total = get_vram_mib()
    print(f"[*] Начальное состояние VRAM: {used} / {total} MiB ({used/1024:.2f} GB)")

    print(f"[*] Генерация документа на ~{target_chars} символов...")
    haystack = build_haystack(target_chars=target_chars, needle_depth=needle_depth)
    print(f"[+] Документ сформирован: {len(haystack):,} символов.")
    print(f"[*] Иголка размещена на глубине: {needle_depth * 100:.1f}% контекста.")
    print(f"[*] Ожидаемый ответ: '{SECRET_EXPECTED}'")

    prompt = (
        f"Ниже приведён длинный технический документ. Внимательно изучи его и ответь на вопрос в конце.\n\n"
        f"=== НАЧАЛО ДОКУМЕНТА ===\n"
        f"{haystack}\n"
        f"=== КОНЕЦ ДОКУМЕНТА ===\n\n"
        f"Вопрос: {QUESTION}\n"
        f"Инструкция: Подумай кратко и сразу дай точный ответ в формате: 'Секретный ключ: <КЛЮЧ>'.\n"
        f"Ответ:"
    )

    payload = {
        "model": "flint-40b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "reasoning_budget_tokens": 256
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    print("\n[*] Отправка запроса в Strata Server API...")
    t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            t_end = time.perf_counter()
            resp_data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[-] Ошибка выполнения запроса: {e}")
        return False

    elapsed = t_end - t_start
    choice = resp_data["choices"][0]["message"]
    content = (choice.get("content") or "").strip()
    reasoning = (choice.get("reasoning_content") or "").strip()
    usage = resp_data.get("usage", {})
    timings = resp_data.get("timings", {})

    prompt_tokens = usage.get("prompt_tokens", 0)
    comp_tokens = usage.get("completion_tokens", 0)

    used_after, _ = get_vram_mib()

    print("\n" + "=" * 65)
    print("  РЕЗУЛЬТАТЫ ТЕСТА")
    print("=" * 65)
    print(f"Prompt Tokens:        {prompt_tokens:,} токенов")
    print(f"Completion Tokens:    {comp_tokens} токенов")
    print(f"Общее время:          {elapsed:.2f} сек")
    if timings:
        pp_tps = timings.get("prompt_per_second", 0)
        dec_tps = timings.get("predicted_per_second", 0)
        print(f"Скорость префилла:    {pp_tps:.1f} т/с")
        print(f"Скорость декода:      {dec_tps:.1f} т/с")
    print(f"Пиковая VRAM:         {used_after} MiB ({used_after/1024:.2f} ГБ)")

    print("\n--- Ответ модели ---")
    if reasoning:
        print(f"[Мысли модели ({len(reasoning)} симв.)]:\n{reasoning}\n")
    print(f"[Текст ответа]: {content if content else '[ответ в блоке reasoning]'}")

    full_output = (content + " " + reasoning).lower()
    is_success = SECRET_EXPECTED.lower() in full_output or "кварк-7749-дельта" in full_output
    print("\n" + "=" * 65)
    if is_success:
        print(f"  ВЕРДИКТ: [PASS] (100% УСПЕХ) - Иголка найдена точно: '{SECRET_EXPECTED}'!")
    else:
        print(f"  ВЕРДИКТ: [FAIL] - Иголка не обнаружена или искажена.")
    print("=" * 65)
    return is_success


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 250000
    depth = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    run_benchmark(target_chars=target, needle_depth=depth)
