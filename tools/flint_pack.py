"""tools/flint_pack.py - Strata pack builder for Flint 40B (256x2.1B MoE).

Adapted from tools/iq_pack.py for the Flint architecture:
  - Total layers: 49
  - Leading dense blocks: 2 (layers 0 and 1: dense FFN)
  - MoE layers: 2..48 (47 MoE layers, 256 routed experts each)
  - Shared experts per MoE layer: 2
  - Routed expert types in Flint-UD-Q3_K_XL:
      gate / up: IQ2_S  (335,872 bytes per expert)
      down:      IQ4_XS (557,056 bytes per expert)
      blob:      1,228,800 bytes per expert
  - Dense FFN (layers 0, 1) and shared experts: served natively from GGUF
  - Router ffn_gate_inp.weight: BF16 in dense.bin
  - Norms and attention sinks: F32 in dense.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

FLOAT = {"BF16", "F32", "F16"}

FORM = {
    "ffn_gate_inp.weight": "BF16",
    "attn_q_norm.weight": "F32",
    "attn_k_norm.weight": "F32",
    "attn_norm.weight": "F32",
    "attn_sinks.weight": "F32",
    "ffn_norm.weight": "F32",
    "output_norm.weight": "F32",
}

KIND = {"BF16": "4", "F16": "5", "F32": "2"}
ALIGN = 64
ROLES = ("gate", "up", "down")


def form_of(name: str):
    return FORM.get(re.sub(r"^blk\.\d+\.", "", name))


def tensor_bytes(mm: np.ndarray, g: G.GGUFFile, t: G.TensorInfo) -> np.ndarray:
    n = t.expected_bytes()
    return mm[g.data_start + t.offset: g.data_start + t.offset + n]


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith("_exps.weight")


class Model:
    """All shards of the model."""

    def __init__(self, first: pathlib.Path):
        paths = [first]
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
        if m:
            total = int(m.group(2))
            paths = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total))
                     for i in range(1, total + 1)]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing model shards: " + ", ".join(missing))
        self.paths = paths
        self.files = [G.GGUFFile(p) for p in paths]
        self.sizes = [p.stat().st_size for p in paths]
        self.where = {}
        for p, g in zip(paths, self.files):
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            for t in g.tensors:
                size = t.expected_bytes()
                if size is None or g.data_start + t.offset + size > mm.size:
                    raise ValueError(f"{p.name}: unsupported or truncated tensor {t.name}")
                if t.name in self.where:
                    raise ValueError(f"{p.name}: duplicate tensor {t.name}")
                self.where[t.name] = (g, t, mm, p)

    def bytes(self, name: str) -> np.ndarray:
        g, t, mm, _ = self.where[name]
        return tensor_bytes(mm, g, t)


def expert_layout_flint(model: Model, src: pathlib.Path):
    """Layout for Flint's routed experts (layers 2..48, 256 experts per layer)."""
    T = {n: w[1] for n, w in model.where.items()}
    exps = [n for n in T if is_expert(n)]
    if not exps:
        return "the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)"

    moe_layers = sorted(set(int(n.split(".")[1]) for n in exps))
    first_moe = moe_layers[0]
    n_expert = int(T[f"blk.{first_moe}.ffn_gate_inp.weight"].shape[1])
    print(f"Flint MoE: {len(moe_layers)} MoE layers (layers {min(moe_layers)}..{max(moe_layers)}), {n_expert} experts per layer")

    layout, lines, offset = [], [], 0
    for l in moe_layers:
        names = [f"blk.{l}.ffn_{r}_exps.weight" for r in ROLES]
        if any(n not in T for n in names):
            return f"layer {l}: missing {', '.join(n for n in names if n not in T)}"
        ts = [T[n] for n in names]
        if any(t.expected_bytes() is None or len(t.shape) != 3 or int(t.shape[2]) != n_expert for t in ts):
            return f"layer {l}: an expert tensor is not [*, *, {n_expert}]"
        per = [t.expected_bytes() // n_expert for t in ts]
        if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
            return f"layer {l}: gate and up differ in type"
        blob = per[0] + per[1] + per[2]
        layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, ts))

        ws = [model.where[n] for n in names]
        line = "%d %d %d %d %d %d %d %d" % (l, ts[0].type_id, ts[2].type_id, offset, blob,
                                            *[w[0].data_start + w[1].offset for w in ws])
        lines.append(line)
        offset += blob * n_expert

    head = ("# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
            "(n_expert %d, total %d; absolute offsets in %s)\n" % (n_expert, offset, src.name))
    return layout, head + "".join(line + "\n" for line in lines), n_expert, offset


def write_index(out: pathlib.Path, rows: list[list[str]], src: pathlib.Path, served: int, n_extra: int, publish: bool = True):
    at = 0
    for r in rows:
        r[5] = str(at)
        at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with open(out / "index.txt.tmp", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata pack index v3 -- generated by tools/flint_pack.py from %s\n" % src.name)
        fo.write("# align %d pool %d tensors %d\n" % (ALIGN, at, len(rows)))
        for r in rows:
            fo.write(" ".join(r) + "\n")
    if publish:
        (out / "index.txt.tmp").replace(out / "index.txt")
    print("index.txt: %d tensors, %d served natively, %d in extra.bin, arena %.2f GiB"
          % (len(rows), served, n_extra, at / 2**30))


def index_standalone_flint(src: pathlib.Path, out: pathlib.Path, model: Model) -> int:
    """Build dense.bin and index.txt for Flint."""
    rows = []
    converted, records = [], []
    at = 0
    served = 0

    with open(out / "dense.bin.tmp", "wb") as fo:
        for name, (g, t, mm, p) in model.where.items():
            if is_expert(name):
                continue
            ne0 = int(t.shape[0])
            ne1 = int(t.shape[1]) if len(t.shape) > 1 else 0

            target_form = form_of(name)
            if target_form is None:
                # Native served tensor (GGUF direct mapping)
                served += 1
                rows.append([t.name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
            else:
                # Put in dense.bin (F32 or BF16)
                raw = tensor_bytes(mm, g, t)
                if target_form == "BF16":
                    kind = "4"
                    if t.type_name == "BF16":
                        data = raw.tobytes()
                    else:
                        raise ValueError(f"{name}: expected BF16, got {t.type_name}")
                elif target_form == "F32":
                    kind = "2"
                    if t.type_name == "F32":
                        data = raw.tobytes()
                    else:
                        raise ValueError(f"{name}: expected F32, got {t.type_name}")
                else:
                    raise ValueError(f"unsupported target form {target_form} for {name}")

                dst_bytes = len(data)
                rows.append([t.name, "0", kind, str(at), str(len(data)), "0", str(dst_bytes), str(ne0), str(ne1),
                             "0", "0", "1"] + ["0"] * 7)
                fo.write(data)
                pad = (-len(data)) % ALIGN
                fo.write(b"\0" * pad)
                at += len(data) + pad

    write_index(out, rows, src, served, 0, publish=False)
    (out / "dense.bin.tmp").replace(out / "dense.bin")
    (out / "index.txt.tmp").replace(out / "index.txt")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack Flint 40B for Strata")
    ap.add_argument("--gguf", required=True, help="Path to flint-UD-Q3_K_XL.gguf")
    ap.add_argument("--out", required=True, help="Output directory for pack")
    ap.add_argument("--experts-bin", action="store_true", help="Write monolithic experts.bin")
    ap.add_argument("--skip-experts", action="store_true", help="Skip writing experts.bin")
    args = ap.parse_args()

    src = pathlib.Path(args.gguf).resolve()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading Flint model from {src}...")
    model = Model(src)
    g = model.files[0]

    # 1. Expert layout
    got = expert_layout_flint(model, src)
    if isinstance(got, str):
        print("Error:", got)
        return 1
    layout, native_txt, n_expert, total_expert_bytes = got

    # Write native_experts.txt
    (out / "native_experts.txt").write_text(native_txt, encoding="utf-8", newline="\n")
    print(f"native_experts.txt written ({len(layout)} layers, {n_expert} experts, {total_expert_bytes / 2**30:.2f} GiB)")

    # 2. Dense and Index
    print("Writing dense.bin and index.txt...")
    rc = index_standalone_flint(src, out, model)
    if rc:
        return rc

    # 3. Tokenizer
    tok_dir = out / "tokenizer"
    if not (tok_dir / "vocab.json").exists() or not (tok_dir / "chat_template.jinja").exists():
        print("Extracting tokenizer...")
        subprocess.run([sys.executable, str(HERE / "strata_tokenizer.py"), "--gguf", str(src), "--out", str(out)],
                       check=True)

    # 4. Optional: experts.bin
    if args.experts_bin and not args.skip_experts:
        path = out / "experts.bin"
        print(f"Writing experts.bin to {path} ({total_expert_bytes / 2**30:.2f} GiB)...")
        with open(path, "wb") as fo:
            for l, gt, dt, off, blob, ts in layout:
                parts = [model.bytes(t.name).reshape(n_expert, -1) for t in ts]
                chunk = np.concatenate(parts, axis=1)
                assert chunk.shape == (n_expert, blob)
                fo.write(chunk.tobytes())
                if l % 8 == 0 or l == layout[-1][0]:
                    print(f"  layer {l:2d}: blob {blob} B, total written {fo.tell() / 2**30:.2f} GiB")
        print("experts.bin written successfully.")

    print(f"\n[SUCCESS] Flint pack ready at: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
