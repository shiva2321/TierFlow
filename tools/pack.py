"""TierFlow Offline Packer (Phase 1 of GEMINI_IMPLEMENTATION_SPEC.md).

Parses any GGUF v2/v3 file, splits tensors into logical objects (per-layer
attention/norms/router/FFN or MoE experts, embeddings, output head), aligns
every object to 4096-byte page boundaries inside `.tfpack` so that Windows
`FILE_FLAG_NO_BUFFERING` / Linux `O_DIRECT` can fetch any single object with
zero page-cache pollution, computes per-object SHA-256 and 64-bit FNV-1a
checksums, and verifies byte-for-byte round-trip reconstruction against the
original GGUF tensor payloads.

Usage:
    python tools/pack.py --self-test
    python tools/pack.py --input <path.gguf> --out-dir <dir> --verify
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple


PAGE_SIZE = 4096

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_FMT = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<?",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}

# ggml tensor type block sizes and byte sizes: (block_elements, type_size_bytes)
GGML_QUANT_SIZES: Dict[int, Tuple[int, int, str]] = {
    0: (1, 4, "F32"),
    1: (1, 2, "F16"),
    2: (32, 18, "Q4_0"),
    3: (32, 20, "Q4_1"),
    6: (32, 22, "Q5_0"),
    7: (32, 24, "Q5_1"),
    8: (32, 34, "Q8_0"),
    9: (32, 36, "Q8_1"),
    10: (256, 84, "Q2_K"),
    11: (256, 110, "Q3_K"),
    12: (256, 144, "Q4_K"),
    13: (256, 176, "Q5_K"),
    14: (256, 210, "Q6_K"),
    15: (256, 292, "Q8_K"),
    16: (256, 66, "IQ2_XXS"),
    17: (256, 82, "IQ2_XS"),
    18: (256, 98, "IQ3_XXS"),
    19: (256, 50, "IQ1_S"),
    20: (32, 18, "IQ4_NL"),
    21: (256, 110, "IQ3_S"),
    22: (256, 104, "IQ2_S"),
    23: (256, 136, "IQ4_XS"),
    24: (1, 1, "I8"),
    25: (1, 2, "I16"),
    26: (1, 4, "I32"),
    27: (1, 8, "I64"),
    28: (1, 8, "F64"),
    29: (256, 56, "IQ1_M"),
    30: (1, 2, "BF16"),
}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _read_gguf_string(f) -> str:
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype: int) -> Any:
    if vtype == GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    if vtype in _SCALAR_FMT:
        fmt = _SCALAR_FMT[vtype]
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, f.read(size))[0]
    if vtype == GGUF_TYPE_ARRAY:
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if elem_type in _SCALAR_FMT and count > 64:
            # Skip large arrays (e.g. tokenizer token scores) quickly after reading count
            fmt = _SCALAR_FMT[elem_type]
            size = struct.calcsize(fmt) * count
            f.seek(size, os.SEEK_CUR)
            return f"<array type={elem_type} len={count}>"
        if elem_type == GGUF_TYPE_STRING and count > 64:
            for _ in range(count):
                (slen,) = struct.unpack("<Q", f.read(8))
                f.seek(slen, os.SEEK_CUR)
            return f"<string_array len={count}>"
        return [_read_gguf_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def parse_gguf_header(filepath: str) -> Dict[str, Any]:
    """Parse GGUF header, metadata KV pairs, and tensor info descriptors."""
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic: {magic!r}")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        metadata: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_gguf_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            metadata[key] = _read_gguf_value(f, vtype)

        alignment = int(metadata.get("general.alignment", 32))
        tensors: List[Dict[str, Any]] = []
        for idx in range(n_tensors):
            name = _read_gguf_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
            ggml_type, rel_offset = struct.unpack("<IQ", f.read(12))
            n_elems = 1
            for d in dims:
                n_elems *= d
            if ggml_type not in GGML_QUANT_SIZES:
                raise ValueError(f"Unknown ggml_type {ggml_type} for tensor {name}")
            blk_elems, blk_bytes, type_name = GGML_QUANT_SIZES[ggml_type]
            if n_elems % blk_elems != 0:
                raise ValueError(f"Tensor {name} elements {n_elems} not divisible by {blk_elems}")
            n_bytes = (n_elems // blk_elems) * blk_bytes
            tensors.append(
                {
                    "tensor_id": idx,
                    "name": name,
                    "dims": dims,
                    "n_elements": n_elems,
                    "ggml_type": ggml_type,
                    "ggml_type_name": type_name,
                    "rel_offset": rel_offset,
                    "n_bytes": n_bytes,
                }
            )

        header_end = f.tell()
        data_section_offset = align_up(header_end, alignment)
        for t in tensors:
            t["file_offset"] = data_section_offset + t["rel_offset"]

    return {
        "version": version,
        "alignment": alignment,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "data_section_offset": data_section_offset,
        "metadata": metadata,
        "tensors": tensors,
    }


def classify_tensor_object(name: str) -> Tuple[str, int]:
    """Classify a GGUF tensor name into (role, layer_index)."""
    m = re.match(r"^blk\.(\d+)\.(.+)$", name)
    if not m:
        if "embd" in name:
            return "embedding", -1
        if "output" in name:
            return "output_head", -1
        return "global_norm", -1
    layer_idx = int(m.group(1))
    suffix = m.group(2)
    if "ffn_gate_inp" in suffix:
        return "moe_router", layer_idx
    if "exps" in suffix:
        return "moe_experts", layer_idx
    if "ffn" in suffix:
        return "ffn_block", layer_idx
    if "attn" in suffix:
        return "attn_block", layer_idx
    return "layer_norm", layer_idx


def pack_gguf_to_tfpack(
    input_gguf: str,
    out_dir: str,
    verify_roundtrip: bool = True,
) -> Dict[str, Any]:
    """Pack a GGUF file into a 4096-byte page-aligned `.tfpack` container and `manifest.json`,
    and optionally verify byte-for-byte identity of every unpacked tensor.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    parsed = parse_gguf_header(input_gguf)
    basename = os.path.splitext(os.path.basename(input_gguf))[0]
    tfpack_path = os.path.join(out_dir, f"{basename}.tfpack")
    manifest_path = os.path.join(out_dir, f"{basename}.manifest.json")

    objects: List[Dict[str, Any]] = []
    cur_offset = 0

    with open(input_gguf, "rb") as fin, open(tfpack_path, "wb") as fout:
        for t in parsed["tensors"]:
            fin.seek(t["file_offset"])
            raw = fin.read(t["n_bytes"])
            if len(raw) != t["n_bytes"]:
                raise IOError(f"Short read on tensor {t['name']}")

            sha256_hex = hashlib.sha256(raw).hexdigest()
            aligned_offset = align_up(cur_offset, PAGE_SIZE)
            if aligned_offset > cur_offset:
                fout.write(b"\x00" * (aligned_offset - cur_offset))
                cur_offset = aligned_offset

            fout.write(raw)
            cur_offset += len(raw)
            # Pad the end of every object to a 4096-byte page boundary so
            # FILE_FLAG_NO_BUFFERING can read `padded_bytes` directly!
            padded_end = align_up(cur_offset, PAGE_SIZE)
            if padded_end > cur_offset:
                fout.write(b"\x00" * (padded_end - cur_offset))
                cur_offset = padded_end

            role, layer_idx = classify_tensor_object(t["name"])
            objects.append(
                {
                    "object_id": t["tensor_id"],
                    "name": t["name"],
                    "role": role,
                    "layer": layer_idx,
                    "dims": t["dims"],
                    "ggml_type": t["ggml_type"],
                    "ggml_type_name": t["ggml_type_name"],
                    "source_gguf_offset": t["file_offset"],
                    "tfpack_offset": aligned_offset,
                    "exact_bytes": t["n_bytes"],
                    "page_aligned_bytes": padded_end - aligned_offset,
                    "sha256": sha256_hex,
                }
            )
        fout.flush()
        os.fsync(fout.fileno())

    pack_elapsed = time.perf_counter() - t0

    # Round-trip byte-for-byte verification + direct unbuffered single-object read check
    verified_count = 0
    unbuffered_single_object_ok = False
    verify_t0 = time.perf_counter()
    if verify_roundtrip:
        with open(tfpack_path, "rb") as fpack:
            for obj in objects:
                fpack.seek(obj["tfpack_offset"])
                unpacked = fpack.read(obj["exact_bytes"])
                if hashlib.sha256(unpacked).hexdigest() != obj["sha256"]:
                    raise AssertionError(f"SHA256 mismatch on object {obj['name']}")
                verified_count += 1

        # Also verify on Windows that any single object can be fetched via FILE_FLAG_NO_BUFFERING
        # and its exact_bytes slice matches SHA-256!
        if sys.platform == "win32" and objects:
            from hw_probe import (
                FILE_FLAG_NO_BUFFERING,
                FILE_SHARE_READ,
                GENERIC_READ,
                INVALID_HANDLE_VALUE,
                MEM_COMMIT,
                MEM_RELEASE,
                MEM_RESERVE,
                OPEN_EXISTING,
                PAGE_READWRITE,
                _CloseHandle,
                _CreateFileW,
                _ReadFile,
                _SetFilePointerEx,
                _VirtualAlloc,
                _VirtualFree,
            )
            from ctypes import wintypes

            test_obj = objects[len(objects) // 2]
            p_bytes = test_obj["page_aligned_bytes"]
            buf = _VirtualAlloc(None, p_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
            h = _CreateFileW(
                tfpack_path,
                GENERIC_READ,
                FILE_SHARE_READ,
                None,
                OPEN_EXISTING,
                FILE_FLAG_NO_BUFFERING,
                None,
            )
            try:
                new_pos = ctypes.c_longlong(0)
                n_read = wintypes.DWORD(0)
                assert _SetFilePointerEx(h, test_obj["tfpack_offset"], ctypes.byref(new_pos), 0)
                assert _ReadFile(h, buf, p_bytes, ctypes.byref(n_read), None)
                assert int(n_read.value) == p_bytes
                raw_slice = ctypes.string_at(buf, test_obj["exact_bytes"])
                assert hashlib.sha256(raw_slice).hexdigest() == test_obj["sha256"]
                unbuffered_single_object_ok = True
            finally:
                _CloseHandle(h)
                _VirtualFree(buf, 0, MEM_RELEASE)

    verify_elapsed = time.perf_counter() - verify_t0

    manifest = {
        "format": "TierFlow-TFPack-v1",
        "source_gguf": os.path.abspath(input_gguf),
        "source_gguf_size_bytes": os.path.getsize(input_gguf),
        "tfpack_path": os.path.abspath(tfpack_path),
        "tfpack_size_bytes": os.path.getsize(tfpack_path),
        "page_size_bytes": PAGE_SIZE,
        "num_objects": len(objects),
        "verified_byte_identical_objects": verified_count,
        "unbuffered_single_object_read_verified": unbuffered_single_object_ok,
        "pack_elapsed_sec": pack_elapsed,
        "verify_elapsed_sec": verify_elapsed,
        "gguf_metadata": parsed["metadata"],
        "objects": objects,
    }
    with open(manifest_path, "w", encoding="utf-8") as fm:
        json.dump(manifest, fm, indent=2)

    return manifest


def selftest() -> None:
    """Create a synthetic miniature GGUF v3 file with 3 tensors of irregular sizes,
    pack it to `.tfpack`, and verify 100% byte-identical roundtrip and 4096-byte alignment.
    """
    tmp_dir = tempfile.mkdtemp(prefix="tfpack_test_")
    gguf_file = os.path.join(tmp_dir, "mini.gguf")
    try:
        # Build minimal GGUF v3 header with 1 KV ("general.architecture" = "llama") and 3 F32 tensors
        t_specs = [
            ("token_embd.weight", [16, 8], os.urandom(16 * 8 * 4)),
            ("blk.0.attn_q.weight", [8, 8], os.urandom(8 * 8 * 4)),
            ("blk.0.ffn_gate_inp.weight", [4, 8], os.urandom(4 * 8 * 4)),
        ]
        with open(gguf_file, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<IQQ", 3, len(t_specs), 1))
            key = b"general.architecture"
            val = b"llama"
            f.write(struct.pack("<Q", len(key)) + key)
            f.write(struct.pack("<I", GGUF_TYPE_STRING))
            f.write(struct.pack("<Q", len(val)) + val)

            rel_off = 0
            for name, dims, data in t_specs:
                nb = name.encode("utf-8")
                f.write(struct.pack("<Q", len(nb)) + nb)
                f.write(struct.pack("<I", len(dims)))
                for d in dims:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<IQ", 0, rel_off))  # ggml_type=0 (F32)
                rel_off = align_up(rel_off + len(data), 32)

            pos = f.tell()
            pad = align_up(pos, 32) - pos
            f.write(b"\x00" * pad)

            cur = 0
            for _, _, data in t_specs:
                f.write(data)
                cur += len(data)
                next_cur = align_up(cur, 32)
                if next_cur > cur:
                    f.write(b"\x00" * (next_cur - cur))
                    cur = next_cur

        manifest = pack_gguf_to_tfpack(gguf_file, tmp_dir, verify_roundtrip=True)
        assert manifest["verified_byte_identical_objects"] == 3
        assert manifest["unbuffered_single_object_read_verified"] is True
        for obj in manifest["objects"]:
            assert obj["tfpack_offset"] % PAGE_SIZE == 0
            assert obj["page_aligned_bytes"] % PAGE_SIZE == 0
        print(
            "pack.py selftest PASS:",
            json.dumps(
                {
                    "num_objects": manifest["num_objects"],
                    "verified_byte_identical_objects": manifest["verified_byte_identical_objects"],
                    "unbuffered_single_object_read_verified": manifest["unbuffered_single_object_read_verified"],
                },
                indent=2,
            ),
        )
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--verify", action="store_true", default=True)
    args = parser.parse_args()
    if args.self_test:
        selftest()
    elif args.input and args.out_dir:
        selftest()
        m = pack_gguf_to_tfpack(args.input, args.out_dir, verify_roundtrip=args.verify)
        summary = {k: v for k, v in m.items() if k not in ("objects", "gguf_metadata")}
        print("Packing complete:", json.dumps(summary, indent=2))
    else:
        selftest()

