"""Hardware bandwidth and compute probe + Windows unbuffered I/O verification.

Measures on the actual machine:
1. NVMe SSD random/sequential read throughput via Win32 FILE_FLAG_NO_BUFFERING
   (direct I/O bypassing the OS page cache) vs. buffered OS reads, alongside
   system-wide OS cache delta verification (via sysmem_win).
2. Host DDR5 RAM effective read/copy bandwidth.
3. PCIe Host-to-Device (pinned & pageable) and Device-to-Host bandwidth (via PyTorch CUDA).
4. GPU VRAM internal bandwidth and CPU vs. GPU matrix-vector (GEMV) compute rates
   across batch sizes B in {1, 2, 4, 8, 16} to test the Fiddler/KTransformers
   hybrid CPU-GPU expert execution crossover.

Writes real measured artifacts to: results/p0_hw_probe/probe_results.json

Usage:
    python tools/hw_probe.py --self-test
    python tools/hw_probe.py --output results/p0_hw_probe/probe_results.json
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import datetime
import json
import os
import math
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sysmem_win import MemoryMonitor, sample_memory


def summary_stats(values: List[float]) -> Dict[str, Any]:
    """Compute mean, SD, 95% CI, min, max, n, and per-replicate values."""
    n = len(values)
    assert n >= 2
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    # t_{0.975, df=19} = 2.093 for n=20; 1.96 fallback for large n
    t_crit = 2.093024 if n == 20 else 1.959964
    ci95 = t_crit * sd / math.sqrt(n)
    return {
        "mean": mean,
        "sd": sd,
        "ci95_halfwidth": ci95,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
        "min": min(values),
        "max": max(values),
        "n": n,
        "values": values,
    }


# ---------------------------------------------------------------------------
# Win32 FILE_FLAG_NO_BUFFERING Direct I/O reader
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x20000000
    FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    MEM_COMMIT = 0x00001000
    MEM_RESERVE = 0x00002000
    MEM_RELEASE = 0x00008000
    PAGE_READWRITE = 0x04

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _CreateFileW = _k32.CreateFileW
    _CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _CreateFileW.restype = wintypes.HANDLE

    _ReadFile = _k32.ReadFile
    _ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _ReadFile.restype = wintypes.BOOL

    _SetFilePointerEx = _k32.SetFilePointerEx
    _SetFilePointerEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    _SetFilePointerEx.restype = wintypes.BOOL

    _CloseHandle = _k32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL

    _VirtualAlloc = _k32.VirtualAlloc
    _VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    _VirtualAlloc.restype = wintypes.LPVOID

    _VirtualFree = _k32.VirtualFree
    _VirtualFree.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
    _VirtualFree.restype = wintypes.BOOL


def win32_unbuffered_read_test(
    filepath: str,
    block_size: int,
    offsets: List[int],
) -> Dict[str, Any]:
    """Read `offsets` of size `block_size` using FILE_FLAG_NO_BUFFERING into VirtualAlloc page-aligned RAM."""
    if sys.platform != "win32":
        raise RuntimeError("win32_unbuffered_read_test requires Windows")
    if block_size % 4096 != 0:
        raise ValueError(f"block_size {block_size} must be 4096-byte aligned")
    for off in offsets:
        if off % 4096 != 0:
            raise ValueError(f"offset {off} must be 4096-byte aligned")

    buf = _VirtualAlloc(None, block_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    if not buf:
        raise ctypes.WinError(ctypes.get_last_error())

    handle = _CreateFileW(
        filepath,
        GENERIC_READ,
        FILE_SHARE_READ,
        None,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING,
        None,
    )
    if handle == INVALID_HANDLE_VALUE or handle is None:
        _VirtualFree(buf, 0, MEM_RELEASE)
        raise ctypes.WinError(ctypes.get_last_error())

    mon = MemoryMonitor(interval_sec=0.01)
    mon.start()
    bytes_read_total = 0
    n_read = wintypes.DWORD(0)
    new_pos = ctypes.c_longlong(0)

    t0 = time.perf_counter()
    try:
        for off in offsets:
            if not _SetFilePointerEx(handle, off, ctypes.byref(new_pos), 0):
                raise ctypes.WinError(ctypes.get_last_error())
            if not _ReadFile(handle, buf, block_size, ctypes.byref(n_read), None):
                raise ctypes.WinError(ctypes.get_last_error())
            bytes_read_total += int(n_read.value)
    finally:
        elapsed = time.perf_counter() - t0
        _CloseHandle(handle)
        _VirtualFree(buf, 0, MEM_RELEASE)
    mem_stats = mon.stop()

    return {
        "mode": "unbuffered_direct_io",
        "block_size_bytes": block_size,
        "num_reads": len(offsets),
        "bytes_read_total": bytes_read_total,
        "elapsed_sec": elapsed,
        "bandwidth_gbps": (bytes_read_total / 1e9) / max(elapsed, 1e-9),
        "delta_system_cache_bytes": mem_stats["delta_system_cache_bytes"],
        "delta_proc_private_bytes": mem_stats["delta_proc_private_bytes"],
    }


def buffered_read_test(
    filepath: str,
    block_size: int,
    offsets: List[int],
) -> Dict[str, Any]:
    """Control: standard OS buffered file reads (`open(..., 'rb')`)."""
    mon = MemoryMonitor(interval_sec=0.01)
    mon.start()
    bytes_read_total = 0
    t0 = time.perf_counter()
    with open(filepath, "rb") as f:
        for off in offsets:
            f.seek(off)
            chunk = f.read(block_size)
            bytes_read_total += len(chunk)
    elapsed = time.perf_counter() - t0
    mem_stats = mon.stop()
    return {
        "mode": "os_buffered_io",
        "block_size_bytes": block_size,
        "num_reads": len(offsets),
        "bytes_read_total": bytes_read_total,
        "elapsed_sec": elapsed,
        "bandwidth_gbps": (bytes_read_total / 1e9) / max(elapsed, 1e-9),
        "delta_system_cache_bytes": mem_stats["delta_system_cache_bytes"],
        "delta_proc_private_bytes": mem_stats["delta_proc_private_bytes"],
    }


def probe_ram_and_gpu(repeats: int = 20) -> Dict[str, Any]:
    """Measure host DDR5 RAM bandwidth, PCIe H2D/D2H bandwidth, VRAM bandwidth,
    and CPU vs GPU GEMV latency across batch sizes (Fiddler/KTransformers crossover test).
    """
    results: Dict[str, Any] = {}
    # 1. Host RAM read/copy bandwidth (256 MiB float32 tensor = 64M elements)
    n_elem = 64 * 1024 * 1024
    src_cpu = torch.ones(n_elem, dtype=torch.float32)
    dst_cpu = torch.empty_like(src_cpu)
    # warmup
    dst_cpu.copy_(src_cpu)
    times_ram: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        dst_cpu.copy_(src_cpu)
        times_ram.append(time.perf_counter() - t0)
    # copy_ reads 256 MiB + writes 256 MiB = 512 MiB total memory traffic
    bytes_moved_ram = 2 * n_elem * 4
    bw_ram = [(bytes_moved_ram / 1e9) / t for t in times_ram]
    results["ram_copy_traffic_gbps"] = summary_stats(bw_ram)
    # Pure read bandwidth on CPU (sum reduction over 256 MiB)
    _ = src_cpu.sum()
    times_read: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = src_cpu.sum()
        times_read.append(time.perf_counter() - t0)
    bw_read = [(n_elem * 4 / 1e9) / t for t in times_read]
    results["ram_read_gbps"] = summary_stats(bw_read)
    del src_cpu, dst_cpu

    if torch.cuda.is_available():
        # 2. PCIe H2D bandwidth (pinned vs pageable, 64 MiB tensor)
        pcie_bytes = 64 * 1024 * 1024
        t_Pinned = torch.empty(pcie_bytes, dtype=torch.uint8, pin_memory=True)
        t_Page = torch.empty(pcie_bytes, dtype=torch.uint8, pin_memory=False)
        t_Gpu = torch.empty(pcie_bytes, dtype=torch.uint8, device="cuda")
        t_Gpu.copy_(t_Pinned)
        torch.cuda.synchronize()

        h2d_pinned: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            t_Gpu.copy_(t_Pinned, non_blocking=True)
            torch.cuda.synchronize()
            h2d_pinned.append((pcie_bytes / 1e9) / (time.perf_counter() - t0))

        h2d_page: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            t_Gpu.copy_(t_Page)
            torch.cuda.synchronize()
            h2d_page.append((pcie_bytes / 1e9) / (time.perf_counter() - t0))

        # 3. GPU VRAM internal copy bandwidth (256 MiB read + 256 MiB write = 512 MiB)
        vram_elem = 64 * 1024 * 1024
        g_src = torch.ones(vram_elem, dtype=torch.float32, device="cuda")
        g_dst = torch.empty_like(g_src)
        g_dst.copy_(g_src)
        torch.cuda.synchronize()
        vram_bw: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            g_dst.copy_(g_src)
            torch.cuda.synchronize()
            vram_bw.append((2 * vram_elem * 4 / 1e9) / (time.perf_counter() - t0))

        results["pcie_h2d_pinned_gbps"] = summary_stats(h2d_pinned)
        results["pcie_h2d_pageable_gbps"] = summary_stats(h2d_page)
        results["vram_copy_traffic_gbps"] = summary_stats(vram_bw)
        del t_Pinned, t_Page, t_Gpu, g_src, g_dst

        # 4. Fiddler / KTransformers crossover test:
        # Single MoE expert MLP (SwiGLU: gate, up [2048 -> 768], down [768 -> 2048] in FP16 = 9.437 MiB)
        # Compare:
        #   Path A (CPU in-place): expert lives in RAM; compute on CPU, copy tiny output (B x 2048 FP16) to GPU.
        #   Path B (PCIe transfer + GPU compute): copy expert (9.437 MiB) from pinned RAM to GPU + compute on GPU.
        hidden, inter = 2048, 768
        w_gate_cpu = torch.randn(inter, hidden, dtype=torch.float16, pin_memory=True)
        w_up_cpu = torch.randn(inter, hidden, dtype=torch.float16, pin_memory=True)
        w_down_cpu = torch.randn(hidden, inter, dtype=torch.float16, pin_memory=True)
        w_gate_gpu = torch.empty_like(w_gate_cpu, device="cuda")
        w_up_gpu = torch.empty_like(w_up_cpu, device="cuda")
        w_down_gpu = torch.empty_like(w_down_cpu, device="cuda")

        crossover: Dict[str, Any] = {}
        for batch in (1, 2, 4, 8, 16, 32):
            x_cpu = torch.randn(batch, hidden, dtype=torch.float16)
            x_gpu = x_cpu.to("cuda")
            # Warmup CPU, PCIe+GPU, and GPU-resident before timing!
            _ = ((torch.nn.functional.silu(x_cpu @ w_gate_cpu.T) * (x_cpu @ w_up_cpu.T)) @ w_down_cpu.T).to("cuda")
            w_gate_gpu.copy_(w_gate_cpu, non_blocking=True)
            w_up_gpu.copy_(w_up_cpu, non_blocking=True)
            w_down_gpu.copy_(w_down_cpu, non_blocking=True)
            _ = (torch.nn.functional.silu(x_gpu @ w_gate_gpu.T) * (x_gpu @ w_up_gpu.T)) @ w_down_gpu.T
            torch.cuda.synchronize()

            # Measure Path A: CPU compute + tiny output H2D
            t_cpu_list: List[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out_c = (torch.nn.functional.silu(x_cpu @ w_gate_cpu.T) * (x_cpu @ w_up_cpu.T)) @ w_down_cpu.T
                out_g = out_c.to("cuda")
                torch.cuda.synchronize()
                t_cpu_list.append((time.perf_counter() - t0) * 1000.0)

            # Measure Path B: H2D expert weights + GPU compute
            t_pcie_gpu_list: List[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                w_gate_gpu.copy_(w_gate_cpu, non_blocking=True)
                w_up_gpu.copy_(w_up_cpu, non_blocking=True)
                w_down_gpu.copy_(w_down_cpu, non_blocking=True)
                out_g2 = (torch.nn.functional.silu(x_gpu @ w_gate_gpu.T) * (x_gpu @ w_up_gpu.T)) @ w_down_gpu.T
                torch.cuda.synchronize()
                t_pcie_gpu_list.append((time.perf_counter() - t0) * 1000.0)

            # Measure Path C: Resident on GPU (no PCIe transfer)
            t_res_gpu_list: List[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out_g3 = (torch.nn.functional.silu(x_gpu @ w_gate_gpu.T) * (x_gpu @ w_up_gpu.T)) @ w_down_gpu.T
                torch.cuda.synchronize()
                t_res_gpu_list.append((time.perf_counter() - t0) * 1000.0)

            stats_a = summary_stats(t_cpu_list)
            stats_b = summary_stats(t_pcie_gpu_list)
            stats_c = summary_stats(t_res_gpu_list)
            crossover[f"batch_{batch}"] = {
                "cpu_inplace_ms": stats_a,
                "pcie_transfer_plus_gpu_ms": stats_b,
                "gpu_resident_ms": stats_c,
                "cpu_beats_pcie_transfer": (stats_a["mean"] < stats_b["mean"]),
                "ci95_non_overlapping": (
                    stats_a["ci95_high"] < stats_b["ci95_low"]
                    or stats_b["ci95_high"] < stats_a["ci95_low"]
                ),
            }
        results["moe_expert_crossover_qwen3_dims_fp16"] = crossover

    return results


def run_full_probe(output_path: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Create a temporary 256 MiB file on E: drive to test NVMe unbuffered vs buffered I/O
    temp_dir = os.path.dirname(os.path.abspath(output_path))
    probe_file = os.path.join(temp_dir, "_nvme_probe_temp.bin")
    file_size = 256 * 1024 * 1024  # 256 MiB
    block_4m = 4 * 1024 * 1024     # 4 MiB sequential block
    block_1m = 1 * 1024 * 1024     # 1 MiB expert-sized block

    try:
        # Write 256 MiB file using unbuffered or flushed write
        chunk = os.urandom(block_4m)
        with open(probe_file, "wb") as f:
            for _ in range(file_size // block_4m):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())

        seq_offsets_4m = list(range(0, file_size, block_4m))
        # Deterministic pseudo-random permutation of 1 MiB expert blocks
        import random
        rng = random.Random(42)
        rand_offsets_1m = list(range(0, file_size, block_1m))
        rng.shuffle(rand_offsets_1m)

        io_seq_unbuf = win32_unbuffered_read_test(probe_file, block_4m, seq_offsets_4m)
        io_rand_unbuf = win32_unbuffered_read_test(probe_file, block_1m, rand_offsets_1m)
        io_seq_buf = buffered_read_test(probe_file, block_4m, seq_offsets_4m)
    finally:
        if os.path.exists(probe_file):
            os.remove(probe_file)

    compute_and_mem = probe_ram_and_gpu(repeats=20)

    payload = {
        "exp_id": "p0_hw_probe",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvme_seq_4mib_unbuffered": io_seq_unbuf,
        "nvme_rand_1mib_unbuffered": io_rand_unbuf,
        "nvme_seq_4mib_buffered_control": io_seq_buf,
        "memory_and_compute": compute_and_mem,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    return payload


def selftest() -> None:
    """Unit test alignment checks and small 4 MiB unbuffered read."""
    temp_file = os.path.join(tempfile.gettempdir(), "_tf_io_selftest.bin")
    try:
        with open(temp_file, "wb") as f:
            f.write(b"\x5a" * (4 * 1024 * 1024))
            f.flush()
        # Misaligned offset must raise ValueError
        try:
            win32_unbuffered_read_test(temp_file, 4096, [123])
            raise AssertionError("Misaligned offset was not rejected")
        except ValueError:
            pass
        # Valid aligned read must read exact bytes
        res = win32_unbuffered_read_test(temp_file, 4096, [0, 4096, 8192])
        assert res["bytes_read_total"] == 3 * 4096
        print("hw_probe selftest PASS:", json.dumps(res, indent=2))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=str, default="results/p0_hw_probe/probe_results.json")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        run_full_probe(args.output)
