"""System-wide memory sampler for Windows (and Linux fallback).

Measures true system-wide physical memory usage, OS page/standby cache,
process private bytes, process working set, and board-wide GPU memory via NVML.
This prevents the 'Process RSS excludes OS page cache' error identified in
Memory-Sovereign Inference (Stepanek, 2026).

Usage:
    python tools/sysmem_win.py --self-test
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class MemorySample:
    timestamp_utc: float
    # System-wide physical memory (bytes)
    phys_total_bytes: int
    phys_avail_bytes: int
    phys_used_bytes: int
    system_cache_bytes: int
    # Process-specific memory (bytes)
    proc_private_bytes: int
    proc_working_set_bytes: int
    proc_peak_working_set_bytes: int
    # Board-wide GPU memory (bytes, 0 if no GPU/NVML)
    gpu_total_bytes: int
    gpu_used_bytes: int
    gpu_free_bytes: int


# ---------------------------------------------------------------------------
# Windows Win32 / PSAPI structures
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", wintypes.DWORD),
            ("ProcessCount", wintypes.DWORD),
            ("ThreadCount", wintypes.DWORD),
        ]

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _GetPerformanceInfo = _psapi.GetPerformanceInfo
    _GetPerformanceInfo.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), wintypes.DWORD]
    _GetPerformanceInfo.restype = wintypes.BOOL

    _GetProcessMemoryInfo = _psapi.GetProcessMemoryInfo
    _GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    _GetProcessMemoryInfo.restype = wintypes.BOOL

    _GetCurrentProcess = _kernel32.GetCurrentProcess
    _GetCurrentProcess.argtypes = []
    _GetCurrentProcess.restype = wintypes.HANDLE


# ---------------------------------------------------------------------------
# Direct NVML C-API binding (avoids deprecated pynvml warning)
# ---------------------------------------------------------------------------
class _NVMLMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NVMLClient:
    def __init__(self) -> None:
        self.lib = None
        self.handle = ctypes.c_void_p()
        try:
            if sys.platform == "win32":
                try:
                    self.lib = ctypes.CDLL("nvml.dll")
                except OSError:
                    program_files = os.environ.get("ProgramW6432", r"C:\Program Files")
                    alt = os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvml.dll")
                    self.lib = ctypes.CDLL(alt)
            else:
                self.lib = ctypes.CDLL("libnvidia-ml.so.1")
            if self.lib.nvmlInit_v2() == 0:
                if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
                    self.lib = None
            else:
                self.lib = None
        except Exception:
            self.lib = None

    def query_gpu0(self) -> tuple[int, int, int]:
        if self.lib is None:
            return 0, 0, 0
        mem = _NVMLMemory()
        res = self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem))
        if res != 0:
            return 0, 0, 0
        return int(mem.total), int(mem.used), int(mem.free)


_NVML = _NVMLClient()


def sample_memory(proc_handle: Optional[int] = None) -> MemorySample:
    """Take one instantaneous snapshot of system, process, and GPU memory."""
    now = time.time()
    gpu_total, gpu_used, gpu_free = _NVML.query_gpu0()

    if sys.platform == "win32":
        pi = PERFORMANCE_INFORMATION()
        pi.cb = ctypes.sizeof(PERFORMANCE_INFORMATION)
        if not _GetPerformanceInfo(ctypes.byref(pi), pi.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        page = int(pi.PageSize)
        phys_total = int(pi.PhysicalTotal) * page
        phys_avail = int(pi.PhysicalAvailable) * page
        phys_used = phys_total - phys_avail
        sys_cache = int(pi.SystemCache) * page

        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        h = proc_handle if proc_handle is not None else _GetCurrentProcess()
        if not _GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return MemorySample(
            timestamp_utc=now,
            phys_total_bytes=phys_total,
            phys_avail_bytes=phys_avail,
            phys_used_bytes=phys_used,
            system_cache_bytes=sys_cache,
            proc_private_bytes=int(pmc.PrivateUsage),
            proc_working_set_bytes=int(pmc.WorkingSetSize),
            proc_peak_working_set_bytes=int(pmc.PeakWorkingSetSize),
            gpu_total_bytes=gpu_total,
            gpu_used_bytes=gpu_used,
            gpu_free_bytes=gpu_free,
        )
    else:
        # Linux /proc fallback
        meminfo: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = int(parts[1].strip().split()[0]) * 1024
                    meminfo[k] = v
        phys_total = meminfo.get("MemTotal", 0)
        phys_avail = meminfo.get("MemAvailable", 0)
        sys_cache = meminfo.get("Cached", 0) + meminfo.get("Buffers", 0)
        rss = 0
        vms = 0
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            fields = f.read().split()
            page = os.sysconf("SC_PAGE_SIZE")
            vms = int(fields[0]) * page
            rss = int(fields[1]) * page
        return MemorySample(
            timestamp_utc=now,
            phys_total_bytes=phys_total,
            phys_avail_bytes=phys_avail,
            phys_used_bytes=phys_total - phys_avail,
            system_cache_bytes=sys_cache,
            proc_private_bytes=vms,
            proc_working_set_bytes=rss,
            proc_peak_working_set_bytes=rss,
            gpu_total_bytes=gpu_total,
            gpu_used_bytes=gpu_used,
            gpu_free_bytes=gpu_free,
        )


class MemoryMonitor:
    """Background 100 ms sampler recording baseline and peak memory footprints."""

    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.samples: List[MemorySample] = []
        self.sample_errors: List[str] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline: Optional[MemorySample] = None

    def start(self) -> None:
        self.baseline = sample_memory()
        self.samples = [self.baseline]
        self.sample_errors = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(sample_memory())
            except OSError as exc:
                self.sample_errors.append(f"loop sample error: {exc}")
                print(f"[sysmem_win] sample error: {exc}", flush=True)
            self._stop.wait(self.interval_sec)

    def stop(self) -> Dict[str, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.samples.append(sample_memory())
        except OSError as exc:
            self.sample_errors.append(f"stop sample error: {exc}")
            print(f"[sysmem_win] stop sample error: {exc}", flush=True)
        assert self.baseline is not None and len(self.samples) >= 1
        base = self.baseline
        peak_proc_priv = max(s.proc_private_bytes for s in self.samples)
        peak_proc_ws = max(s.proc_working_set_bytes for s in self.samples)
        peak_sys_used = max(s.phys_used_bytes for s in self.samples)
        peak_cache = max(s.system_cache_bytes for s in self.samples)
        peak_gpu_used = max(s.gpu_used_bytes for s in self.samples)

        return {
            "baseline_phys_used_bytes": base.phys_used_bytes,
            "baseline_system_cache_bytes": base.system_cache_bytes,
            "baseline_proc_private_bytes": base.proc_private_bytes,
            "baseline_gpu_used_bytes": base.gpu_used_bytes,
            "peak_proc_private_bytes": peak_proc_priv,
            "delta_proc_private_bytes": max(0, peak_proc_priv - base.proc_private_bytes),
            "peak_proc_working_set_bytes": peak_proc_ws,
            "peak_system_used_bytes": peak_sys_used,
            "delta_system_used_bytes": max(0, peak_sys_used - base.phys_used_bytes),
            "peak_system_cache_bytes": peak_cache,
            "delta_system_cache_bytes": max(0, peak_cache - base.system_cache_bytes),
            "peak_gpu_used_bytes": peak_gpu_used,
            "delta_gpu_used_bytes": max(0, peak_gpu_used - base.gpu_used_bytes),
            "gpu_total_bytes": base.gpu_total_bytes,
            "phys_total_bytes": base.phys_total_bytes,
            "num_samples": len(self.samples),
        }


def selftest() -> None:
    """Verify known-answer properties of the memory monitor:
    1. Physical total > 0 and available <= total.
    2. Allocating a 64 MiB bytearray increases proc_private_bytes by >= 60 MiB.
    """
    s0 = sample_memory()
    assert s0.phys_total_bytes > 1024 * 1024 * 1024, f"Unexpected RAM: {s0.phys_total_bytes}"
    assert 0 < s0.phys_avail_bytes <= s0.phys_total_bytes

    mon = MemoryMonitor(interval_sec=0.02)
    mon.start()
    alloc_size = 64 * 1024 * 1024
    buf = bytearray(alloc_size)
    # Touch every page (4 KiB stride) so OS commits physical pages
    for offset in range(0, alloc_size, 4096):
        buf[offset] = 1
    time.sleep(0.08)
    summary = mon.stop()
    del buf

    assert summary["delta_proc_private_bytes"] >= 60 * 1024 * 1024, (
        f"Expected >= 60 MiB private delta, got {summary['delta_proc_private_bytes']}"
    )
    print("sysmem_win selftest PASS:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        selftest()
    else:
        print(json.dumps(asdict(sample_memory()), indent=2))

