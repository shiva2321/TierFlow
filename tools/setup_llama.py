"""Bootstrap and build the pinned third_party/llama.cpp dependency with TierFlow's hook patch."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(REPO_ROOT, "third_party", "llama.cpp")
PATCH_FILE = os.path.join(REPO_ROOT, "third_party", "patches", "llama_tierflow_v2.patch")
PINNED_COMMIT = "3ba1ce0e28d991800e1585b7d3533f77213939d6"
UPSTREAM_URL = "https://github.com/ggml-org/llama.cpp.git"


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def ensure_llama_checkout() -> None:
    if not os.path.exists(os.path.join(LLAMA_DIR, ".git")):
        os.makedirs(os.path.dirname(LLAMA_DIR), exist_ok=True)
        run(["git", "clone", "--depth", "1", UPSTREAM_URL, LLAMA_DIR])
        run(["git", "fetch", "--depth", "1", "origin", PINNED_COMMIT], cwd=LLAMA_DIR)
        run(["git", "checkout", PINNED_COMMIT], cwd=LLAMA_DIR)


def ensure_patch_applied() -> None:
    check_applied = subprocess.run(
        ["git", "apply", "--reverse", "--check", PATCH_FILE],
        cwd=LLAMA_DIR,
        capture_output=True,
        text=True,
    )
    if check_applied.returncode == 0:
        print("TierFlow llama.cpp patch is already applied.")
        return

    run(["git", "apply", "--check", PATCH_FILE], cwd=LLAMA_DIR)
    run(["git", "apply", PATCH_FILE], cwd=LLAMA_DIR)
    print("Applied TierFlow llama.cpp patch cleanly.")


def build_llama(gpu_backend: str) -> None:
    build_dir = os.path.join(LLAMA_DIR, "build")
    cmake_args = [
        "cmake",
        "-S",
        LLAMA_DIR,
        "-B",
        build_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=ON",
    ]
    if shutil.which("ninja"):
        cmake_args.extend(["-G", "Ninja"])
    if gpu_backend == "cuda":
        cmake_args.append("-DGGML_CUDA=ON")
    elif gpu_backend == "vulkan":
        cmake_args.append("-DGGML_VULKAN=ON")

    run(cmake_args)
    run(["cmake", "--build", build_dir, "--config", "Release", "--parallel"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup and build pinned llama.cpp for TierFlow.")
    parser.add_argument("--gpu", choices=["none", "cuda", "vulkan"], default="none")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    ensure_llama_checkout()
    ensure_patch_applied()
    if not args.skip_build:
        build_llama(args.gpu)
    return 0


if __name__ == "__main__":
    sys.exit(main())

