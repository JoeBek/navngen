#!/usr/bin/env python3
"""Modular Docker pipeline orchestrator.

Chains the containerized stages:

    images --> [depth] --\
    images --> [segment] -+-> [filter] --> [lightglue] --> [backend] --> out.tum
    images --> [superpoint]

The three producers (depth, segmentation, superpoint) are independent and run
concurrently; GPU producers are serialized by default to avoid VRAM contention.
Each stage is a `docker run`; intermediate artifacts live under a single
work_dir and are reused on re-runs (skip-if-present unless --force).

Run on the HOST (needs docker + pyyaml). Config: see scripts/configs/pipeline.yaml.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(msg: str):
    print(f"[pipeline] {msg}", flush=True)


def nonempty_dir(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def count_npz(p: Path) -> int:
    return len(list(p.glob("*.npz")))


def count_npy(p: Path) -> int:
    return len(list(p.glob("*.npy")))


def derive_intrinsics_npy(calib_path: Path, calib_type: str, out_path: Path):
    """Write a 3x3 pinhole intrinsics .npy for the depth stage, derived from the
    navngen calibration so the user supplies calibration only once."""
    if calib_type == "surfnav":
        data = np.asarray(yaml.safe_load(calib_path.read_text())["camera_matrix"]["data"], float)
        K = data.reshape(3, 3)
    elif calib_type == "euroc":
        fx, fy, cx, cy = (float(v) for v in yaml.safe_load(calib_path.read_text())["intrinsics"])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    elif calib_type == "kitti":
        K = None
        for line in calib_path.read_text().splitlines():
            if line.startswith("P0:"):
                p = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
                K = np.array([[p[0, 0], 0, p[0, 2]], [0, p[1, 1], p[1, 2]], [0, 0, 1]], float)
                break
        if K is None:
            raise ValueError("No P0 line in kitti calib")
    else:
        raise ValueError(f"Unknown calib type {calib_type}")
    np.save(out_path, K.astype(np.float64))
    return out_path


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(self, cfg: dict, force: bool, dry_run: bool):
        self.cfg = cfg
        self.force = force
        self.dry_run = dry_run

        self.images = Path(cfg["input"]["images"]).resolve()
        self.times = cfg["input"].get("times")
        self.times = Path(self.times).resolve() if self.times else None

        self.work = Path(cfg["work_dir"]).resolve()
        self.calib = Path(cfg["calib"]["navngen_yaml"]).resolve()
        self.calib_type = cfg["calib"]["navngen_type"]
        self.hf_token = cfg.get("hf_token")
        self.hf_token = Path(self.hf_token).resolve() if self.hf_token else None
        self.hf_cache = Path(cfg.get("hf_cache", self.work / "hf_cache")).resolve()
        self.images_on_gpu = bool(cfg.get("images_on_gpu", True))
        self.gpu_serialize = bool(cfg.get("scheduling", {}).get("gpu_serialize", True))
        self.uid_gid = f"{os.getuid()}:{os.getgid()}"
        self.stages = cfg["stages"]

        # work subdirs
        self.d_depth = self.work / "depth_masks"
        self.d_seg = self.work / "seg_masks"
        self.d_sp = self.work / "superpoint"
        self.d_filt = self.work / "filtered"
        self.d_match = self.work / "matches"
        self.out_tum = self.work / "out.tum"

    # -- docker command runner -------------------------------------------- #
    def run_docker(self, name: str, args: list[str]):
        cmd = ["docker", "run", "--rm", "--user", self.uid_gid] + args
        log(f"[{name}] $ {' '.join(cmd)}")
        if self.dry_run:
            return
        t0 = time.time()
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"stage '{name}' failed (exit {proc.returncode})")
        log(f"[{name}] done in {time.time() - t0:.1f}s")

    # -- stages ------------------------------------------------------------ #
    def stage_depth(self):
        s = self.stages["depth"]
        if not self.force and nonempty_dir(self.d_depth):
            log("[depth] skip (output present)")
            return
        if not self.hf_token:
            raise ValueError("depth stage needs calib.hf_token")
        K_npy = self.work / "K.npy"
        if not self.dry_run:
            derive_intrinsics_npy(self.calib, self.calib_type, K_npy)
        self.run_docker("depth", [
            "--gpus", "all", "--network", "host",
            "-e", "HF_HOME=/cache/huggingface", "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp",
            "-v", f"{self.hf_token}:/app/.hf_token:ro",
            "-v", f"{K_npy}:/app/my_intrinsics.npy:ro",
            "-v", f"{self.images}:/data/in:ro",
            "-v", f"{self.work}:/app/output",
            "-v", f"{self.hf_cache}:/cache/huggingface",
            s["image"],
            "--input_dir", "/data/in", "--calib_path", "my_intrinsics.npy", "-o", "mask",
        ])

    def stage_segmentation(self):
        s = self.stages["segmentation"]
        if not self.force and nonempty_dir(self.d_seg):
            log("[segmentation] skip (output present)")
            return
        driver = Path(s["driver"]).resolve()
        self.d_seg.mkdir(parents=True, exist_ok=True)
        gpu = bool(s.get("gpu", True))
        gpu_flags = ["--gpus", "all"] if gpu else []
        device = "cuda:0" if gpu else "cpu"
        self.run_docker("segmentation", gpu_flags + [
            "--network", "host",
            "-e", "HOME=/tmp", "-e", "XDG_CACHE_HOME=/cache", "-e", "TORCH_HOME=/cache/torch",
            "-e", "MPLCONFIGDIR=/tmp",
            "-v", f"{driver}:/workspace/get_segmentation.py:ro",
            "-v", f"{self.images}:/data/in:ro",
            "-v", f"{self.d_seg}:/out",
            "-v", f"{self.hf_cache}:/cache",
            "--entrypoint", "python3", s["image"],
            "/workspace/get_segmentation.py", "--input_dir", "/data/in", "--output_dir", "/out",
            "--device", device,
        ])

    def stage_superpoint(self):
        s = self.stages["superpoint"]
        if not self.force and nonempty_dir(self.d_sp):
            log("[superpoint] skip (output present)")
            return
        self.d_sp.mkdir(parents=True, exist_ok=True)
        gpu = ["--gpus", "all"] if self.images_on_gpu else []
        device = "cuda" if self.images_on_gpu else "cpu"
        self.run_docker("superpoint", gpu + [
            "-e", "HOME=/tmp",
            "-v", f"{self.images}:/in:ro",
            "-v", f"{self.d_sp}:/out",
            s["image"],
            "--input_path", "/in", "--output_path", "/out",
            "--max_num_keypoints", str(s.get("max_keypoints", 2048)),
            "--device", device,
        ])

    def stage_filter(self):
        s = self.stages["filter"]
        if not self.force and nonempty_dir(self.d_filt):
            log("[filter] skip (output present)")
            return
        mode = s["mode"]
        # preflight: counts must match for whichever masks are used
        nsp = count_npz(self.d_sp)
        if mode in ("depth", "both"):
            nd = count_npy(self.d_depth)
            if nd != nsp:
                raise RuntimeError(f"filter: {nsp} features vs {nd} depth maps (counts must match)")
        if mode in ("seg", "both"):
            ns = count_npy(self.d_seg)
            if ns != nsp:
                raise RuntimeError(f"filter: {nsp} features vs {ns} seg masks (counts must match)")
        self.d_filt.mkdir(parents=True, exist_ok=True)
        extra = []
        if mode in ("depth", "both"):
            extra += ["--depth_path", "/work/depth_masks", "--tl", str(s.get("tl", 0.0)),
                      "--th", str(s.get("th", 0.0))]
            if s.get("normalize", False):
                extra += ["--normalize"]
        if mode in ("seg", "both"):
            extra += ["--seg_path", "/work/seg_masks"]
            if s.get("filter_ids"):
                extra += ["--filter_ids", str(s["filter_ids"])]
        self.run_docker("filter", [
            "-v", f"{self.work}:/work", self.stages["filter"]["image"],
            "--features_path", "/work/superpoint", "--output_path", "/work/filtered",
            "--mode", mode, *extra,
        ])

    def stage_lightglue(self):
        s = self.stages["lightglue"]
        if not self.force and nonempty_dir(self.d_match):
            log("[lightglue] skip (output present)")
            return
        # features from filter output if enabled, else raw superpoint
        feat = "/work/filtered" if self.stages["filter"].get("enabled") else "/work/superpoint"
        self.d_match.mkdir(parents=True, exist_ok=True)
        gpu = ["--gpus", "all"] if self.images_on_gpu else []
        device = "cuda" if self.images_on_gpu else "cpu"
        self.run_docker("lightglue", gpu + [
            "-e", "HOME=/tmp", "-v", f"{self.work}:/work", s["image"],
            "--features_path", feat, "--output_path", "/work/matches", "--device", device,
        ])

    def stage_backend(self):
        s = self.stages["backend"]
        if not self.force and self.out_tum.exists():
            log("[backend] skip (output present)")
            return
        mounts = ["-v", f"{self.work}:/work", "-v", f"{self.calib}:/calib:ro"]
        extra = ["--config_path", "/calib", "--config_type", self.calib_type]
        if self.times:
            mounts += ["-v", f"{self.times}:/times.txt:ro"]
            extra += ["--times_path", "/times.txt"]
        scale_mode = s.get("scale_mode", "unit")
        extra += ["--scale_mode", scale_mode, "--scale", str(s.get("scale", 1.0))]
        if scale_mode == "metric":
            extra += ["--depth_path", "/work/depth_masks"]
        self.run_docker("backend", mounts + [
            s["image"],
            "--matches_path", "/work/matches", "--output_path", "/work/out.tum", *extra,
        ])

    # -- scheduling -------------------------------------------------------- #
    def run_producers(self):
        producers = []  # (name, fn, is_gpu)
        if self.stages["depth"].get("enabled"):
            producers.append(("depth", self.stage_depth, bool(self.stages["depth"].get("gpu", True))))
        if self.stages["segmentation"].get("enabled"):
            producers.append(("segmentation", self.stage_segmentation,
                              bool(self.stages["segmentation"].get("gpu", True))))
        if self.stages["superpoint"].get("enabled"):
            producers.append(("superpoint", self.stage_superpoint, self.images_on_gpu))

        gpu_stages = [p for p in producers if p[2]]
        cpu_stages = [p for p in producers if not p[2]]
        errors = []

        def worker(name, fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                errors.append((name, e))

        threads = [threading.Thread(target=worker, args=(n, f)) for n, f, _ in cpu_stages]
        for t in threads:
            t.start()

        if self.gpu_serialize:
            for n, f, _ in gpu_stages:
                worker(n, f)
        else:
            gthreads = [threading.Thread(target=worker, args=(n, f)) for n, f, _ in gpu_stages]
            for t in gthreads:
                t.start()
            threads += gthreads

        for t in threads:
            t.join()

        if errors:
            for n, e in errors:
                log(f"ERROR in producer '{n}': {e}")
            raise RuntimeError("one or more producer stages failed")

    def run(self):
        self.work.mkdir(parents=True, exist_ok=True)
        self.hf_cache.mkdir(parents=True, exist_ok=True)

        log(f"work_dir = {self.work}")
        self.run_producers()

        if self.stages["filter"].get("enabled"):
            self.stage_filter()
        if self.stages["lightglue"].get("enabled"):
            self.stage_lightglue()
        if self.stages["backend"].get("enabled"):
            self.stage_backend()

        log(f"pipeline complete -> {self.out_tum}")


def main():
    parser = argparse.ArgumentParser(description="Run the modular Docker VO pipeline.")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline YAML config.")
    parser.add_argument("--force", action="store_true", help="Re-run stages even if outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print docker commands without running.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    Pipeline(cfg, force=args.force, dry_run=args.dry_run).run()


if __name__ == "__main__":
    main()
