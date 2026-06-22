"""GPU graphics-clock locking helpers.

Locking the graphics clock removes DVFS jitter and makes inference timings
reproducible across runs — useful when reporting numbers. Locking requires
``sudo nvidia-smi``; this module does NOT embed any password. When sudo is
unavailable or non-interactive, locking fails gracefully with a warning and
inference proceeds at the default (boosting) clock.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def _nvidia_smi() -> str | None:
    return shutil.which("nvidia-smi")


def query_max_graphics_clock(device: int = 0) -> int | None:
    """Return the GPU's maximum supported graphics clock (MHz), or None."""
    smi = _nvidia_smi()
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [smi, f"--id={device}", "--query-gpu=clocks.max.graphics",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError, OSError, subprocess.SubprocessError):
        return None


def lock_gpu_clock(clock: int | None = None, device: int = 0) -> bool:
    """Lock the graphics clock to ``clock`` MHz (or the max supported).

    Returns True on success. On failure (no nvidia-smi, sudo declined, etc.)
    prints a warning and returns False without raising — callers should
    continue regardless.
    """
    smi = _nvidia_smi()
    if smi is None:
        print("    [clock] nvidia-smi not found; skipping clock lock.",
              file=sys.stderr)
        return False

    if clock is None:
        clock = query_max_graphics_clock(device)
    if not clock:
        print("    [clock] could not determine a target clock; skipping lock.",
              file=sys.stderr)
        return False

    try:
        # No password is piped: sudo prompts on the controlling terminal.
        result = subprocess.run(
            ["sudo", smi, f"--id={device}", "--lock-gpu-clocks", str(clock)],
        )
    except OSError as exc:
        print(f"    [clock] failed to invoke sudo nvidia-smi ({exc}); "
              "continuing without lock.", file=sys.stderr)
        return False

    if result.returncode != 0:
        print(f"    [clock] sudo nvidia-smi exited {result.returncode}; "
              "continuing without lock.", file=sys.stderr)
        return False

    print(f"    [clock] graphics clock locked to {clock} MHz (device {device}).")
    return True


def reset_gpu_clock(device: int = 0) -> bool:
    """Release a previously locked graphics clock. Best effort."""
    smi = _nvidia_smi()
    if smi is None:
        return False
    try:
        result = subprocess.run(["sudo", smi, f"--id={device}",
                                 "--reset-gpu-clocks"])
    except OSError:
        return False
    return result.returncode == 0
