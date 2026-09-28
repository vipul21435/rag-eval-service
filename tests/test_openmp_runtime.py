"""``import ragsvc`` tolerates the duplicate OpenMP runtime that faiss-cpu and torch ship.

Both wheels bundle their own ``libomp`` on macOS; the second to load aborts
the process with ``OMP: Error #15`` unless ``KMP_DUPLICATE_LIB_OK`` is set.
The package sets it on import, before either library loads, and leaves an
explicit value alone. Checked in a fresh interpreter because the test
process imported ``ragsvc`` long before any test runs.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

PROBE = "import os, ragsvc; print(os.environ.get('KMP_DUPLICATE_LIB_OK'))"
# The real thing: FAISS first (as ragsvc.core.vector_store imports it), then torch.
LOAD_BOTH = "import ragsvc.core.vector_store, torch; print('both runtimes loaded')"


def run_python(code: str, **env_overrides: str) -> str:
    env = {name: value for name, value in os.environ.items() if name != "KMP_DUPLICATE_LIB_OK"}
    env.update(env_overrides)
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


def test_import_sets_kmp_duplicate_lib_ok_when_unset() -> None:
    assert run_python(PROBE) == "TRUE"


def test_import_keeps_an_explicit_kmp_duplicate_lib_ok() -> None:
    assert run_python(PROBE, KMP_DUPLICATE_LIB_OK="FALSE") == "FALSE"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the neural extra")
def test_faiss_then_torch_survive_in_one_process() -> None:
    assert run_python(LOAD_BOTH) == "both runtimes loaded"
