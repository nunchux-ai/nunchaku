"""
Pytest fixture hooks for the NVFP4 kernel test suite.

The actual oracle and decoder helpers live in
[`tests.kernels._helpers`](packages/nunchaku/tests/kernels/_helpers.py:1)
- this file only contains shared fixtures that pytest auto-loads.

Tests import the helpers explicitly via::

    from tests.kernels._helpers import (
        fp4_decode_ascales, nvfp4_reference, ...,
    )
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Tests import shared helpers as `import _helpers as h` (a sibling module).
# pytest already adds the directory containing each test file to sys.path
# when collecting tests from a package-less directory, but to be robust
# when invoked as `pytest packages/nunchaku/tests/kernels/`, also add the
# directory containing this conftest. We append (not prepend) so that any
# already-installed top-level package wins over local source directories
# (specifically, the source `nunchaku/` checkout in `packages/nunchaku/`
# must not shadow the compiled, pip-installed `nunchaku` package that
# carries the kernel bindings we are testing).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

# Aggressively remove any sys.path entry that contains an un-compiled
# `nunchaku/` source directory (i.e. one without a `_C.*.so`) so that
# `import nunchaku` resolves to the installed wheel and `nunchaku._C` is
# available. This makes the tests robust to being invoked from a workspace
# checkout that has a local `nunchaku/` source tree.
def _strip_uncompiled_nunchaku_from_path() -> None:
    keep: list[str] = []
    for p in sys.path:
        cand = os.path.join(p, "nunchaku")
        if os.path.isdir(cand):
            has_compiled = any(
                f.startswith("_C") and (f.endswith(".so") or f.endswith(".pyd"))
                for f in os.listdir(cand)
            )
            if not has_compiled:
                # Skip this path entry; it would shadow the installed package.
                continue
        keep.append(p)
    sys.path[:] = keep


_strip_uncompiled_nunchaku_from_path()


def _force_installed_nunchaku_first() -> None:
    """
    Find the installed `nunchaku` package via importlib (which uses
    site-packages), and prepend its parent directory to sys.path so it
    *always* wins over any source-tree `nunchaku/` directory that may be
    added later by pytest's rootdir-handling.
    """
    import importlib.util
    spec = importlib.util.find_spec("nunchaku")
    if spec is not None and spec.origin:
        # spec.origin = ".../site-packages/nunchaku/__init__.py"
        # parent dir = ".../site-packages"
        site_packages_dir = os.path.dirname(os.path.dirname(spec.origin))
        if site_packages_dir in sys.path:
            sys.path.remove(site_packages_dir)
        sys.path.insert(0, site_packages_dir)


# Strip first, then prepend, then re-strip to guard against any caller
# that already imported the source `nunchaku` package half-way.
_force_installed_nunchaku_first()


# Evict any cached half-imported `nunchaku` modules so the next
# `import nunchaku` re-resolves through the clean sys.path.
def _evict_nunchaku() -> None:
    for mod in [m for m in list(sys.modules) if m == "nunchaku" or m.startswith("nunchaku.")]:
        del sys.modules[mod]


_evict_nunchaku()


# pytest's rootdir auto-discovery adds the rootdir to sys.path AFTER our
# conftest has been loaded; re-apply our path repair from session-start
# hooks to defend against the late insertion.
def pytest_sessionstart(session):
    _strip_uncompiled_nunchaku_from_path()
    _force_installed_nunchaku_first()
    _evict_nunchaku()


def pytest_collection(session):
    pytest_sessionstart(session)


def pytest_collectstart(collector):
    _strip_uncompiled_nunchaku_from_path()
    _force_installed_nunchaku_first()



def pytest_configure(config):
    """Register custom marks used by the kernel test suite."""
    config.addinivalue_line(
        "markers",
        "slow: tests that exercise diffusion-realistic GEMM shapes "
        "(e.g. M=4096 K=15360 N=3072); skip with `-m 'not slow'`.",
    )
    config.addinivalue_line(
        "markers",
        "nondeterministic: kernel known to reorder K-reductions across "
        "launches in a data-dependent way (per MATH_TEST_PLAN.md §0.4).",
    )


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """
    Force a deterministic seed for every test, per MATH_TEST_PLAN.md §0.4.
    Tests that need a non-default seed should call torch.manual_seed(...)
    themselves at the top of the test.
    """
    torch.manual_seed(0xC0FFEE)
    yield


@pytest.fixture
def cuda_device() -> torch.device:
    return torch.device("cuda")
