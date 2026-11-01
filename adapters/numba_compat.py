"""numba_compat.py
Lightweight compatibility layer for environments where numba's native C-extensions
are blocked by Windows Application Control / AppLocker / Smart App Control.
"""

from __future__ import annotations

import sys
import types


def patch_numba_if_needed() -> None:
    """Installs a minimal pure-Python shim for numba if importing native numba fails."""
    try:
        import numba  # noqa: F401
        from numba.experimental.jitclass import boxing  # noqa: F401
    except Exception:
        if "numba" in sys.modules and getattr(sys.modules["numba"], "_is_stub", False):
            return

        class NumbaStub(types.ModuleType):
            _is_stub = True
            uint32 = "uint32"
            int32 = "int32"
            int64 = "int64"
            float32 = "float32"
            float64 = "float64"
            boolean = "bool"
            prange = range

            @staticmethod
            def jit(*args, **kwargs):
                if len(args) == 1 and callable(args[0]):
                    return args[0]
                return lambda f: f

            @staticmethod
            def njit(*args, **kwargs):
                if len(args) == 1 and callable(args[0]):
                    return args[0]
                return lambda f: f

            @staticmethod
            def guvectorize(*args, **kwargs):
                return lambda f: f

            @staticmethod
            def vectorize(*args, **kwargs):
                return lambda f: f

            @staticmethod
            def stencil(*args, **kwargs):
                return lambda f: f

        stub = NumbaStub("numba")
        sys.modules["numba"] = stub


patch_numba_if_needed()
