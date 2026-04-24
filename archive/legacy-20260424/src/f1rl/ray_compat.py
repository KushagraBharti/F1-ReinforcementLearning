"""Compatibility helpers for importing Ray on Windows hosts with broken WMI."""

from __future__ import annotations

import collections
import platform
import sys

_RAY_IMPORT_PREPARED = False


def prepare_ray_import() -> None:
    """Patch platform lookups that can hang inside Ray's Windows import path.

    Ray calls ``platform.system()`` during import, which can block in
    ``platform._wmi_query()`` on some Windows hosts. The project only needs a
    stable Windows identifier there, so we provide a minimal cached response.
    """

    global _RAY_IMPORT_PREPARED
    if _RAY_IMPORT_PREPARED:
        return

    if sys.platform == "win32":
        uname_result = collections.namedtuple(
            "uname_result", "system node release version machine processor"
        )
        platform.system = lambda: "Windows"  # type: ignore[assignment]
        platform.release = lambda: "Windows"  # type: ignore[assignment]
        platform.uname = lambda: uname_result(  # type: ignore[assignment]
            "Windows", "", "Windows", "", "", ""
        )

    _RAY_IMPORT_PREPARED = True
