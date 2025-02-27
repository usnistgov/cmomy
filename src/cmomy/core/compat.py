"""
Version compatibility

Mostly used for numpy2.0 compatibility...


"""

# pylint: disable=no-name-in-module,useless-suppression
from __future__ import annotations

import numpy as np

_IS_NUMPY_2 = np.lib.NumpyVersion(np.__version__) >= "2.0.0"

_COPY_IF_NEEDED = None if _IS_NUMPY_2 else False

if _IS_NUMPY_2:
    from numpy.lib.array_utils import (  # type: ignore[import-not-found,unused-ignore]
        normalize_axis_index as np_normalize_axis_index,
    )
    from numpy.lib.array_utils import (  # type: ignore[import-not-found,unused-ignore]
        normalize_axis_tuple as np_normalize_axis_tuple,
    )
else:
    from numpy.core.multiarray import (  # type: ignore[attr-defined,no-redef,unused-ignore]
        normalize_axis_index as np_normalize_axis_index,
    )
    from numpy.core.numeric import (  # type: ignore[attr-defined,no-redef,unused-ignore]
        normalize_axis_tuple as np_normalize_axis_tuple,
    )


def copy_if_needed(
    copy: bool | None,
) -> bool:  # Lie here so can support both versions...
    """Callable to return copy if needed convention..."""
    if not copy:
        return _COPY_IF_NEEDED  # type: ignore[return-value]
    return copy


__all__ = ["copy_if_needed", "np_normalize_axis_index", "np_normalize_axis_tuple"]
