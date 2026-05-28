"""Model-internals visualization (sibling of `nnx.vis_utils`).

`nnx.vis_utils` handles run-output viz (training curves, confusion
matrices, t-SNE of checkpoint logits). This subpackage handles the
model itself — parameter tables, weight distributions, activation
maps, and Netron-friendly ONNX graph export.

Both subpackages return Plotly `Figure` objects (or, in the case of
`summary`, the printable `torchinfo.ModelStatistics`) so callers can
compose them into dashboards or notebook layouts.
"""

from .activation import activation_map
from .netron import netron_export
from .summary import summary
from .weight_histogram import weight_histogram

__all__ = [
    "activation_map",
    "netron_export",
    "summary",
    "weight_histogram",
]
