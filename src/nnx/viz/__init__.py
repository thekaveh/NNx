"""Model-internals visualization (sibling of `nnx.vis_utils`).

`nnx.vis_utils` handles run-output viz (training curves, confusion
matrices, t-SNE of checkpoint logits). This subpackage handles the
model itself — parameter tables, weight distributions, and later
activation maps / Captum attribution / Netron export (SP-11).

Both subpackages return Plotly `Figure` objects (or, in the case of
`summary`, the printable `torchinfo.ModelStatistics`) so callers can
compose them into dashboards or notebook layouts.
"""

from .attribute import attribute
from .summary import summary
from .weight_histogram import weight_histogram

__all__ = ["attribute", "summary", "weight_histogram"]
