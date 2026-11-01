from __future__ import annotations

from typing import Any

from torch import nn


def restore_params_property(net: nn.Module, state: dict[str, Any]) -> None:
    """``nn.Module.__setstate__`` for a net whose ``params`` is a property
    that fills the fields its forward reads.

    A whole-module pickle (``torch.save(net)``, or a net inside a pickled
    ``NNModel``) stores the module's ``__dict__``. 0.3.x stored ``params`` as
    a plain attribute on every built-in net, as 0.4.x still did on ``ConvNN``
    and ``FeedFwdMoENN``, and those pickles lack some or all of the cached
    fields. Restored as stored, the property hid that ``params`` and the
    forward missed its fields: the net loaded but failed on first use. The
    stored params are set through the property instead, which refills every
    cached field, so the net forwards as the release that pickled it did and
    re-pickles in the current layout. A current pickle has no ``params`` key
    and restores unchanged.
    """
    state = dict(state)
    legacy = "params" in state
    params = state.pop("params", None)
    nn.Module.__setstate__(net, state)
    if legacy:
        net.params = params
