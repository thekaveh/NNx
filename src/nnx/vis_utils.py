"""Run-output visualization helpers.

`VisUtils` collects the Plotly-based visualizations for *run outputs* —
the artifacts produced after `NNModel.train()` has completed: training
curves, confusion matrices, classification reports, t-SNE projections
of held-out logits, etc. It is the sibling of `nnx.viz` (model-internals
visualization — weight histograms, activation maps, gradient flow,
Netron export); the two subpackages are deliberately independent and
do not share code.

Every method is a `@staticmethod` returning either a `plotly.graph_objects.Figure`
(for plots) or a `pandas.DataFrame` (for tables). The class itself
carries layout constants (`TITLE_SIZE`, `LABEL_SIZE`, `FIG_SIZE`,
`MARGIN_SIZE`) shared across methods, plus an opt-in `RENDERER`
override for environments where Plotly's default renderer doesn't
work (e.g., when serving from a headless container).

Convenience module-level aliases re-export the most common methods at
the bottom of this file so callers can write `nnx.vis_utils.confusion_matrix(...)`
instead of `nnx.vis_utils.VisUtils.confusion_matrix(...)`.
"""

from __future__ import annotations

import colorsys
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

from .nn.dataset.nn_dataset import NNDataset
from .nn.nn_model import NNModel
from .nn.params.nn_checkpoint import NNCheckpoint


class VisUtils:
    """Plotly-based visualizations for NNx run outputs.

    The static-method facade for run-output visualization. Companion to
    `nnx.viz` (model internals — weight / activation / gradient inspection)
    and `nnx.vis_utils`'s module-level aliases (which re-export the most
    common methods of this class for ergonomic top-level access).

    Class attributes are styling defaults shared by every plotting method:

    - `TITLE_SIZE` / `LABEL_SIZE`: Plotly title and axis-label font sizes.
    - `FIG_SIZE`: default `(width, height)` in pixels.
    - `MARGIN_SIZE`: default margin spec (passed to `Figure.update_layout`).
    - `RENDERER`: Plotly renderer override. `None` lets Plotly auto-detect
      (notebook / browser / etc.); set to a string like ``"png"`` or
      ``"browser"`` to pin behavior in non-default environments.

    See the README §1.2 "Visualization" bullet and `docs/concepts.md`
    for usage examples.
    """

    TITLE_SIZE = 14
    LABEL_SIZE = 12
    RENDERER = None
    FIG_SIZE = (1000, 600)
    MARGIN_SIZE = dict(l=15, r=15, t=30, b=15, pad=0)

    @staticmethod
    def generate_colors(n):
        """Return `n` distinct hex-color strings spaced evenly around the HSV hue circle.

        Uses 60% saturation and 95% brightness for legend readability. `endpoint=False`
        keeps the first and last hues maximally distinct — the hue circle wraps (0.0 ==
        1.0 == red), so a closed interval would hand the first and last class identical
        colors. Used by every other VisUtils method that draws categorical groups.
        """
        hues = np.linspace(0, 1, n, endpoint=False)
        rgb_colors = [colorsys.hsv_to_rgb(h, 0.6, 0.95) for h in hues]
        hex_colors = [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in rgb_colors]

        return hex_colors

    @staticmethod
    def multi_line_plot(
        x,
        yss,
        title,
        yss_legend,
        x_axis_label,
        y_axis_label,
        x_ticks_inc=20,
        label_size=LABEL_SIZE,
        title_size=TITLE_SIZE,
        fig_size: tuple = FIG_SIZE,
        margin_size=MARGIN_SIZE,
        renderer=RENDERER,
    ):
        """Render a multi-group line chart and return the Plotly Figure.

        Each group in `yss` is drawn with a distinct color; each line within
        a group uses a distinct dash style. `yss_legend` is a (group_labels,
        line_labels) tuple — group_labels name the colored groups (one per
        entry in `yss`), line_labels name the dash styles shared across
        groups. Both legends are added as no-trace markers so the legend
        reads cleanly; data traces carry "group (line)" hover names.

        Returns the Figure. If `renderer` is non-None, also calls
        `fig.show(renderer=renderer)` so notebook callers see the chart
        inline; pass `renderer=None` (the default) for headless usage.
        """
        if not yss:
            raise ValueError("multi_line_plot requires at least one series in `yss`")

        fig = make_subplots()

        ls = ["solid", "dash", "dot", "dashdot"]
        cs = VisUtils.generate_colors(n=len(yss))
        n_lines_per_series = len(yss[0])

        group_labels, line_labels = yss_legend
        for ys_idx, ys in enumerate(yss):
            for y_idx, y in enumerate(ys):
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        showlegend=False,
                        name=f"{group_labels[ys_idx]} ({line_labels[y_idx]})",
                        line=dict(width=2, color=cs[ys_idx], dash=ls[y_idx]),
                    )
                )

        for idx, linestyle in enumerate(ls[:n_lines_per_series]):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=line_labels[idx],
                    line=dict(width=2, dash=linestyle, color="black"),
                )
            )

        for idx, color in enumerate(cs[: len(yss)]):
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="lines", line=dict(width=2, color=color), name=group_labels[idx])
            )

        fig.update_layout(
            width=fig_size[0],
            height=fig_size[1],
            margin=margin_size,
            title=dict(text=title, x=0.5, font=dict(size=title_size)),
            yaxis=dict(title=dict(text=y_axis_label, font=dict(size=label_size))),
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99),
            xaxis=dict(
                title=dict(text=x_axis_label, font=dict(size=label_size)),
                tickmode="array",
                tickvals=list(range(0, len(x), x_ticks_inc)),
            ),
        )

        if renderer is not None:
            fig.show(renderer=renderer)
        return fig

    @staticmethod
    def scatter_plot(
        vm,
        renderer=RENDERER,
        fig_size: tuple = FIG_SIZE,
        label_size: int = LABEL_SIZE,
        title_size: int = TITLE_SIZE,
        margin_size=MARGIN_SIZE,
    ):
        """Render a colored scatter plot from a view-model dict and return
        the Plotly Figure.

        `vm` is the structure produced by `get_scatter_plot_vm`: title, xs/ys
        column views, plus a `ts` group axis carrying labels + colors per
        category. Honors `renderer` the same way as `multi_line_plot`.
        """
        fig = go.Figure()

        for t_idx, _ in enumerate(vm["ts"]["uni_vals"]):
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    x=vm["xs-ts"][t_idx],
                    y=vm["ys-ts"][t_idx],
                    name=vm["ts"]["labels"][t_idx],
                    marker=dict(size=3, color=vm["ts"]["colors"][t_idx]),
                )
            )

        fig.update_layout(
            width=fig_size[0],
            height=fig_size[1],
            margin=margin_size,
            title=dict(text=vm["title"], x=0.5, font=dict(size=title_size)),
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99),
            yaxis=dict(title=dict(text=vm["ys"]["label"], font=dict(size=label_size))),
            xaxis=dict(title=dict(text=vm["xs"]["label"], font=dict(size=label_size))),
        )

        if renderer is not None:
            fig.show(renderer=renderer)
        return fig

    @staticmethod
    def get_scatter_plot_vm(data, title, col_xs, label_xs, col_ys, label_ys, col_ts, labels_ts, colors_ts, uni_ts):
        """Build the view-model dict consumed by `scatter_plot`.

        Splits the input dataframe by the categorical column `col_ts`, attaches
        per-category colors and labels, and precomputes the per-category
        x/y slices so the plotting function stays simple.
        """
        vm = {
            "title": title,
            "xs": {"vals": data[col_xs], "label": label_xs},
            "ys": {"vals": data[col_ys], "label": label_ys},
            "ts": {"vals": data[col_ts], "uni_vals": uni_ts, "colors": colors_ts, "labels": labels_ts},
        }

        vm["xs-ts"] = [vm["xs"]["vals"][vm["ts"]["vals"] == t_val] for t_val in vm["ts"]["uni_vals"]]
        vm["ys-ts"] = [vm["ys"]["vals"][vm["ts"]["vals"] == t_val] for t_val in vm["ts"]["uni_vals"]]

        return vm

    @staticmethod
    def two_dim_tsne_checkpoint_logits(
        checkpoint: NNCheckpoint,
        ds: NNDataset,
        n_samples: int,
        random_state: int | None = 0,
        renderer: str | None = RENDERER,
        fig_size: tuple = FIG_SIZE,
        title_size: int = TITLE_SIZE,
        label_size: int = LABEL_SIZE,
        margin_size=MARGIN_SIZE,
    ):
        """Project the first `n_samples` test logits of `checkpoint` to 2D
        via t-SNE and render them colored by ground-truth class.

        Useful for eyeballing class separability of an intermediate
        checkpoint — pass the BEST checkpoint to see how well-trained the
        decision space ended up. Returns the Plotly Figure.
        """
        if n_samples < 2:
            raise ValueError("two_dim_tsne_checkpoint_logits requires n_samples >= 2")

        model = cast(Any, NNModel.from_checkpoint(checkpoint=checkpoint))

        ts = [t for t in range(ds.output_dim)]
        cs = VisUtils.generate_colors(n=ds.output_dim)

        if ds.test_loader is None:
            raise ValueError("two_dim_tsne_checkpoint_logits requires ds.test_loader")

        logits_chunks = []
        target_chunks = []
        remaining = n_samples
        test_loader = cast(Any, ds.test_loader)
        for test_batch in test_loader:
            test_X, test_Y = model.net.unpack_batch(test_batch)
            test_Y_hat = model.predict(X=test_X)
            take = min(remaining, len(test_Y_hat.logits))
            logits_chunks.append(np.asarray(test_Y_hat.logits)[:take])
            target_chunks.append(test_Y.detach().cpu().numpy()[:take])
            remaining -= take
            if remaining == 0:
                break

        logits = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, 0))
        targets = np.concatenate(target_chunks, axis=0) if target_chunks else np.empty((0,))
        if len(logits) < 2:
            raise ValueError("two_dim_tsne_checkpoint_logits requires at least 2 samples from ds.test_loader")

        return VisUtils.scatter_plot(
            renderer=renderer,
            title_size=title_size,
            label_size=label_size,
            fig_size=fig_size,
            margin_size=margin_size,
            vm=VisUtils.get_scatter_plot_vm(
                data=pd.concat(
                    axis=1,
                    objs=[
                        pd.DataFrame(
                            data=TSNE(
                                n_components=2,
                                perplexity=min(30, len(logits) - 1),
                                random_state=random_state,
                            ).fit_transform(X=pd.DataFrame(data=logits)),
                            columns=["tsne_1", "tsne_2"],
                        ),
                        pd.DataFrame(data=targets, columns=["target"]),
                    ],
                ),
                uni_ts=ts,
                colors_ts=cs,
                labels_ts=ts,
                col_xs="tsne_1",
                col_ys="tsne_2",
                col_ts="target",
                label_xs="tsne_1",
                label_ys="tsne_2",
                title=f"2D t-SNE of output logits of best checkpoint @ epoch={checkpoint.idp.epoch_idx}",
            ),
        )

    @staticmethod
    def confusion_matrix(
        Y_true,
        Y_pred,
        class_names=None,
        title: str = "Confusion matrix",
        normalize: bool = False,
    ):
        """Render a confusion matrix heatmap. Y_true and Y_pred are 1-D arrays
        of integer class labels. If `class_names` is provided, axis labels use
        the named classes; otherwise integer indices."""
        from sklearn.metrics import confusion_matrix as _sk_cm

        Y_true = np.asarray(Y_true)
        Y_pred = np.asarray(Y_pred)
        # labels=...: without it sklearn orders rows by the classes
        # PRESENT in the data, so an absent class silently shifts every
        # later row/column onto the wrong name.
        cm_labels = list(range(len(class_names))) if class_names is not None else None
        cm = _sk_cm(Y_true, Y_pred, labels=cm_labels)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm = cm / row_sums

        n_classes = cm.shape[0]
        labels = class_names if class_names is not None else list(range(n_classes))

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm.round(3) if normalize else cm.astype(int),
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="True",
            width=VisUtils.FIG_SIZE[0],
            height=VisUtils.FIG_SIZE[1],
            margin=VisUtils.MARGIN_SIZE,
            yaxis=dict(autorange="reversed"),
        )
        if VisUtils.RENDERER is not None:
            fig.show(renderer=VisUtils.RENDERER)
        return fig

    @staticmethod
    def classification_report(Y_true, Y_pred, class_names=None) -> pd.DataFrame:
        """Per-class precision / recall / f1 / support as a DataFrame. Use the
        return value for tabular display (`print(df.to_string())` or notebook
        auto-display) or to feed back into downstream analysis."""
        from sklearn.metrics import classification_report as _sk_report

        target_names = [str(c) for c in class_names] if class_names is not None else None
        report = _sk_report(
            Y_true,
            Y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        return pd.DataFrame(report).transpose()


# Module-level aliases so callers can `from nnx.vis_utils import confusion_matrix`
# (preferred for new code) without giving up the `VisUtils.confusion_matrix`
# class API existing notebooks depend on. Each alias points at the same
# underlying function object as the class static method — there is no
# duplication of behavior.
generate_colors = VisUtils.generate_colors
multi_line_plot = VisUtils.multi_line_plot
scatter_plot = VisUtils.scatter_plot
get_scatter_plot_vm = VisUtils.get_scatter_plot_vm
two_dim_tsne_checkpoint_logits = VisUtils.two_dim_tsne_checkpoint_logits
confusion_matrix = VisUtils.confusion_matrix
classification_report = VisUtils.classification_report

__all__ = [
    "VisUtils",
    "generate_colors",
    "multi_line_plot",
    "scatter_plot",
    "get_scatter_plot_vm",
    "two_dim_tsne_checkpoint_logits",
    "confusion_matrix",
    "classification_report",
]
