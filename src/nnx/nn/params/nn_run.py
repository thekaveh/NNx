from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

import pandas as pd
import yaml

from ..._metrics import _resolve_metric
from ..enum.checkpoints import Checkpoints
from ..params.nn_checkpoint import NNCheckpoint
from ..params.nn_iteration_data_point import NNIterationDataPoint
from ..params.nn_model_params import NNModelParams
from ..params.nn_params import NNParams
from ..params.nn_train_params import NNTrainParams

if TYPE_CHECKING:
    from ...trainer.params import NNTrainerParams


def _runs_root(root: Optional[str] = None) -> str:
    """Resolve the on-disk root for `runs/`. Defaults to `<cwd>/runs` so
    existing notebook callers (which pass nothing) keep their layout."""
    return os.path.join(root if root is not None else os.getcwd(), "runs")


def _validate_run_id(run_id: str) -> str:
    """Reject `run_id` values that would escape the `runs/` directory.

    `NNRun.load(id=...)` and `NNCheckpoint.load(run=...)` join the caller-
    supplied identifier directly into a path under `runs/`. Internal
    callers always pass the md5 hex of `NNRun.state()` (32 hex chars), so
    the happy-path identifier is harmless. But the API surface is public,
    and a value like `"../../etc/passwd"` would resolve `runs/<id>/run.yaml`
    to a filesystem location outside the runs root — a classic path-
    traversal escalation if NNx is running anywhere a sensitive file
    sits next to `runs/`. Defense-in-depth: validate at the boundary.

    Accepts: any non-empty string that is its own basename (no path
    separators, no `..`, no embedded null).

    Raises `ValueError` on any traversal-shaped input.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"run id must be a non-empty string; got {run_id!r}")
    # `os.path.basename` strips trailing separators and the prefix path; if
    # the result differs from the input, the input contained a separator.
    # The explicit ``..`` and null-byte checks catch the residual cases that
    # basename happens to leave alone (e.g., `..` itself is its own basename).
    if "/" in run_id or "\\" in run_id or "\x00" in run_id or run_id in (".", ".."):
        raise ValueError(f"run id must not contain path separators or `..`; got {run_id!r}")
    if os.path.basename(run_id) != run_id:
        raise ValueError(f"run id must be a single path component; got {run_id!r}")
    return run_id


def _atomic_write_text(path: str, content: str) -> None:
    """Write `content` to `path` atomically — fsync, rename. A
    KeyboardInterrupt during the rename either leaves the prior file
    intact OR the new file fully written; never a half-written file."""
    tmp = path + ".tmp"
    # Explicit utf-8 — the default text encoding varies by platform
    # locale (cp1252 on Windows pre-3.15 / PEP 686). yaml.safe_dump
    # output is ASCII-safe today, but pinning utf-8 here makes the
    # contract platform-independent if a future state() ever emits
    # non-ASCII (e.g., a user-supplied tokenizer path with unicode).
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # fsync isn't supported on every filesystem (e.g., some
            # network mounts). Atomic rename is still useful even
            # without the fsync guarantee.
            pass
    os.replace(tmp, path)


def _point_best(best_run_path: str, run_path: str) -> None:
    """Make `best_run_path` point at `run_path`. Uses a symlink where the
    platform supports it (POSIX and Windows-with-developer-mode); falls
    back to writing a `best/POINTER.txt` text file with the run path.
    Either way ``_read_best_pointer`` can recover the target run.

    Atomicity: the symlink is created at a temp name and os.replace'd
    over the old pointer, so a crash mid-repoint leaves either the old
    or the new pointer — never none (a missing pointer would make the
    next save claim `best` unconditionally). Concurrent savers from
    separate processes race check-then-act benignly (last writer wins);
    multi-process best tracking is out of scope.

    target_is_directory=True matters on Windows-with-developer-mode:
    without it a *file* symlink to a directory is created, which
    Windows cannot traverse — best tracking would silently break."""
    tmp_link = best_run_path + ".tmp"
    # The symlink target is the sibling run-dir BASENAME: a symlink
    # target resolves relative to the symlink's own directory (the runs
    # root), not the creator's cwd. A raw run_path target broke under a
    # relative root= (dangling from birth, so every save took the
    # repoint-unconditionally branch — best tracked the most RECENT run,
    # not the best); an absolute target would dangle on any repo move.
    target = os.path.basename(os.path.normpath(run_path))
    try:
        if os.path.lexists(tmp_link):
            os.remove(tmp_link)  # stale temp from a prior crash
        os.symlink(src=target, dst=tmp_link, target_is_directory=True)
        if os.path.lexists(best_run_path) and not os.path.islink(best_run_path):
            # Prior POINTER.txt fallback layout — clear the directory so
            # os.replace can land the symlink (one-time layout upgrade;
            # this transition window existed before and is unavoidable).
            import shutil

            shutil.rmtree(best_run_path)
        os.replace(tmp_link, best_run_path)
    except (OSError, NotImplementedError):
        if os.path.lexists(tmp_link):
            os.remove(tmp_link)
        # Windows without developer mode: write a pointer file instead.
        # Atomic write so a KeyboardInterrupt during the fallback can't
        # leave a half-written POINTER.txt that confuses _read_best_pointer.
        os.makedirs(best_run_path, exist_ok=True)
        _atomic_write_text(os.path.join(best_run_path, "POINTER.txt"), run_path)


def _read_best_pointer(best_run_path: str) -> Optional[str]:
    """Return the run id currently pointed to by `runs/best`, or None when
    nothing is pointed there yet. Supports both symlink and POINTER.txt
    fallback layouts."""
    if not os.path.lexists(best_run_path):
        return None
    if os.path.islink(best_run_path):
        # Resolve the symlink to a run path → the basename is the run id.
        target = os.readlink(best_run_path)
        return os.path.basename(target.rstrip(os.sep)) or None
    pointer_file = os.path.join(best_run_path, "POINTER.txt")
    if os.path.isfile(pointer_file):
        with open(pointer_file, encoding="utf-8") as f:
            target = f.read().strip()
        return os.path.basename(target.rstrip(os.sep)) or None
    return None


def _best_err(checkpoint: Optional[NNCheckpoint]) -> float:
    """Pull the comparable metric from a checkpoint via the shared
    val→train, error→loss fallback resolver (the same walk the
    schedulers and tqdm postfix use). Returns +inf for missing
    checkpoints or fully missing metrics so caller comparisons always
    prefer the *new* run when there's no prior signal.

    The loss fallback keeps BEST tracking alive for runs whose steps
    leave `.error` unset (custom trainer/GAN step functions — the
    shipped diffusion/SimCLR/DPO factories do populate `.error`) —
    previously every such run scored +inf, `inf < inf` is False, and
    `runs/best` stayed frozen on whichever run saved first.
    Disclosure: in a runs root mixing error-scored and loss-scored
    runs, the cross-run comparison is between unlike metrics
    (error ∈ [0,1] vs unbounded loss) — accepted, since the
    alternative was a best pointer paradigm runs could never claim."""
    if checkpoint is None:
        return float("inf")
    metric = _resolve_metric(checkpoint.idp.val_edp, checkpoint.idp.train_edp)
    return float("inf") if metric is None else metric


@dataclass(frozen=True, kw_only=True, slots=True)
class NNRun:
    net: NNParams
    train: NNTrainParams
    model: NNModelParams

    # Optional trainer-mode marker. Populated by Trainer.train(); None for
    # NNModel.train()-produced runs. state() omits it when None so existing
    # run.id hashes for NNModel runs are preserved exactly.
    trainer: Optional[NNTrainerParams] = field(default=None)

    _id: Optional[str] = field(repr=False, default=None)
    _state: Optional[dict] = field(repr=False, default=None)
    idps: Optional[list[NNIterationDataPoint]] = field(repr=False, default=None)

    def __str__(self):
        # Delegate to NNSchedulerParams.__str__ for the scheduler block —
        # it knows which fields apply to the configured `kind` (the
        # plateau-only `patience`/`cooldown`/`threshold` are misleading
        # for cosine/onecycle/linear-warmup schedulers).
        return (
            "{"
            f"loss={self.model.loss}"
            f", device={self.model.device}"
            f", net={self.model.net}"
            f", dims={self.net.dims}"
            f", dropout={self.net.dropout_prob}"
            f", activation={self.net.activation}"
            f", n_heads={self.net.n_heads}"
            f", n_epochs={self.train.n_epochs}"
            f", max_lr={self.train.optim.max_lr}"
            f", momentum={self.train.optim.momentum}"
            f", decay={self.train.optim.weight_decay}"
            f", scheduler={self.train.scheduler}"
            "}"
        )

    @property
    def id(self) -> str:
        return self._id

    def __post_init__(self):
        state = dict(model=self.model.state(), net=self.net.state(), train=self.train.state())
        # `trainer` is omitted when None so existing NNModel runs hash to
        # the same run.id as before this field existed. Same omit-when-
        # default pattern as NNTrainParams.seed / save_phase_checkpoints
        # and NNOptimParams.param_groups.
        if self.trainer is not None:
            state["trainer"] = self.trainer.state()

        id = hashlib.md5(str(state).encode("utf-8")).hexdigest()

        object.__setattr__(self, "_id", id)
        object.__setattr__(self, "_state", {"id": id, **state})

    def state(self) -> dict:
        return self._state

    def _repr_html_(self) -> str:
        """Jupyter rich-display: config table + per-epoch metric chart.

        Returns the HTML Jupyter will render when this :class:`NNRun`
        is the last expression in a cell. Outside Jupyter the same
        method can be called directly to grab the HTML string.

        The config table reflects the same state the run.id is hashed
        from. The metric chart plots train/val loss + error across
        epochs when ``self.idps`` is populated; otherwise the chart is
        omitted and only the config table appears.
        """
        config_html = self._render_config_table_html()
        # Empty list and None both collapse to "no chart" — only render
        # when self.idps actually has at least one idp to chart from.
        # The explicit `is not None and len()` form (vs truthy `if self.idps`)
        # narrows the type for static checkers and makes the empty-list
        # case visible to readers.
        chart_html = self._render_metric_chart_html() if self.idps is not None and len(self.idps) > 0 else ""
        return f'<div style="font-family: sans-serif;">{config_html}{chart_html}</div>'

    def _render_config_table_html(self) -> str:
        """HTML table of the canonical run config (subset of state())."""
        rows = [
            ("run.id", self.id),
            ("net", str(self.model.net)),
            ("device", str(self.model.device)),
            ("loss", str(self.model.loss)),
            ("input_dim → output_dim", f"{self.net.input_dim} → {self.net.output_dim}"),
            ("hidden_dims", str(self.net.hidden_dims)),
            ("dropout", str(self.net.dropout_prob)),
            ("activation", str(self.net.activation)),
            ("n_epochs", str(self.train.n_epochs)),
            ("optim", f"{self.train.optim.name} (max_lr={self.train.optim.max_lr})"),
        ]
        rows_html = "".join(
            f'<tr><td style="padding:2px 8px;font-weight:600;">{k}</td>'
            f'<td style="padding:2px 8px;font-family:monospace;">{v}</td></tr>'
            for k, v in rows
        )
        return f'<table style="border-collapse:collapse;border:1px solid #ddd;margin-bottom:8px;">{rows_html}</table>'

    def _render_metric_chart_html(self) -> str:
        """Plotly per-epoch metric chart embedded as HTML."""
        # Lazy-import plotly so test collection stays fast and
        # non-Jupyter callers who never trigger _repr_html_ don't pay
        # the import cost.
        # Group idps by epoch_idx. Each idp carries its epoch index
        # directly; the last idp per epoch may also carry val_edp.
        from collections import defaultdict

        import plotly.graph_objects as go

        epoch_buckets: dict[int, list] = defaultdict(list)
        for idp in self.idps:
            epoch_buckets[idp.epoch_idx].append(idp)

        epochs: list[int] = sorted(epoch_buckets.keys())
        if not epochs:
            return ""  # No data at all.

        train_losses: list[float] = []
        train_errs: list[float] = []
        val_losses: list[float] = []
        val_errs: list[float] = []

        for epoch_idx in epochs:
            idp_list = epoch_buckets[epoch_idx]
            losses = [idp.train_edp.loss for idp in idp_list if idp.train_edp.loss is not None]
            errs = [idp.train_edp.error for idp in idp_list if idp.train_edp.error is not None]
            train_losses.append(sum(losses) / len(losses) if losses else float("nan"))
            train_errs.append(sum(errs) / len(errs) if errs else float("nan"))
            # val_edp is set only on the last idp of each epoch (when a
            # val_loader was supplied). Use the last non-None val_edp found.
            val_idp = next((idp for idp in reversed(idp_list) if idp.val_edp is not None), None)
            val_losses.append(val_idp.val_edp.loss if val_idp and val_idp.val_edp.loss is not None else float("nan"))
            val_errs.append(val_idp.val_edp.error if val_idp and val_idp.val_edp.error is not None else float("nan"))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_losses, name="train_loss", mode="lines+markers"))
        # Only add val traces when at least one epoch had a val_edp.
        if any(v == v for v in val_losses):  # any non-NaN
            fig.add_trace(go.Scatter(x=epochs, y=val_losses, name="val_loss", mode="lines+markers"))
        if any(v == v for v in train_errs):
            fig.add_trace(go.Scatter(x=epochs, y=train_errs, name="train_err", mode="lines+markers", yaxis="y2"))
        if any(v == v for v in val_errs):
            fig.add_trace(go.Scatter(x=epochs, y=val_errs, name="val_err", mode="lines+markers", yaxis="y2"))
        fig.update_layout(
            title=f"NNRun {self.id[:8]}… — {len(epochs)} epoch{'s' if len(epochs) != 1 else ''}",
            xaxis_title="Epoch",
            yaxis=dict(title="Loss"),
            yaxis2=dict(title="Error", overlaying="y", side="right"),
            height=350,
            margin=dict(t=40, b=40, l=40, r=40),
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")

    def with_idps(self, value: list[NNIterationDataPoint]) -> NNRun:
        return replace(self, idps=value)

    def checkpoints(self, root: Optional[str] = None) -> list[Optional[NNCheckpoint]]:
        """Load this run's five phase checkpoints, in cadence order
        (FIRST, Q1, Q2, Q3, LAST). Entries are None when the tag was
        never written — e.g. runs trained with
        ``save_phase_checkpoints=False`` write only LAST and BEST.

        BEST is deliberately excluded: it duplicates whichever phase
        checkpoint won, so including it would double-count. Load it
        directly via ``NNCheckpoint.load(run=run.id,
        type=Checkpoints.BEST)``.
        """
        return [
            NNCheckpoint.load(run=self.id, type=Checkpoints.FIRST, root=root),
            NNCheckpoint.load(run=self.id, type=Checkpoints.Q1, root=root),
            NNCheckpoint.load(run=self.id, type=Checkpoints.Q2, root=root),
            NNCheckpoint.load(run=self.id, type=Checkpoints.Q3, root=root),
            NNCheckpoint.load(run=self.id, type=Checkpoints.LAST, root=root),
        ]

    def save(self, root: Optional[str] = None) -> NNRun:
        runs_root = _runs_root(root)
        run_path = os.path.join(runs_root, self.id)
        best_run_path = os.path.join(runs_root, "best")

        csv_path = os.path.join(run_path, "idps.csv")
        yaml_path = os.path.join(run_path, "run.yaml")
        metadata_path = os.path.join(run_path, "metadata.yaml")

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        # All three writes go through the atomic write helper so a
        # KeyboardInterrupt mid-save leaves either the old file or the
        # new file, never a half-written one. This is what makes the
        # per-epoch incremental save actually safe — without atomicity,
        # a partial write here corrupts the run.
        # safe_dump — never plain `yaml.dump`. NNRun.state() is a plain
        # dict of primitive types (the round-trip contract — see the
        # corresponding safe_load call in NNRun.load); using safe_dump
        # here makes that contract enforced at write-time too, so a
        # future state() change that smuggles in a non-primitive (e.g.
        # a torch.dtype or a numpy scalar) fails loudly here rather
        # than producing a run.yaml that fails to safe_load.
        # sort_keys=True is explicit so the on-disk YAML stays stable
        # across PyYAML versions (the default was False prior to 5.1
        # and True from 5.1 onward; we pin to the post-5.1 alphabetical
        # ordering so a future major-version bump can't silently shift
        # the file shape downstream tooling diffs against).
        _atomic_write_text(yaml_path, yaml.safe_dump(self.state(), sort_keys=True))

        # Env snapshot: written separately so it does NOT contribute to
        # run.id (which is md5(state())). Captures library/torch/python
        # versions + git commit so a run.yaml from six months ago is
        # debuggable even if the library has moved on.
        from ...seeding import env_snapshot

        _atomic_write_text(metadata_path, yaml.safe_dump(env_snapshot(), sort_keys=True))

        _atomic_write_text(
            csv_path,
            # `or []`: idps defaults to None on the dataclass; an empty
            # frame round-trips cleanly through read_csv on load.
            pd.json_normalize(data=[idp.state() for idp in (self.idps or [])]).to_csv(),
        )

        if not os.path.lexists(best_run_path) or not os.path.exists(best_run_path):
            # Either no symlink yet, or one dangling after a repo move — repoint.
            _point_best(best_run_path, run_path)
        else:
            # Resolve the current best target via the symlink OR pointer file —
            # `NNCheckpoint.load(run="best", ...)` only works under a symlink
            # layout, so on the Windows pointer-file fallback we go through
            # the run id explicitly to make a fair comparison.
            best_run_id = _read_best_pointer(best_run_path)
            best_ckpt = (
                NNCheckpoint.load(run=best_run_id, type=Checkpoints.BEST, root=root)
                if best_run_id is not None
                else None
            )
            best_err = _best_err(best_ckpt)
            curr_err = _best_err(NNCheckpoint.load(run=self.id, type=Checkpoints.BEST, root=root))

            if curr_err < best_err:
                _point_best(best_run_path, run_path)

        return self

    @staticmethod
    def load(id: str, root: Optional[str] = None) -> NNRun:
        # Reject path-traversal identifiers before joining; see
        # `_validate_run_id` for the threat model. Internal callers pass
        # md5 hex (always safe), so this is a defense-in-depth guard on
        # the public API surface.
        id = _validate_run_id(id)
        run_path = os.path.join(_runs_root(root), id)
        yaml_path = os.path.join(run_path, "run.yaml")
        csv_path = os.path.join(run_path, "idps.csv")

        with open(yaml_path, encoding="utf-8") as f:
            # safe_load — never FullLoader. NNRun.state() is a plain dict
            # of primitive types (strings / ints / floats / lists / nested
            # dicts), so safe_load round-trips it losslessly. Using
            # FullLoader would let a tampered or attacker-supplied
            # run.yaml instantiate arbitrary Python objects via the
            # `!!python/object/...` tag — defense-in-depth even though
            # the file is normally application-written. Matches the
            # safe_dump/safe_load pair the sibling metadata.yaml has
            # always used (see seeding.py's "yaml.safe_load-compatible"
            # comment for the metadata round-trip).
            rep = yaml.safe_load(f)
        if not isinstance(rep, dict):
            # An empty / truncated-to-zero file safe_loads to None and
            # would otherwise die on rep.get(...) with no file context.
            raise ValueError(f"malformed run.yaml at {yaml_path}: expected a mapping, got {type(rep).__name__}")

        try:
            raw_idps = pd.read_csv(csv_path).to_dict(orient="records")
        except pd.errors.EmptyDataError as e:
            # A zero-byte file (external truncation — our own atomic
            # writes never produce one; even an idps-less run writes the
            # frame header) raises a pandas error with no file context.
            raise ValueError(f"malformed idps.csv at {csv_path}: {e}") from e
        # Separate try-scopes per source file so a missing key names the
        # FILE that's actually corrupt (a dropped CSV column must not be
        # reported as a run.yaml problem, and vice versa).
        try:
            idps = [NNIterationDataPoint.from_state(idp) for idp in raw_idps]
        except KeyError as e:
            raise ValueError(f"malformed idps.csv at {csv_path}: missing column/key {e}") from e

        try:
            # Lazy import for trainer params — keeps `nnx.nn.params`
            # importable without dragging the trainer subpackage in, and
            # avoids a cycle if anything in `nnx.trainer` ever needs to
            # import NNRun.
            trainer_state = rep.get("trainer")
            if trainer_state is not None:
                from ...trainer.params import NNTrainerParams

                trainer = NNTrainerParams.from_state(trainer_state)
            else:
                trainer = None

            return NNRun(
                # resolve_from_state: a TRANSFORMER run's net params must
                # come back as NNTransformerParams, not be downgraded to
                # NNParams.
                net=NNParams.resolve_from_state(rep["net"]),
                train=NNTrainParams.from_state(rep["train"]),
                model=NNModelParams.from_state(rep["model"]),
                trainer=trainer,
                idps=idps,
            )
        except KeyError as e:
            # A hand-edited / truncated run.yaml otherwise surfaces as a
            # bare KeyError with no file context.
            raise ValueError(f"malformed run.yaml at {yaml_path}: missing key {e}") from e

    @staticmethod
    def all(root: Optional[str] = None) -> list[NNRun]:
        """List every saved NNRun under the runs root, skipping the `best`
        pointer. Returns [] when the runs/ directory doesn't exist yet.
        Non-directory entries (stray files, .DS_Store) are filtered out
        so they don't trigger spurious NNRun.load failures."""
        runs_root = _runs_root(root)
        if not os.path.isdir(runs_root):
            return []
        result: list[NNRun] = []
        for entry in os.listdir(runs_root):
            if entry == "best":
                continue
            if not os.path.isdir(os.path.join(runs_root, entry)):
                continue
            # Defensive: a directory in runs/ that lacks run.yaml isn't a
            # real run (could be a leftover from an aborted experiment).
            if not os.path.isfile(os.path.join(runs_root, entry, "run.yaml")):
                continue
            result.append(NNRun.load(id=entry, root=root))
        return result
