---
id: FEAT-067
title: "[FEAT-067] Remove the inert noqa: BLE001 markers"
labels: ["enhancement", "area: packaging", "priority: P3", "review: 2026-11-27"]
sources: ["DEAD-14"]
depends_on: []
related_issues: [507]
status: READY
---

**Category:** Packaging · **Priority:** P3 · **Size:** S · **Evidence:** read at `9ad461e` (develop)

## Summary

FEAT-067: eleven `except Exception:` clauses in `src/nnx` carry `# noqa: BLE001`, but ruff's `select` does not enable the BLE rules, so the markers suppress nothing; other broad `except Exception` clauses carry none, so they do not even mark the deliberate ones consistently. Delete all eleven markers, keep the reason text where a marker had one, and add a test so none comes back while BLE stays disabled.

## Context

**What happens today.** `pyproject.toml` L237:

```toml
select = ["E", "F", "W", "B", "I", "UP"]
```

and, for example, `nn_model.py` L655 and `nn_run.py` L682:

```python
        except Exception:  # noqa: BLE001
```

```python
    except Exception:  # noqa: BLE001 — a scan over untrusted directories must skip, not abort
```

The eleven markers: `nn_model.py:655, 1506, 1518, 2644`; `nn_run.py:682`; `history.py:947, 1195, 1339`; `components.py:426`; `provenance.py:783`; `decisions/jev.py:229`.

**Why it matters.** A suppression comment for a rule that is off is noise and suggests a lint check that does not run.

**History.** #507 (FIX-040) added the "warn, but never let a `-W error` filter replace the propagating error" blocks where six of these markers sit. FEAT-083 (SMELL-05) later replaces those six blocks with one helper; it lands after this ticket and must not add any marker back.

**Background for the worker.**
- *`noqa: BLE001`*: a ruff suppression for flake8-blind-except's "do not catch blind `Exception`". It has an effect only when `BLE` is in `[tool.ruff.lint] select`.
- *Reason text*: four markers carry a short reason after the code (`— ...` or `- ...`). That reason documents why the broad except is deliberate and stays as a plain comment.

**Proposal.**

1. Edit each line below exactly as shown (old → new). Only the comment changes; the `except` clause stays.

   | File:line | New text of the line (indentation unchanged) |
   |---|---|
   | `src/nnx/nn/nn_model.py:655` | `except Exception:` |
   | `src/nnx/nn/nn_model.py:1506` | `except Exception:` |
   | `src/nnx/nn/nn_model.py:1518` | `except Exception:` |
   | `src/nnx/nn/nn_model.py:2644` | `except Exception:` |
   | `src/nnx/nn/params/nn_run.py:682` | `except Exception:  # a scan over untrusted directories must skip, not abort` |
   | `src/nnx/history.py:947` | `except Exception as exc:  # re-raised unless tolerant` |
   | `src/nnx/history.py:1195` | `except Exception:  # unreadable for any reason: the documented fallback applies` |
   | `src/nnx/history.py:1339` | `except Exception:  # a corrupt LAST: fall back to the child's first record` |
   | `src/nnx/components.py:426` | `except Exception:` |
   | `src/nnx/provenance.py:783` | `except Exception:` |
   | `src/nnx/decisions/jev.py:229` | `except Exception:  # a response built without HTTP metadata has no request id` |

2. Add this test to `tests/test_pypi_publication.py` (it already reads `pyproject.toml` through `_PYPROJECT` and imports `tomllib` and `pathlib`):
   ```python
   def test_no_blind_except_marker_while_the_rule_is_off():
       """FEAT-067: ruff does not select BLE, so a `noqa: BLE001` in the package
       suppresses nothing; none may be added until BLE is enabled."""
       select = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["ruff"]["lint"]["select"]
       if "BLE" in select:
           pytest.skip("BLE is enabled: the markers are meaningful")
       package = pathlib.Path(nnx.__file__).parent
       marked = [
           f"{path.relative_to(package)}:{number}"
           for path in package.rglob("*.py")
           for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
           if "BLE001" in line
       ]
       assert marked == []
   ```

No behaviour changes, so no CHANGELOG entry.

**Worker setup.** No torch needed for the edit itself; `uv sync --frozen --all-extras` installs what `tests/test_pypi_publication.py` imports.

## Acceptance criteria

- [ ] The new test `tests/test_pypi_publication.py::test_no_blind_except_marker_while_the_rule_is_off` fails on `develop` (it lists the eleven lines) and passes after the change. Check: `uv run pytest tests/test_pypi_publication.py::test_no_blind_except_marker_while_the_rule_is_off`
- [ ] No marker is left in the package. Check: `grep -rn "BLE001" src/nnx` prints nothing
- [ ] The four reason texts survive as plain comments. Check: `grep -c "must skip, not abort\|re-raised unless tolerant\|the documented fallback applies\|fall back to the child's first record\|has no request id" src/nnx/nn/params/nn_run.py src/nnx/history.py src/nnx/decisions/jev.py` totals 5 lines
- [ ] `uv run ruff check src/ tests/ examples/ scripts/` and `uv run ruff format --check src/ tests/ examples/ scripts/` pass. Check: both commands exit 0

## Evidence

| Location | What it shows |
|---|---|
| [pyproject.toml L236-L237](https://github.com/thekaveh/NNx/blob/9ad461e/pyproject.toml#L236-L237) | `select` without `BLE` |
| [nn_model.py L655](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L655), [L1506](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1506), [L1518](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L1518), [L2644](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/nn_model.py#L2644) | Four bare markers |
| [nn_run.py L682](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/nn/params/nn_run.py#L682) | Marker with a reason |
| [history.py L947](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L947), [L1195](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L1195), [L1339](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/history.py#L1339) | Markers with reasons |
| [components.py L426](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/components.py#L426), [provenance.py L783](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/provenance.py#L783) | Bare markers |
| [jev.py L229](https://github.com/thekaveh/NNx/blob/9ad461e/src/nnx/decisions/jev.py#L229) | Marker with a reason |
| #507 | The warn-while-failing blocks six markers sit in |

## Scope

**In:** the eleven markers in `src/nnx`, one test.

**Out:**
- Enabling ruff's BLE rules and marking every deliberate broad except (a larger change; no ticket).
- The same inert markers in `tests/ddp_scenarios.py` (21) and `examples/11_tinystories_lm.py` (1); this ticket covers the package only.
- Sharing the warn-while-failing blocks: FEAT-083.

## Dependencies

**Blocked by:** none
**Blocks:** FEAT-083
**Related:** #507
