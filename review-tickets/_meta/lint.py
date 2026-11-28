"""Lint ticket files in out/ against style.md and plan.md. Usage: python3 lint.py [--quiet]"""
import json, re, sys, pathlib

BASE = pathlib.Path(__file__).parent
OUT = BASE / "out"
plan = (BASE / "plan.md").read_text()
PLAN_IDS = re.findall(r"^\| ((?:DEC|FIX|FEAT)-\d+) \|", plan, re.M)
PLAN_TITLES = {m.group(1): m.group(2).strip() for m in re.finditer(r"^\| ((?:DEC|FIX|FEAT)-\d+) \| ([^|]+) \|", plan, re.M)}
AREAS = {"configuration", "data", "decisions", "documentation", "evaluation", "execution", "fluent-api", "graph",
         "inference", "interoperability", "model-integration", "model-transforms", "numerics", "packaging", "peft",
         "persistence", "training"}
TYPES = {"bug", "enhancement", "documentation", "question"}
EXTRA = {"github_actions", "dependencies"}
SECTIONS = ["Summary", "Context", "Acceptance criteria", "Evidence", "Scope", "Dependencies"]
DEC_SECTIONS = ["Summary", "Context", "Options", "Acceptance criteria", "Evidence", "Scope", "Dependencies"]
HEDGES = re.compile(r"\b(optionally|consider|either\b.*\bor\b|pick one|choose between|if you prefer|maybe|perhaps)\b", re.I)
PERMALINK = re.compile(r"https://github\.com/thekaveh/NNx/blob/([0-9a-f]{7,40})/[^\s)#]+(?:#L\d+(?:-L\d+)?)?")


def parse(path):
    text = path.read_text()
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    if not m:
        return None, text
    fm = {}
    for line in m.group(1).splitlines():
        line = re.sub(r"\s+#.*$", "", line)
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        try:
            fm[k.strip()] = json.loads(v) if v[:1] in "[\"{" or v in ("true", "false") else v
        except json.JSONDecodeError:
            fm[k.strip()] = v
    return fm, m.group(2)


def section(body, name):
    m = re.search(rf"^## {re.escape(name)}\s*$(.*?)(?=^## |\Z)", body, re.M | re.S)
    return m.group(1) if m else None


def main():
    quiet = "--quiet" in sys.argv
    files = sorted(OUT.glob("*.md"))
    seen = {}
    problems = {}
    for f in files:
        p = problems.setdefault(f.stem, [])
        fm, body = parse(f)
        if fm is None:
            p.append("no front matter"); continue
        tid = fm.get("id")
        seen[tid] = fm
        if tid != f.stem: p.append(f"id {tid} != filename")
        if tid not in PLAN_IDS: p.append("id not in plan")
        title = fm.get("title", "")
        if not title.startswith(f"[{tid}] "): p.append("title prefix")
        elif PLAN_TITLES.get(tid) and title[len(tid) + 3:] != PLAN_TITLES[tid]: p.append("title differs from plan")
        labels = fm.get("labels") or []
        if not labels or labels[0] not in TYPES: p.append(f"first label not a type: {labels[:1]}")
        areas = [l for l in labels if l.startswith("area: ")]
        if not 1 <= len(areas) <= 2 or any(a[6:] not in AREAS for a in areas): p.append(f"areas {areas}")
        if sum(l.startswith("priority: P") for l in labels) != 1: p.append("priority label")
        if "review: 2026-11-27" not in labels: p.append("review label")
        bad = [l for l in labels if l not in TYPES | EXTRA and not l.startswith(("area: ", "priority: ", "review: "))]
        if bad: p.append(f"unknown labels {bad}")
        status = fm.get("status")
        is_dec = tid.startswith("DEC-")
        if status not in ("READY", "DECISION", "DROP"): p.append(f"status {status}")
        if is_dec and status != "DECISION": p.append("DEC status must be DECISION")
        first = body.lstrip().splitlines()[0] if body.strip() else ""
        if not first.startswith("**Category:**") or "**Priority:**" not in first or "**Size:**" not in first:
            p.append("header line")
        want = DEC_SECTIONS if is_dec else SECTIONS
        heads = re.findall(r"^## (.+?)\s*$", body, re.M)
        if heads != want: p.append(f"sections {heads}")
        summ = section(body, "Summary") or ""
        if not summ.strip().startswith(f"{tid}:"): p.append("summary must start with ID:")
        ctx = section(body, "Context") or ""
        if "**Background for the worker.**" not in ctx and status != "DROP": p.append("no Background")
        if "**Worker setup.**" not in body and status != "DROP": p.append("no Worker setup")
        if not is_dec and status == "READY" and "**Proposal.**" not in ctx: p.append("no Proposal")
        ac = section(body, "Acceptance criteria") or ""
        boxes = [l for l in ac.splitlines() if l.strip().startswith("- [ ]")]
        if status != "DROP" and not boxes: p.append("no checkboxes")
        nocheck = [l for l in boxes if "Check:" not in l and "records the chosen option" not in l and not re.search(r"`(uv run|python3? |git |grep |test )", l)]
        if nocheck: p.append(f"{len(nocheck)} AC without Check:")
        if not is_dec:
            prop = ctx.split("**Proposal.**", 1)[1] if "**Proposal.**" in ctx else ""
            for m in HEDGES.finditer(prop + ac):
                p.append(f"hedge word: '{m.group(0)[:40]}'")
        ev = section(body, "Evidence") or ""
        links = PERMALINK.findall(body)
        if status != "DROP" and not links: p.append("no permalinks")
        if any(sha != "9ad461e" and not "9ad461e".startswith(sha[:7]) for sha in links): p.append("permalink not at 9ad461e")
        for d in fm.get("depends_on") or []:
            if d not in PLAN_IDS: p.append(f"depends_on unknown {d}")
        if re.search(r"\b(Claude|Haiku|Opus|Sonnet|subagent|LLM-generated)\b", body): p.append("mentions tooling/model")
        if "/tmp/" in body or "/home/" in body: p.append("local path in body")
    missing = [i for i in PLAN_IDS if i not in seen]
    bad = {k: v for k, v in problems.items() if v}
    print(f"files {len(files)}/{len(PLAN_IDS)}; missing {len(missing)}; with problems {len(bad)}")
    if missing and not quiet: print("missing:", " ".join(missing))
    for k, v in sorted(bad.items()):
        print(f"  {k}: " + "; ".join(dict.fromkeys(v)))
    st = {}
    for fm in seen.values(): st[fm.get("status")] = st.get(fm.get("status"), 0) + 1
    print("status:", st)


if __name__ == "__main__":
    main()
