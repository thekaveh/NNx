import re,pathlib,sys,collections
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from lint import parse, section
OUT=pathlib.Path(__file__).parent/"out" if (pathlib.Path(__file__).parent/"out").is_dir() else pathlib.Path(__file__).parent.parent
T={}
for f in sorted(OUT.glob("*.md")):
    if re.fullmatch(r"(?:DEC|FIX|FEAT)-\d{3}",f.stem):
        fm,body=parse(f); T[fm["id"]]=(fm,body)
def key(i): p,n=i.split("-"); return ({"FIX":0,"FEAT":1,"DEC":2}[p],int(n))
pri=lambda fm: next(l for l in fm["labels"] if l.startswith("priority"))[-2:]
size=lambda b: (re.search(r"\*\*Size:\*\* (\w)",b) or [None,"?"])[1]
dep={k:set(v[0].get("depends_on") or []) for k,v in T.items()}
# checks
bad=[]
for k,(fm,body) in T.items():
    d=section(body,"Dependencies") or ""; m=re.search(r"\*\*Blocked by:\*\*(.*)",d)
    ids=(set(re.findall(r"\b(?:DEC|FIX|FEAT)-\d{3}\b",m.group(1))) if m else set())-{"DEC-009"}
    if ids!=dep[k]: bad.append((k,sorted(dep[k]-ids),sorted(ids-dep[k])))
    for x in dep[k]:
        if x not in T: bad.append((k,"unknown",x))
order=[];done=set();rem=set(T)
while rem:
    ready=sorted((k for k in rem if dep[k]<=done),key=lambda k:(pri(T[k][0]),key(k)))
    if not ready: sys.exit(f"cycle among {sorted(rem)}")
    k=ready[0]; order.append(k); done.add(k); rem.remove(k)
pos={k:i+1 for i,k in enumerate(order)}
c=collections.Counter((k.split("-")[0],pri(T[k][0])) for k in T)
L=["# Review tickets: develop f486c85..9ad461e","",
f"{len(T)} tickets for the thekaveh/NNx board, in the board's own format (Summary, Context, Acceptance criteria, Evidence, Scope, Dependencies), each detailed enough for a small worker agent to implement without guessing.",
"",
"- **Written against:** develop `9ad461e`, then re-checked against develop `1402b49` (#859, merged 2026-11-28). Code that #859 changed is cited at `1402b49`.",
"- **Supersedes** the finding IDs in `../REVIEW-develop-2026-11-27.md`. Each ticket's `sources:` names the findings it covers.",
"- **Checked against all 196 existing board issues.** Each ticket's `related_issues:` lists the issues it builds on.","",
"| | P1 | P2 | P3 | Total |","|---|---:|---:|---:|---:|"]
for p in ("FIX","FEAT","DEC"):
    r=[c[(p,x)] for x in ("P1","P2","P3")]; L.append(f"| {p} | "+" | ".join(map(str,r))+f" | {sum(r)} |")
L.append("| **All** | "+" | ".join(str(sum(c[(p,x)] for p in ('FIX','FEAT','DEC'))) for x in ("P1","P2","P3"))+f" | {len(T)} |")
L+=["","- **READY:** start once the tickets in `Blocked by` have merged.",
"- **DECISION:** the owner records an option in a comment first; the worker steps for each option are in the ticket.",
"- **Order:** a work order that respects every dependency, highest priority first.",
"- **Labels:** every ticket also carries `review: 2026-11-27`.",
"- **Posting:** `python3 _meta/post_tickets.py` (dry run), then `--yes`. It needs `gh` logged in with write access.","",
"| Order | ID | Title | Pri | Size | Status | Blocked by | Sources | Board history |","|---:|---|---|---|---|---|---|---|---|"]
for k in sorted(T,key=key):
    fm,b=T[k]
    L.append(f"| {pos[k]} | [{k}]({k}.md) | {fm['title'][len(k)+3:]} | {pri(fm)} | {size(b)} | {fm['status']} | {', '.join(sorted(dep[k],key=key)) or '—'} | {', '.join(fm.get('sources') or []) or '—'} | {', '.join('#'+str(n) for n in (fm.get('related_issues') or [])) or '—'} |")
(OUT/"README.md").write_text("\n".join(L)+"\n")
print("tickets",len(T),"| dependency problems:",bad,"| first 12 in order:",order[:12])
