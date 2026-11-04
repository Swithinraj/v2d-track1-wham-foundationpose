"""Fix the kit's latest-submitter/team identity comparison without changing uploads."""
from __future__ import annotations
import ast


def effective_submission_identities(info):
    identities=set()
    for entry in info.values():
        team=entry.get('team_name')
        user=entry.get('user')
        if team:
            identities.add(('team',team))
        elif user:
            identities.add(('user',user))
    return identities


def patch_preflight_source(source):
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight']
    if len(functions)!=1: raise ValueError('unrecognized upstream preflight')
    assignments=[n for n in ast.walk(functions[0]) if isinstance(n,ast.Assign) and
                 any(isinstance(t,ast.Name) and t.id=='identities' for t in n.targets)]
    if len(assignments)!=1 or not isinstance(assignments[0].value,ast.SetComp):
        raise ValueError('unrecognized upstream identity comparison')
    node=assignments[0]; old=ast.get_source_segment(source,node)
    expected='(entry.get("user"), entry.get("team_name"))'
    if expected not in old: raise ValueError('unrecognized upstream identity comparison')
    lines=source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=[' '*node.col_offset+'identities = effective_submission_identities(info)\n']
    patched=''.join(lines); ast.parse(patched)
    return patched
