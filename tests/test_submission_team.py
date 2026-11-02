import ast
from pathlib import Path
from types import SimpleNamespace
import pytest


def preflight(info):
    from submission_team import effective_submission_identities
    identities=effective_submission_identities(info)
    if len(identities)>1: raise ValueError('different teams or solo identities')
    return identities


def test_teammates_on_the_same_team_are_one_submission_identity():
    info={'cdh':{'user':'swithinraj','team_name':'Izus_systems'},'cdo':{'user':'zaid3727','team_name':'Izus_systems'}}
    assert preflight(info)=={('team','Izus_systems')}


def test_different_teams_remain_blocked_even_if_submitter_matches():
    with pytest.raises(ValueError):
        preflight({'a':{'user':'same','team_name':'Team A'},'b':{'user':'same','team_name':'Team B'}})


def test_solo_users_remain_distinct():
    with pytest.raises(ValueError): preflight({'a':{'user':'one'},'b':{'user':'two'}})


def test_actual_official_preflight_accepts_same_team_but_keeps_entry_rejection():
    from submission_team import patch_preflight_source,effective_submission_identities
    source=(Path(__file__).resolve().parents[1]/'official/v2d_submission_kit/submit/v2d_submit.py').read_text()
    tree=ast.parse(patch_preflight_source(source))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight')
    env={'effective_submission_identities':effective_submission_identities,'MISSING':object(),
         'entry_status':lambda api,slug:True,'http_status':lambda error:None,'report_missing_entries':lambda *rows:None}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<patched official preflight>','exec'),env)
    class API:
        def competition_submissions(self,slug,page_size):
            return [SimpleNamespace(team_name='Izus_systems',submitted_by_ref='member-a' if slug=='a' else 'member-b')]
        def competition_get_submission_limits(self,slug): return SimpleNamespace(num_allowed_now=4,num_today=16)
    result=env['preflight'](API(),{'cdh':'a','cdo':'b'})
    assert result['cdh']['remaining']==4
    env['entry_status']=lambda api,slug:False
    with pytest.raises(SystemExit): env['preflight'](API(),{'cdh':'a','cdo':'b'})
