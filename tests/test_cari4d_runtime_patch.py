import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent / 'fixtures/cari4d_selected_metrics.py'


def patched_selected_metric_values():
    assert importlib.util.find_spec('cari4d_runtime_patch') is not None, 'runtime patch is missing'
    from cari4d_runtime_patch import patch_foundationpose_export
    tree=ast.parse(patch_foundationpose_export(SOURCE.read_text()))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_selected_metric_values')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual patched upstream function>','exec'),namespace)
    return namespace['_selected_metric_values']


def metrics(observed=100, rendered=90, distance=.05):
    return SimpleNamespace(observed_visible_pixels=observed,rendered_visible_pixels=np.array([rendered]),
                           rendered_visibility=np.array([.9]),iou=np.array([.8]),precision=np.array([.85]),
                           recall=np.array([.9]),centroid_distance=np.array([distance]))


@pytest.mark.parametrize('observed,rendered',[(0,0),(0,20),(20,0)])
def test_undefined_visible_centroid_is_nan_not_infinite(observed,rendered):
    output=patched_selected_metric_values()(metrics(observed,rendered,np.inf),0)
    assert np.isnan(output['centroid_distance'])
    assert output['observed_visible_pixels']==observed
    assert output['rendered_visible_pixels']==rendered


def test_valid_centroid_diagnostic_is_preserved():
    output=patched_selected_metric_values()(metrics(),0)
    assert output['centroid_distance']==.05
    assert output['iou']==.8


def test_infinity_with_visible_support_is_not_hidden():
    output=patched_selected_metric_values()(metrics(distance=np.inf),0)
    assert np.isinf(output['centroid_distance'])


def test_patch_fails_on_unrecognized_upstream_version():
    patched_selected_metric_values()
    from cari4d_runtime_patch import patch_foundationpose_export
    with pytest.raises(ValueError,match='upstream'):
        patch_foundationpose_export('def different_function(): return 0\n')


def test_manifest_hashes_exact_input_bytes_including_crlf(tmp_path):
    original=SOURCE.read_bytes().replace(b'\n',b'\r\n')
    source=tmp_path/'source.py'; source.write_bytes(original)
    target=tmp_path/'patched.py'
    script=Path(__file__).resolve().parents[1]/'scripts/cari4d_runtime_patch.py'
    subprocess.run([sys.executable,str(script),'--source',str(source),'--out',str(target)],check=True,capture_output=True)
    manifest=json.loads(target.with_suffix('.patch.json').read_text())
    assert manifest['source_sha256']==hashlib.sha256(original).hexdigest()
    assert manifest['patched_sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
