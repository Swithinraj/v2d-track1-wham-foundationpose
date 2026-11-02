"""Write the one-episode, read-only camera-source diagnostic notebook."""
from pathlib import Path
from textwrap import indent
import nbformat as nbf

root = Path(__file__).resolve().parents[3]
script = (root / "repo/scripts/compare_camera_source_episode.py").read_text()
original = nbf.read(root / "Untitled-1.ipynb", as_version=4)
n = nbf.v4.new_notebook()
n.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
n.cells = [
    nbf.v4.new_markdown_cell("""# Episode 9: centered MoGe vs measured camera at SAM 3D Body source

**Plan before GPU use.** This compares the same 415 competition frames for one successful common episode. Baseline lives under `v2d-track1/outputs`; Claude's measured-source run wrote to ephemeral `/content/track1_shadow/outputs`, so it may have disappeared when Colab was released. The inventory and recovery cells are CPU only. If the comparison is absent, prepare the native SAM 3D Body environment and run only episode 9 on a GPU, saving it to Drive. The comparison cell is CPU only.

Primary diagnostic: paired COCO-17 median pixel error against independent ViTPose-H tracks, on all frames and the official scored frames, confidence ≥0.5. A 15-frame block bootstrap gives a 95% interval for the paired mean difference. We also check first scored frame, identity reliability, camera K, frame counts, and optional refit error. Negative delta favors the measured-source run on this diagnostic. This is **not** a Kaggle/CD-H result; the public competition data has no 3D ground truth.

No additional competition data or model weights are downloaded by the diagnostic. The raw rerun, if needed, uses only the existing Track 1 episode 9 video, masks, depth and already downloaded weights. Keep GPU runtime disconnected for setup and after the one-episode run."""),
    nbf.v4.new_code_cell("""from pathlib import Path
import json, os, subprocess, sys
from google.colab import drive
if not Path('/content/drive/MyDrive').is_dir():
    drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive/v2d-track1')
BASE = DRIVE / 'outputs'
COMPARE = DRIVE / 'outputs_measured_source'
TAG = 'episode_000009'
required_base = [BASE/'sam3d_body'/TAG/f'raw_{TAG}.npz', BASE/'sam3d_body'/TAG/'prepare.npz',
                 BASE/'camera'/TAG/'intrinsics.json', BASE/'vitpose'/f'{TAG}.npz']
for p in required_base:
    print('BASE', p.exists(), p)
assert all(p.is_file() for p in required_base), 'Baseline inputs missing from Drive'
needed_compare = [COMPARE/'sam3d_body'/TAG/f'raw_{TAG}.npz',
                  COMPARE/'sam3d_body'/TAG/'prepare.npz', COMPARE/'camera'/TAG/'intrinsics.json']
for p in needed_compare:
    print('COMPARE', p.exists(), p)
READY = all(p.is_file() for p in needed_compare)
print('Comparison ready:', READY)
print('Legacy ephemeral comparison:', (Path('/content/track1_shadow/outputs/sam3d_body')/TAG/f'raw_{TAG}.npz').is_file())
"""),
    nbf.v4.new_code_cell("""# CPU only: preserve the old VM's episode 9 comparison if it survived reconnect.
import shutil
legacy = Path('/content/track1_shadow/outputs')
if not READY:
    pairs = [(legacy/'sam3d_body'/TAG/f'raw_{TAG}.npz', needed_compare[0]),
             (legacy/'sam3d_body'/TAG/'prepare.npz', needed_compare[1]),
             (legacy/'camera'/TAG/'intrinsics.json', needed_compare[2])]
    if all(src.is_file() for src,_ in pairs):
        for src,dst in pairs:
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
        READY = True
        print('Recovered complete comparison from old VM into Drive; GPU unnecessary')
    else:
        print('Legacy comparison incomplete; one-episode rerun will be needed')
"""),
    nbf.v4.new_code_cell("""# CPU only: restore the pinned source checkout; no dataset or weight download.
REPO = Path('/content/v2d-track1-repo')
if not (REPO/'.git').is_dir():
    subprocess.run(['git','clone','--branch','vpq-vitpose','--single-branch','--filter=blob:none',
                    'https://github.com/Swithinraj/v2d-track1-wham-foundationpose.git',str(REPO)],check=True)
subprocess.run(['git','-C',str(REPO),'checkout','2cd29f4'],check=True)
for name in ('data','weights','outputs','.cache'):
    link = REPO/name
    target = DRIVE/name
    if link.is_symlink():
        assert link.resolve() == target.resolve(), (link,target)
    elif link.exists():
        raise RuntimeError(f'Inspect existing non-link path: {link}')
    else:
        target.mkdir(parents=True,exist_ok=True)
        link.symlink_to(target,target_is_directory=True)
print('Pinned source ready:',REPO)
if not READY:
    source_inputs = [REPO/'data/track_1/videos/chunk-000/observation.images.exo_camera'/f'{TAG}.mp4',
                     BASE/'person'/TAG/'sam2_masks/0/000414.png',
                     BASE/'object'/TAG/'depth/000414.png',
                     BASE/'object'/TAG/'sam2_masks/1/000414.png',
                     REPO/'weights/sam3d_body/sam-3d-body-dinov3/model.ckpt',
                     REPO/'weights/sam3d_body/sam-3d-body-dinov3/assets/mhr_model.pt',
                     REPO/'weights/sam3d_body/torch_home/hub/facebookresearch_dinov3_main/hubconf.py']
    for p in source_inputs: print('RECOVERY INPUT',p.is_file(),p)
    assert all(p.is_file() for p in source_inputs), 'Recovery inputs incomplete; do not start GPU setup'
"""),
    nbf.v4.new_markdown_cell("""## If comparison is missing: one-episode GPU recovery

The next setup cells are embedded from the previously successful native setup, so this notebook can run in a fresh Colab VM. They fetch only pinned source code, create `/content/.venvs/sam3d_body`, install SAM 3D Body and its dependencies, and reuse model weights already on Drive. Building PyTorch3D needs the CUDA runtime and may take time. Attach A100 **only if** the CPU inventory above proves the comparison is missing and all recovery inputs exist. After setup, run the one-episode cell once, then disconnect A100. It writes directly to Drive and skips refit/hands, which the diagnostic does not require. If comparison is already present, skip all GPU cells."""),
    nbf.v4.new_code_cell("""# GPU cell: execute only if READY is False and after native dependency preflight.
if READY:
    print('Comparison already persisted; GPU rerun skipped.')
else:
    import shutil
    import numpy as np
    import torch
    assert torch.cuda.is_available(), 'Attach GPU for this one cell only'
    REPO = Path('/content/v2d-track1-repo')
    BODY_PY = Path('/content/.venvs/sam3d_body/bin/python')
    assert (REPO/'scripts/sam3d_body_video.py').is_file(), 'Repo source missing'
    assert BODY_PY.is_file(), 'Run native setup cells in this notebook first'
    env = {**os.environ, 'HF_HOME':str(REPO/'.cache/huggingface'),
           'TORCH_CUDA_ARCH_LIST':'8.0', 'MAX_JOBS':'2', 'PYOPENGL_PLATFORM':'egl',
           'PYTHONPATH':str(REPO/'third_party/video_to_data/reconstruction/modules/v2d_sam3d_body/lib'),
           'LD_LIBRARY_PATH':'/usr/local/lib/python3.13/dist-packages/torch/lib:'+os.environ.get('LD_LIBRARY_PATH','')}
    probe = subprocess.run([str(BODY_PY),'-c','import torch,cv2,sam_3d_body; assert torch.cuda.is_available()'],
                           env=env, capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr[-2000:]
    cam0 = json.loads((BASE/'camera'/TAG/'intrinsics.json').read_text())
    ids = json.loads((REPO/'config/camera/episode_cameras.json').read_text())
    rig = json.loads((REPO/'config/camera/rig_cameras.json').read_text())
    picked = ids[TAG]
    assert picked['corr'] >= .8 and picked['corr']-picked['corr_other'] >= .3
    cam0['cx'], cam0['cy'] = rig[picked['camera']]['cx'], rig[picked['camera']]['cy']
    shadow = Path('/content/track1_shadow')
    shadow.mkdir(exist_ok=True)
    for name in ('data','weights','scripts','official','config','third_party'):
        link = shadow/name
        if not link.exists(): link.symlink_to(REPO/name)
    (shadow/'outputs').mkdir(exist_ok=True)
    for name in ('person','object'):
        link = shadow/'outputs'/name
        if not link.exists(): link.symlink_to(BASE/name)
    for name in ('camera','sam3d_body','keypoints2d_sam'):
        link = shadow/'outputs'/name
        dest = COMPARE/name
        dest.mkdir(parents=True, exist_ok=True)
        if not link.exists(): link.symlink_to(dest)
    camera_file = COMPARE/'camera'/TAG/'intrinsics.json'
    camera_file.parent.mkdir(parents=True, exist_ok=True)
    camera_file.write_text(json.dumps(cam0,indent=2))
    track1 = Path('/track1')
    prior = track1.resolve() if track1.exists() or track1.is_symlink() else None
    if track1.is_symlink(): track1.unlink()
    elif track1.exists(): raise RuntimeError('/track1 is not a symlink; will not replace it')
    track1.symlink_to(shadow)
    try:
        for name in ('sam3d_body_video.py','sam3d_body_identity.py'):
            cmd = [str(BODY_PY),'-u',str(REPO/'scripts'/name),'--episodes','9']
            print('RUN', name, flush=True)
            subprocess.run(cmd,cwd=REPO,env=env,check=True)
    finally:
        track1.unlink()
        if prior is not None: track1.symlink_to(prior)
    READY = all(p.is_file() for p in needed_compare)
    assert READY, 'One-episode run did not produce complete comparison'
    print('PASS: comparison persisted on Drive; disconnect GPU now')
"""),
    nbf.v4.new_markdown_cell("""## Paired diagnostic on CPU

The next cell creates a small analysis virtual environment, runs the embedded analysis script, and saves a JSON report to Drive. It refuses to score if either run or its camera is missing or mismatched. Reconnect on CPU if the A100 was used above."""),
    nbf.v4.new_code_cell("""assert READY, 'Comparison missing. Inventory/recover episode 9 before scoring.'
import shutil, venv, zipfile
VENV = Path('/content/.venvs/camera_ab')
PY = VENV/'bin/python'
if VENV.exists() and (not PY.is_file() or subprocess.run([str(PY),'-c','import numpy,pandas,pyarrow'],capture_output=True).returncode != 0):
    shutil.rmtree(VENV)
if not PY.is_file():
    venv.EnvBuilder(system_site_packages=True,with_pip=False).create(VENV)
subprocess.run([str(PY),'-c','import numpy,pandas,pyarrow'],check=True)
REPO = Path('/content/v2d-track1-repo')
SAMPLE = REPO/'official/v2d_submission_kit/data/track_1_sample_submission.parquet'
if not SAMPLE.is_file():
    V2D_SRC = REPO/'third_party/video_to_data'
    ZIP = V2D_SRC/'docs/v2d_challenge/assets/v2d_submission_kit.zip'
    if not ZIP.is_file():
        subprocess.run(['git','-C',str(V2D_SRC),'sparse-checkout','add','docs/v2d_challenge/assets'],check=True)
    SAMPLE.parent.parent.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(REPO/'official')
assert SAMPLE.is_file()
SCRIPT = Path('/content/compare_camera_source_episode.py')
SCRIPT.write_text(ANALYSIS_SOURCE)
REPORT = DRIVE/'comparison_tests/episode_000009_camera_source_ab.json'
cmd = [str(PY),str(SCRIPT),'--baseline',str(BASE),'--comparison',str(COMPARE),
       '--vitpose',str(BASE/'vitpose'/f'{TAG}.npz'),
       '--sample',str(SAMPLE),
       '--camera-id',str(REPO/'config/camera/episode_cameras.json'),
       '--rig',str(REPO/'config/camera/rig_cameras.json'), '--episode','9','--out',str(REPORT)]
subprocess.run(cmd,check=True)
print('Saved:',REPORT)
"""),
]
fetch_source = """# GPU path setup, source code only; skips if episode 9 comparison was recovered.
if not READY:
    import os
    V2D_SRC = REPO/'third_party/video_to_data'
    PIN = '193382a184308f6800331f3db829f76dc4294cfe'
    if not (V2D_SRC/'.git').is_dir():
        V2D_SRC.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','init','-q',str(V2D_SRC)],check=True)
        subprocess.run(['git','-C',str(V2D_SRC),'remote','add','origin',
                        'https://github.com/nvidia-isaac/video_to_data.git'],check=True)
        subprocess.run(['git','-C',str(V2D_SRC),'sparse-checkout','init','--cone'],check=True)
        subprocess.run(['git','-C',str(V2D_SRC),'sparse-checkout','set',
                        'reconstruction/modules/v2d_common','reconstruction/modules/v2d_mv',
                        'reconstruction/modules/v2d_sam3d_body'],check=True)
    probe = subprocess.run(['git','-C',str(V2D_SRC),'rev-parse','HEAD'],text=True,
                           capture_output=True)
    got = probe.stdout.strip() if probe.returncode == 0 else ''
    if got != PIN:
        git_env = {**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'}
        subprocess.run(['git','-C',str(V2D_SRC),'-c','remote.origin.promisor=true',
                        'fetch','--quiet','--depth','1','--filter=blob:none','origin',PIN],
                       env=git_env,check=True)
        subprocess.run(['git','-C',str(V2D_SRC),'checkout','--quiet','--detach','FETCH_HEAD'],
                       env=git_env,check=True)
    assert subprocess.check_output(['git','-C',str(V2D_SRC),'rev-parse','HEAD'],text=True).strip() == PIN
    modules = V2D_SRC/'reconstruction/modules'
    assert (modules/'v2d_sam3d_body/lib').is_dir()
    print('Pinned SAM 3D Body source ready')
"""
def only_if_needed(source: str) -> str:
    return "if not READY:\n" + indent(source, "    ")

setup_cells = [nbf.v4.new_code_cell(fetch_source)]
for number in (20, 21, 22, 23):
    source = original.cells[number].source
    if number == 20:
        source += "\nassert r.returncode == 0, r.stderr[-2000:]\n"
    if number == 21:
        source += "\nassert r.returncode == 0, r.stderr[-3000:]\n"
    if number == 22:
        source += "\nassert r.returncode == 0, r.stderr[-3000:]\n"
    if number == 23:
        source += "\nassert r.returncode == 0, r.stderr[-3000:]\n"
    setup_cells.append(nbf.v4.new_code_cell(only_if_needed(source)))
n.cells[5:5] = setup_cells
# Embed analysis source as data in the preceding cell without relying on uncommitted repo changes.
n.cells[-1].source = "ANALYSIS_SOURCE = " + repr(script) + "\n" + n.cells[-1].source
out = root / "camera_source_ab_ep09.ipynb"
nbf.write(n, out)
print(out)
