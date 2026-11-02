"""Cells for native Track 1 GroundingDINO and SAM 2 on a Colab A100."""

from pathlib import Path
from textwrap import dedent


def cells(root: Path) -> list[tuple[str, str]]:
    entries = [
        ("markdown", """
        ## Build actor and object masks first

        These stages use pinned GroundingDINO and SAM 2 sources and only their
        required model checkpoints. The Colab GPU remains disconnected during
        notebook preparation. When connected, run one episode and review its
        timing before running the other 29. Outputs persist in Drive.
        """),
        ("code", """
        # 5a. Fetch only the three pinned source trees needed for masks.
        import os
        import subprocess
        from pathlib import Path

        def fetch_pinned(name, url, commit, sparse=None):
            path = REPO / "third_party" / name
            if not (path / ".git").is_dir():
                path.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "init", "-q", str(path)], check=True)
                subprocess.run(["git", "-C", str(path), "remote", "add", "origin", url], check=True)
                if sparse:
                    subprocess.run(["git", "-C", str(path), "sparse-checkout", "init", "--cone"], check=True)
                    subprocess.run(["git", "-C", str(path), "sparse-checkout", "set", *sparse], check=True)
            current = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
            if current != commit:
                env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
                subprocess.run(["git", "-C", str(path), "-c", "remote.origin.promisor=true",
                                "fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", commit],
                               env=env, check=True)
                subprocess.run(["git", "-C", str(path), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
                               env=env, check=True)
            actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            assert actual == commit, (name, actual)
            print(name, actual[:12])
            return path

        V2D_SRC = fetch_pinned("video_to_data", "https://github.com/nvidia-isaac/video_to_data.git",
            "193382a184308f6800331f3db829f76dc4294cfe",
            ["reconstruction/modules/v2d_common", "reconstruction/modules/v2d_mv",
             "reconstruction/modules/v2d_sam2", "reconstruction/modules/v2d_grounding_dino"])
        DINO_SRC = fetch_pinned("GroundingDINO", "https://github.com/IDEA-Research/GroundingDINO.git",
            "856dde20aee659246248e20734ef9ba5214f5e44")
        SAM2_SRC = fetch_pinned("sam2", "https://github.com/facebookresearch/sam2.git",
            "2b90b9f5ceec907a1c18123530e92e794ad901a4")
        """),
        ("code", """
        # 5b. Create a masks venv using Colab's CUDA PyTorch; build only DINO's required extension.
        import os
        import subprocess
        import sys
        from pathlib import Path

        MASK_VENV = Path("/content/.venvs/masks")
        MASK_PY = MASK_VENV / "bin/python"
        if not MASK_PY.is_file():
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(MASK_VENV)], check=True)
        env = {**os.environ, "HF_HOME": str(REPO / ".cache/huggingface"),
               "SAM2_BUILD_CUDA": "0", "CUDA_HOME": "/usr/local/cuda",
               "TORCH_CUDA_ARCH_LIST": "8.0", "MAX_JOBS": "2",
               "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
        subprocess.run([str(MASK_PY), "-c",
            "import torch,torchvision,cv2,numpy,PIL; assert torch.cuda.is_available(); print('CUDA base:', torch.__version__, torchvision.__version__)"],
            env=env, check=True)
        subprocess.run([str(MASK_PY), "-m", "pip", "install", "--quiet", "--prefer-binary",
            "setuptools", "wheel", "transformers>=4.40,<5", "addict", "yapf==0.40.1", "timm",
            "supervision==0.22.0", "pycocotools", "hydra-core>=1.3.2", "iopath>=0.1.10",
            "eva-decord>=0.6.1,<0.7", "av", "imageio", "h5py", "scipy", "pydantic", "omegaconf"],
            env=env, check=True)
        modules = V2D_SRC / "reconstruction/modules"
        for source in (SAM2_SRC, DINO_SRC, modules / "v2d_common", modules / "v2d_mv",
                       modules / "v2d_grounding_dino/lib", modules / "v2d_sam2/lib"):
            subprocess.run([str(MASK_PY), "-m", "pip", "install", "--quiet", "--no-build-isolation",
                            "--no-deps", "-e", str(source)], env=env, check=True)
        subprocess.run([str(MASK_PY), "-c",
            "import groundingdino._C, sam2, decord; from v2d.sam2.lib.video_to_masks import video_to_masks; from v2d.grounding_dino.lib.video_to_object_bboxes import _get_model; print('Mask imports OK')"],
            env=env, check=True)
        """),
        ("code", """
        # 5c. Download only the official DINO and SAM 2 checkpoints and BERT base weights.
        import hashlib
        import os
        import shutil
        import subprocess
        import urllib.request
        from pathlib import Path

        dino_path = REPO / "weights/object/grounding_dino/groundingdino_swint_ogc.pth"
        dino_path.parent.mkdir(parents=True, exist_ok=True)
        if not dino_path.is_file() or dino_path.stat().st_size < 100_000_000:
            tmp = dino_path.with_suffix(".part")
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            with urllib.request.urlopen(url, timeout=120) as source, tmp.open("wb") as target:
                shutil.copyfileobj(source, target)
            assert tmp.stat().st_size > 100_000_000
            tmp.replace(dino_path)
        print("GroundingDINO checkpoint:", round(dino_path.stat().st_size / 1e6), "MB")

        sam_dir = REPO / "weights/sam2"
        sam_dir.mkdir(parents=True, exist_ok=True)
        sam_path = sam_dir / "sam2.1_hiera_large.pt"
        if not sam_path.is_file():
            code = ("from huggingface_hub import hf_hub_download; "
                    "hf_hub_download(repo_id='facebook/sam2.1-hiera-large', "
                    "filename='sam2.1_hiera_large.pt', "
                    "revision='665f8e2ad61cf5f53d65644ff27c8ee525124610', "
                    "local_dir='/content/v2d-track1-repo/weights/sam2')")
            subprocess.run([str(MASK_PY), "-c", code], env=env, check=True)
        with sam_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
        config_source = SAM2_SRC / "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        config_target = sam_dir / "configs/sam2.1/sam2.1_hiera_l.yaml"
        config_target.parent.mkdir(parents=True, exist_ok=True)
        if not config_target.is_file():
            shutil.copy2(config_source, config_target)
        subprocess.run([str(MASK_PY), "-c",
            "from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased'); print('BERT base cached')"],
            env=env, check=True)
        print("SAM 2 checkpoint SHA256 verified")
        """),
    ]

    runner_source = (root / "scripts/colab_mask_batch.py").read_text(encoding="utf-8")
    entries.extend([
        ("code", dedent("""
        # 5d. Write the resumable mask runner into the cloned repo, then time episode 0.
        import subprocess
        import time
        import json
        runner = REPO / "scripts/colab_mask_batch.py"
        runner.write_text(RUNNER_SOURCE, encoding="utf-8")
        started = time.monotonic()
        subprocess.run([str(MASK_PY), str(runner), "0"], cwd=REPO, env=env, check=True)
        mask_minutes = (time.monotonic() - started) / 60
        print(f"Episode 0 masks ready in {mask_minutes:.1f} minutes.")
        rows = [json.loads(line) for line in (REPO / "data/track_1/meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        frames = {int(row["episode_index"]): int(row["length"]) for row in rows}
        estimated_hours = mask_minutes * sum(frames.values()) / frames[0] / 60
        print(f"Rough 30-episode estimate: {estimated_hours:.1f} hours, "
              f"{5*estimated_hours:.0f} to {13*estimated_hours:.0f} A100 units. Review before continuing.")
        """).lstrip().replace("RUNNER_SOURCE", repr(runner_source), 1)),
        ("code", dedent("""
        # 5e. Resume the remaining 29 episodes. Completed masks are skipped.
        import subprocess
        subprocess.run([str(MASK_PY), str(REPO / "scripts/colab_mask_batch.py"),
                        *map(str, range(1, 30))], cwd=REPO, env=env, check=True)
        print("Mask batch finished; run the actor-mask gate next.")
        """).lstrip()),
    ])
    return [(kind, dedent(source).lstrip("\n")) for kind, source in entries]
