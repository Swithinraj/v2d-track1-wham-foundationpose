#!/usr/bin/env python3
"""Lay out the Track 1 submission container's build context on the cluster, checking everything first.

The organizers compare the container with the file Kaggle scored (CONTAINER_SUBMISSION.md in the kit). It holds one code
commit, so the file selected on the five boards must cite that commit. Before copying anything this checks that

- the receipt holds all five boards, scored, and names this file,
- the file cites one pushed commit of this repository, and that commit holds a finished pipeline/recipes/final_recipe.sh,
- the stored episodes re-pack to exactly this file, with the kit copy that packed it,
- the images, weights and sources the pipeline loads are present,

then writes <out>/context/ for docker/Dockerfile.submission (layout in that file). Run with the submission venv from the
repository root:

    .venvs/submission/bin/python docker/collect_container.py --candidate NAME --receipt RECEIPT \\
        --kit /path/to/v2d_submission_kit_official.zip

--dry-run tests the layout without a receipt, images or weights. Its image can run `verify`, not `reproduce`.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TEAM = "Izus_systems"
LABEL = "track1"
METRICS = ("cd_h_cm", "cd_o_cm", "acc_h_cm", "acc_o_cm", "interpenetration_cm")
RECIPE = "pipeline/recipes/final_recipe.sh"
MARKER = "FINAL_RECIPE_NOT_WRITTEN"
ASSETS = ("containers", "weights", "third_party", "inputs")
SIFS = ("mhr_converter.sif", "v2d_cari4d.sif", "v2d_grounding_dino.sif", "v2d_sam2.sif", "v2d_sam3d.sif", "v2d_sam3d_body.sif")
COMMIT_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/commit/([0-9a-f]{40})$")


def receipt_errors(receipt: dict, file_name: str) -> list[str]:
    """What stops this receipt from standing behind the file; the verifier reads the IDs from it."""
    errors = []
    if receipt.get("track") != "track_1":
        errors.append(f"receipt track is {receipt.get('track')!r}")
    if Path(receipt.get("file", "")).name != file_name:
        errors.append(f"receipt is for {receipt.get('file')!r}, not {file_name}")
    boards = receipt.get("submissions", {})
    for metric in METRICS:
        entry = boards.get(metric)
        if entry is None:
            errors.append(f"receipt has no {metric} submission")
        elif entry.get("state") != "COMPLETE" or not entry.get("ref"):
            errors.append(f"{metric} is {entry.get('state')!r}, not scored: {entry.get('error', '')}")
    return errors


def cited_commit(table: pd.DataFrame) -> tuple[str, str]:
    urls = table["code_commit_url"].astype(str).unique()
    if len(urls) != 1:
        sys.exit(f"error: the file cites {len(urls)} commits; the container holds one")
    match = COMMIT_URL.match(urls[0])
    if not match:
        sys.exit(f"error: {urls[0]} is not a full-sha GitHub commit URL")
    return urls[0], match.group(1)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def commit_errors(url: str, sha: str, dry_run: bool) -> list[str]:
    errors = []
    if git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        return [f"{sha} is not in this clone; fetch it"]
    if not git("branch", "-r", "--contains", sha).stdout.strip():
        errors.append(f"{sha} is not pushed; organizers cannot inspect it")
    remote = re.sub(r"\.git$", "", git("remote", "get-url", "origin").stdout.strip()).replace("git@github.com:", "https://github.com/")
    if not url.startswith(remote + "/commit/"):
        errors.append(f"{url} is not a commit of {remote}")
    if not dry_run:
        recipe = git("show", f"{sha}:{RECIPE}")
        if recipe.returncode != 0 or MARKER in recipe.stdout:
            errors.append(f"{RECIPE} at {sha[:12]} is missing or unfinished; `run.sh reproduce` must work")
    return errors


def repack_errors(file: Path, episodes: Path, pack_kit: Path, url: str, table: pd.DataFrame) -> list[str]:
    """The verifier's own re-pack check, run here before anything large is copied."""
    verifier = pack_kit / "tools" / "verify_container.py"
    sys.path.insert(0, str(verifier.parent))
    import verify_container as vc  # kit code, so this is the comparison the organizers run

    report = vc.Report()
    vc.repack(report, LABEL, table, {"kit": str(pack_kit), "outputs": str(episodes)}, REPO, "track_1", url, None)
    return [f"{file.name}: {r['check']}: {r['detail']}" for r in report.results if r["status"] != "PASS"]


def kit_root(kit: Path, scratch: Path) -> Path:
    if kit.suffix == ".zip":
        with zipfile.ZipFile(kit) as archive:
            archive.extractall(scratch)
        kit = scratch
    found = [p.parents[1] for p in kit.rglob("tools/verify_container.py")]
    if len(found) != 1:
        sys.exit(f"error: {kit} is not the official kit (needs exactly one tools/verify_container.py)")
    return found[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, help="NAME of submissions/track1_cand_NAME.parquet")
    parser.add_argument("--receipt", type=Path, help="the upload's .kaggle-receipt.json")
    parser.add_argument("--kit", type=Path, required=True, help="the official kit (.zip or extracted), the verifier")
    parser.add_argument("--pack-kit", type=Path, default=REPO / "official" / "v2d_submission_kit",
                        help="the kit copy that packed the file")
    parser.add_argument("--file", type=Path, help="default submissions/track1_cand_NAME.parquet")
    parser.add_argument("--episodes", type=Path, help="default submissions/episodes_cand_NAME")
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "container_build")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    file = (args.file or REPO / "submissions" / f"track1_cand_{args.candidate}.parquet").resolve()
    episodes = (args.episodes or REPO / "submissions" / f"episodes_cand_{args.candidate}").resolve()
    pack_kit = args.pack_kit.resolve()
    if not file.is_file() or not episodes.is_dir():
        sys.exit(f"error: need {file} and {episodes}")
    table = pd.read_parquet(file)
    url, sha = cited_commit(table)

    errors = commit_errors(url, sha, args.dry_run)
    receipt = None
    if args.receipt:
        receipt = json.loads(args.receipt.read_text())
        errors += receipt_errors(receipt, file.name)
    elif not args.dry_run:
        errors.append("--receipt is required")
    if not args.dry_run:
        errors += [f"{REPO / name} is missing" for name in ASSETS if not (REPO / name).is_dir()]
        errors += [f"containers/{sif} is missing" for sif in SIFS if not (REPO / "containers" / sif).is_file()]
        if not (REPO / ".venvs" / "mediapipe" / "bin" / "python").is_file():
            errors.append(".venvs/mediapipe is missing")
    count = len(list(episodes.glob("episode_??????.npz")))
    if count != 30:
        errors.append(f"{episodes} has {count} episodes, not 30")
    if not errors:
        errors = repack_errors(file, episodes, pack_kit, url, table)
    if errors:
        print("\n".join(f"error: {e}" for e in errors), file=sys.stderr)
        return 1
    print(f"checked: {file.name} cites {sha[:12]}, its 30 episodes re-pack to it")

    context = args.out.resolve() / "context"
    if context.exists():
        sys.exit(f"error: {context} exists; remove it to rebuild")
    root = context / "submission"
    root.mkdir(parents=True)
    scratch = Path(tempfile.mkdtemp(prefix="v2d_kit_"))
    skip = shutil.ignore_patterns("__pycache__", ".venv", "*.pyc", "uploads", "*.csv", "*.parquet.*", "*.kaggle-receipt.json")
    shutil.copytree(pack_kit, root / "kit", ignore=skip)
    shutil.copytree(kit_root(args.kit.resolve(), scratch), root / "verifier_kit", ignore=skip)

    subprocess.run(["git", "clone", "--quiet", "--no-checkout", str(REPO), str(root / "code")], check=True)
    subprocess.run(["git", "-C", str(root / "code"), "checkout", "--quiet", sha], check=True)
    subprocess.run(["git", "-C", str(root / "code"), "remote", "set-url", "origin", url.split("/commit/")[0] + ".git"], check=True)

    (root / "submitted").mkdir()
    shutil.copyfile(file, root / "submitted" / file.name)
    if args.receipt:
        shutil.copyfile(args.receipt, root / "submitted" / args.receipt.name)
    shutil.copytree(episodes, root / "outputs" / LABEL / "reconstructions")
    if not args.dry_run:
        for name in ASSETS:
            shutil.copytree(REPO / name, root / name, symlinks=False,
                            ignore=shutil.ignore_patterns("apptainer-cache", "apptainer-tmp"))

    ids = {m: receipt["submissions"][m]["ref"] for m in METRICS} if receipt else {}
    manifest = {"schema": "v2d_container_v1", "team": TEAM, "code": {"path": "code", "commit_url": url}, "kit": "kit",
                "submissions": [{"label": LABEL, "track": "track_1", "file": f"submitted/{file.name}",
                                 "kaggle_submission_ids": ids, "outputs": f"outputs/{LABEL}/reconstructions"}]}
    (root / "submission.json").write_text(json.dumps(manifest, indent=2) + "\n")

    here = Path(__file__).parent
    shutil.copyfile(here / "run.sh", root / "run.sh")
    readme = (here / "SUBMISSION_README.md").read_text()
    (root / "README.md").write_text(readme.replace("@SHA@", sha[:12]).replace("@FILE@", file.name).replace("@URL@", url))
    shutil.copyfile(here / "Dockerfile.submission", context / "Dockerfile")
    shutil.copyfile(here / "apptainer_shim.sh", context / "apptainer_shim.sh")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout
    (context / "requirements.lock").write_text(freeze)
    mediapipe = REPO / ".venvs" / "mediapipe" / "bin" / "python"
    (context / "mediapipe.lock").write_text("" if args.dry_run else subprocess.run(
        [str(mediapipe), "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout)
    print(f"wrote {context}; next: DATASET=... bash docker/build_container.sh {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
