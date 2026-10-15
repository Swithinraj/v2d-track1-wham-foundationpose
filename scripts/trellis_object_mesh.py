#!/usr/bin/env python3
"""Generate coarse metric-scale mesh candidates for one Track 1 object with TRELLIS image-to-3D.

Runs inside containers/v2d_sam3d.sif with the workspace mounted at /track1. The public
challenge release has no object meshes, so this follows the "object mesh unknown" branch
of the baseline plan: pick a clean, unoccluded view of the object in each of its three
episodes from the SAM 2 masks, generate meshes with TRELLIS (each view alone and all views
together, several seeds), and give each an approximate metric size from MoGe-2 depth.
scripts/select_object_mesh.py then keeps the candidate FoundationPose fits best in all three
cameras and refines its scale.

Outputs under outputs/object_meshes/<object>/:
  views/                       contact sheet of view candidates and the RGBA condition crops
  candidates/cXX/trellis_raw.glb      TRELLIS mesh in its normalized unit cube (y-up)
  candidates/cXX/coarse_metric.glb    the same mesh scaled to the depth-derived extent
  scale_frames/episode_*/      rgb/mask/depth/intrinsics of each episode's best view
  provenance.json              views, scores, candidates, coarse scales
These meshes are generated estimates, never challenge ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
DATA = ROOT / "data" / "track_1"
OBJECT_OUT = ROOT / "outputs" / "object"
MESH_OUT = ROOT / "outputs" / "object_meshes"
TRELLIS_WEIGHTS = ROOT / "weights" / "trellis" / "TRELLIS-image-large"
MANUAL_VIEWS = ROOT / "config" / "manual_mesh_views.json"
TARGET_FACES = 40_000
PLATE_DARK = 0.1  # max RGB channel below which a TRELLIS vertex counts as plate-dark


def episodes_for(object_name: str) -> list[int]:
    rows = [json.loads(line) for line in (DATA / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if line.strip()]
    episodes = sorted(r["episode_index"] for r in rows if r["object"] == object_name)
    if not episodes:
        raise SystemExit(f"unknown object {object_name!r}")
    return episodes


def video_path(episode: int) -> Path:
    return DATA / "videos" / "chunk-000" / "observation.images.exo_camera" / f"episode_{episode:06d}.mp4"


def load_mask(episode: int, frame: int) -> np.ndarray:
    path = OBJECT_OUT / f"episode_{episode:06d}" / "sam2_masks" / "1" / f"{frame:06d}.png"
    return np.array(Image.open(path)) > 127


def load_depth_m(episode: int, frame: int) -> np.ndarray:
    # v2d DepthImage encoding: pixel = 65535 / (depth_m + 1).
    raw = np.array(Image.open(OBJECT_OUT / f"episode_{episode:06d}" / "depth" / f"{frame:06d}.png")).astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.where(raw > 0, 65535.0 / raw - 1.0, np.inf)


def load_intrinsics(episode: int, frame: int) -> dict:
    return json.loads((OBJECT_OUT / f"episode_{episode:06d}" / "intrinsics" / f"{frame:06d}.json").read_text())


def read_frames(episode: int, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode the requested frames sequentially (seeking is not frame-exact for every codec)."""
    capture = cv2.VideoCapture(str(video_path(episode)))
    frames, index = {}, 0
    last = max(wanted)
    while index <= last:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    missing = wanted - frames.keys()
    if missing:
        raise SystemExit(f"episode {episode}: could not decode frames {sorted(missing)[:5]}")
    return frames


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def score_episode_views(episode: int, stride: int, stability_gap: int) -> list[dict]:
    """Score every stride-th frame for use as a clean TRELLIS condition image.

    A good view is large (more pixels for the image encoder), compact relative to the
    episode's best compactness (hands and bodies carve concavities and split masks), not
    truncated by the image border, and temporally stable (static, unoccluded objects keep
    the same mask over a short window, while manipulated or occluded ones do not).
    """
    mask_dir = OBJECT_OUT / f"episode_{episode:06d}" / "sam2_masks" / "1"
    count = len(list(mask_dir.glob("*.png")))
    candidates = []
    for frame in range(stability_gap, count - stability_gap, stride):
        mask = load_mask(episode, frame)
        area = int(mask.sum())
        if area < 400:
            continue
        ys, xs = np.nonzero(mask)
        h, w = mask.shape
        if xs.min() <= 3 or ys.min() <= 3 or xs.max() >= w - 4 or ys.max() >= h - 4:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        largest = stats[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
        contours, _ = cv2.findContours((labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = cv2.contourArea(cv2.convexHull(np.concatenate(contours)))
        stability = min(mask_iou(mask, load_mask(episode, frame - stability_gap)),
                        mask_iou(mask, load_mask(episode, frame + stability_gap)))
        candidates.append({
            "episode": episode, "frame": frame, "area": area,
            "solidity": float(largest / hull_area) if hull_area else 0.0,
            "largest_component_fraction": float(largest / area),
            "stability": stability,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        })
    if not candidates:
        return []
    solidity_ref = np.percentile([c["solidity"] for c in candidates], 95) or 1.0
    area_ref = np.percentile([c["area"] for c in candidates], 95) or 1.0
    for c in candidates:
        c["score"] = float(min(c["area"] / area_ref, 1.2)
                           * min(c["solidity"] / solidity_ref, 1.0) ** 2
                           * c["largest_component_fraction"]
                           * c["stability"] ** 2)
    return sorted(candidates, key=lambda c: -c["score"])


def rgba_crop(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Full-frame RGBA with the SAM 2 mask as alpha; TRELLIS crops and centres it itself."""
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def contact_sheet(tiles: list[tuple[str, np.ndarray]], path: Path, size: int = 256) -> None:
    images = []
    for label, image in tiles:
        tile = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        images.append(tile)
    while len(images) % 4:
        images.append(np.zeros_like(images[0]))
    rows = [np.hstack(images[i:i + 4]) for i in range(0, len(images), 4)]
    Image.fromarray(np.vstack(rows)).save(path)


def square_crop(rgb: np.ndarray, mask: np.ndarray, bbox: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    half = int(max(x1 - x0, y1 - y0) * 0.65) + 8
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    shaded = rgb.copy()
    shaded[~mask] = (shaded[~mask] * 0.35).astype(np.uint8)
    pad = np.pad(shaded, ((half, half), (half, half), (0, 0)))
    return pad[cy:cy + 2 * half, cx:cx + 2 * half]


def observed_extent(episode: int, frame: int) -> tuple[float, int]:
    """Largest robust principal extent (m) of the masked MoGe-2 point cloud."""
    mask = cv2.erode(load_mask(episode, frame).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    depth = load_depth_m(episode, frame)
    k = load_intrinsics(episode, frame)
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > 0.05))
    z = depth[ys, xs]
    median = np.median(z)
    mad = np.median(np.abs(z - median)) * 1.4826 + 1e-6
    keep = np.abs(z - median) < 3.0 * mad + 0.02 * median
    xs, ys, z = xs[keep], ys[keep], z[keep]
    points = np.stack([(xs - k["cx"]) * z / k["fx"], (ys - k["cy"]) * z / k["fy"], z], axis=1)
    return principal_extent(points), int(len(points))


def principal_extent(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered[:: max(1, len(centered) // 20000)], full_matrices=False)
    projected = centered @ axes.T
    spans = np.percentile(projected, 98, axis=0) - np.percentile(projected, 2, axis=0)
    return float(spans.max())


def install_import_shims() -> None:
    """TRELLIS imports rembg, transformers, and open3d at package import time.

    The image pipeline only calls rembg when the input has no alpha channel, and only the
    text-to-3D pipeline uses transformers and open3d (whose wheel in the SAM3D image also
    lacks libusb). We always pass an RGBA image with the SAM 2 mask, so none of them is used;
    these stand-ins fail loudly if that assumption is ever broken.
    """
    def unavailable(name):
        def fail(*_args, **_kwargs):
            raise RuntimeError(f"{name} is not installed in this image and should not be needed")
        return fail

    if "rembg" not in sys.modules:
        rembg = types.ModuleType("rembg")
        rembg.new_session = unavailable("rembg.new_session")
        rembg.remove = unavailable("rembg.remove")
        sys.modules["rembg"] = rembg
    try:
        import open3d  # noqa: F401
    except ImportError:
        open3d = types.ModuleType("open3d")
        # trellis_text_to_3d uses o3d.geometry.TriangleMesh in a signature annotation.
        open3d.geometry = types.SimpleNamespace(TriangleMesh=type("TriangleMesh", (), {}))
        sys.modules["open3d"] = open3d
    try:
        import transformers  # noqa: F401
    except ImportError:
        shim = types.ModuleType("transformers")
        for name in ("CLIPTextModel", "AutoTokenizer"):
            setattr(shim, name, type(name, (), {"from_pretrained": staticmethod(unavailable(f"transformers.{name}"))}))
        sys.modules["transformers"] = shim


def load_pipeline():
    install_import_shims()
    sys.path.insert(0, str(ROOT / "third_party" / "TRELLIS"))
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(TRELLIS_WEIGHTS))
    pipeline.cuda()
    return pipeline


def trellis_mesh(pipeline, condition: list[Image.Image], seed: int):
    import torch

    sampler = dict(sparse_structure_sampler_params={"steps": 25, "cfg_strength": 7.5},
                   slat_sampler_params={"steps": 25, "cfg_strength": 3.0})
    with torch.no_grad():
        if len(condition) == 1:
            outputs = pipeline.run(condition[0], seed=seed, formats=["mesh"], **sampler)
        else:
            outputs = pipeline.run_multi_image(condition, seed=seed, formats=["mesh"], mode="stochastic", **sampler)
    mesh = outputs["mesh"][0]
    if not mesh.success:
        return None
    vertices = mesh.vertices.detach().cpu().numpy().astype(np.float64)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int64)
    colors = np.clip(mesh.vertex_attrs[:, :3].detach().float().cpu().numpy(), 0.0, 1.0)
    return vertices, faces, colors


def plate_suspect(vertices: np.ndarray, colors: np.ndarray) -> bool:
    """TRELLIS sometimes embeds dim, top-down objects in a dark plate filling the whole grid."""
    return bool(sorted(np.ptp(vertices, axis=0))[-2] > 0.97 and (colors.max(axis=1) < PLATE_DARK).mean() > 0.4)


def remove_plate(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    """Drop faces whose vertices are all near-black; the embedded object keeps its own colours."""
    return vertices, faces[~(colors.max(axis=1) < PLATE_DARK)[faces].all(axis=1)], colors


def clean_mesh(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, keep_fraction: float = 0.02):
    """Drop floater components, decimate to a FoundationPose-friendly size, and go y-up."""
    import pyvista as pv
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=(colors * 255).astype(np.uint8), process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        largest = max(len(p.faces) for p in parts)
        mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) >= keep_fraction * largest])
    if len(mesh.faces) > TARGET_FACES:
        source_vertices = np.asarray(mesh.vertices)
        source_colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
        poly = pv.PolyData(source_vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel())
        poly = poly.decimate(1.0 - TARGET_FACES / len(mesh.faces)).triangulate().clean()
        decimated_vertices = np.asarray(poly.points, dtype=np.float64)
        decimated_faces = np.asarray(poly.faces).reshape(-1, 4)[:, 1:]
        # VTK decimation moves vertices; carry colour over from the nearest original vertex.
        nearest = cKDTree(source_vertices).query(decimated_vertices)[1]
        mesh = trimesh.Trimesh(decimated_vertices, decimated_faces,
                               vertex_colors=source_colors[nearest], process=True)
    # TRELLIS is z-up; glTF is y-up (same rotation as trellis.utils.postprocessing_utils.to_glb).
    mesh.vertices = mesh.vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    mesh.vertices -= mesh.bounds.mean(axis=0)
    return mesh


def manual_views(object_name: str) -> list[dict] | None:
    if not MANUAL_VIEWS.exists():
        return None
    entry = json.loads(MANUAL_VIEWS.read_text()).get(object_name)
    return entry or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--object", required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--stability_gap", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()

    out = MESH_OUT / args.object
    views_dir = out / "views"
    if out.exists():
        shutil.rmtree(out)
    views_dir.mkdir(parents=True)

    episodes = episodes_for(args.object)
    ranked = {e: score_episode_views(e, args.stride, args.stability_gap) for e in episodes}
    override = manual_views(args.object)
    if override:
        best_per_episode = {v["episode"]: {**v, "score": None, "source": "manual_visual_review"} for v in override}
        for e in episodes:
            if e not in best_per_episode and ranked[e]:
                best_per_episode[e] = {**ranked[e][0], "source": "automatic"}
    else:
        best_per_episode = {e: {**ranked[e][0], "source": "automatic"} for e in episodes if ranked[e]}
    if not best_per_episode:
        raise SystemExit(f"{args.object}: no usable unoccluded, untruncated view in episodes {episodes}")

    # Contact sheet of the top automatic candidates per episode for visual review.
    sheet_frames = {e: [c["frame"] for c in ranked[e][:4]] for e in episodes}
    for e, view in best_per_episode.items():
        sheet_frames.setdefault(e, []).append(view["frame"])
    decoded = {e: read_frames(e, set(frames)) for e, frames in sheet_frames.items() if frames}
    tiles = []
    for e in episodes:
        for c in ranked[e][:4]:
            mask = load_mask(e, c["frame"])
            tiles.append((f"ep{e} f{c['frame']} s{c['score']:.2f}", square_crop(decoded[e][c["frame"]], mask, c["bbox"])))
    if tiles:
        contact_sheet(tiles, views_dir / "candidates.jpg")

    # Each episode's best view, and all of them together, condition TRELLIS. Single-view
    # generations from these small exo crops are semantically unstable (benches, dishes,
    # plates), so every condition/seed pair becomes a candidate and
    # scripts/select_object_mesh.py keeps the one FoundationPose fits best in all cameras.
    images = {}
    for e, view in sorted(best_per_episode.items()):
        images[e] = rgba_crop(decoded[e][view["frame"]], load_mask(e, view["frame"]))
        images[e].save(views_dir / f"condition_ep{e:06d}_f{view['frame']:06d}.png")
    conditions = [([e], [images[e]]) for e in sorted(images)]
    if len(images) > 1:
        conditions.append((sorted(images), [images[e] for e in sorted(images)]))

    observed = {}
    for e, view in sorted(best_per_episode.items()):
        extent, n_points = observed_extent(e, view["frame"])
        observed[e] = {"episode": e, "frame": view["frame"], "source": view["source"], "score": view["score"],
                       "observed_extent_m": extent, "depth_points": n_points}
        frame_dir = out / "scale_frames" / f"episode_{e:06d}"
        frame_dir.mkdir(parents=True)
        Image.fromarray(decoded[e][view["frame"]]).save(frame_dir / "rgb.png")
        shutil.copy(OBJECT_OUT / f"episode_{e:06d}" / "sam2_masks" / "1" / f"{view['frame']:06d}.png", frame_dir / "mask.png")
        shutil.copy(OBJECT_OUT / f"episode_{e:06d}" / "depth" / f"{view['frame']:06d}.png", frame_dir / "depth.png")
        shutil.copy(OBJECT_OUT / f"episode_{e:06d}" / "intrinsics" / f"{view['frame']:06d}.json", frame_dir / "intrinsics.json")

    pipeline = load_pipeline()
    candidates = []
    generated = 0
    for condition_episodes, condition_images in conditions:
        for seed in args.seeds:
            cid = f"c{generated:02d}"
            generated += 1
            result = trellis_mesh(pipeline, condition_images, seed)
            entry = {"id": cid, "condition_episodes": condition_episodes, "seed": seed}
            if result is None:
                candidates.append({**entry, "status": "empty"})
                continue
            vertices, faces, colors = result
            plate = plate_suspect(vertices, colors)
            variants = [(cid, entry, (vertices, faces, colors), 0.02, plate)]
            if plate:
                # A de-plated copy usually recovers the embedded object; a genuinely black object
                # would lose itself instead, so the raw mesh stays a candidate too.
                variants.append((f"{cid}d", {**entry, "id": f"{cid}d", "derived_from": cid, "deplated": True},
                                 remove_plate(vertices, faces, colors), 0.05, False))
            for vid, ventry, (v, f, c), keep_fraction, vplate in variants:
                if len(f) == 0:
                    candidates.append({**ventry, "status": "empty"})
                    continue
                mesh = clean_mesh(v, f, c, keep_fraction)
                mesh_extent = principal_extent(np.asarray(mesh.sample(30000)))
                coarse = float(np.median([o["observed_extent_m"] / mesh_extent for o in observed.values()]))
                cdir = out / "candidates" / vid
                cdir.mkdir(parents=True)
                mesh.export(cdir / "trellis_raw.glb")
                metric = mesh.copy()
                metric.apply_scale(coarse)
                metric.export(cdir / "coarse_metric.glb")
                candidates.append({**ventry, "status": "ok", "plate_suspect": vplate,
                                   "faces": int(len(mesh.faces)), "normalized_principal_extent": mesh_extent,
                                   "coarse_scale": coarse, "coarse_extents_m": [float(x) for x in metric.extents]})
                print(json.dumps(candidates[-1]), flush=True)

    provenance = {
        "schema": "v2d.track1.generated_object_mesh.v2",
        "object": args.object,
        "ground_truth": False,
        "generator": {"model": "microsoft/TRELLIS-image-large",
                      "revision": (TRELLIS_WEIGHTS / "REVISION").read_text().strip(),
                      "sampler": {"sparse_structure": {"steps": 25, "cfg_strength": 7.5},
                                  "slat": {"steps": 25, "cfg_strength": 3.0}}},
        "condition_views": [{k: observed[e][k] for k in ("episode", "frame", "source", "score")} for e in sorted(observed)],
        "per_episode": [observed[e] for e in sorted(observed)],
        "candidates": candidates,
        "top_view_candidates": {str(e): ranked[e][:4] for e in episodes},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    usable = [c for c in candidates if c["status"] == "ok"]
    if not usable:
        raise SystemExit(f"{args.object}: TRELLIS produced no usable candidate")
    print(f"PASS: {len(usable)} TRELLIS mesh candidates for {args.object} "
          f"({sum(c['plate_suspect'] for c in usable)} plate-suspect)")


if __name__ == "__main__":
    main()
