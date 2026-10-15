#!/usr/bin/env python3
"""Review sheet for the installed generated object meshes (CPU, no GPU or display needed).

One row per object: its condition crops, then the installed mesh from four azimuths
(painter's-algorithm render of the vertex colours), labelled with the selection result.
Runs in containers/v2d_sam3d.sif. Writes outputs/object_meshes/review_sheet.png.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = Path(os.environ.get("TRACK1_ROOT") or Path(__file__).resolve().parents[1])
TILE = 220


def mesh_views(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    vertices = mesh.vertices - mesh.bounds.mean(axis=0)
    vertices = vertices / np.abs(vertices).max()
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)
    tiles = []
    for azimuth in (0, 90, 180, 270):
        a, el = np.radians(azimuth), np.radians(25)
        ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
        tri = (vertices @ (rx @ ry).T)[mesh.faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
        shade = colors[mesh.faces].mean(axis=1) * (0.35 + 0.65 * np.abs(normal[:, 2]))[:, None]
        image = Image.new("RGB", (TILE, TILE), (70, 70, 90))
        draw = ImageDraw.Draw(image)
        for face in np.argsort(-tri[:, :, 2].mean(axis=1)):
            draw.polygon([((x * 0.45 + 0.5) * TILE, (0.5 - y * 0.45) * TILE) for x, y, _ in tri[face]],
                         fill=tuple(int(c) for c in np.clip(shade[face], 0, 255)))
        tiles.append(np.asarray(image))
    return tiles


def condition_tile(path: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA")).astype(float) / 255
    ys, xs = np.nonzero(rgba[..., 3] > 0.5)
    half = int(max(np.ptp(xs), np.ptp(ys)) * 0.6) + 4
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    rgb = (rgba[..., :3] * rgba[..., 3:4] * 255).astype(np.uint8)
    crop = Image.fromarray(np.pad(rgb, ((half, half), (half, half), (0, 0))))
    crop = crop.crop((cx, cy, cx + 2 * half, cy + 2 * half))
    return np.asarray(crop.resize((TILE, TILE)))


def main() -> None:
    rows = []
    for provenance_path in sorted((ROOT / "inputs" / "object_meshes").glob("*.provenance.json")):
        provenance = json.loads(provenance_path.read_text())
        name = provenance["object"]
        conditions = sorted((ROOT / "outputs" / "object_meshes" / name / "views").glob("condition_*.png"))
        tiles = [condition_tile(p) for p in conditions[:3]]
        tiles += [np.full((TILE, TILE, 3), 30, np.uint8)] * (3 - len(tiles))
        tiles += mesh_views(trimesh.load(provenance_path.with_name(f"{name}.glb"), force="mesh"))
        row = Image.fromarray(np.hstack(tiles))
        selection = provenance.get("selection", {})
        refined = selection.get("refined_per_episode", {})
        iou = np.mean([r["iou"] for r in refined.values()]) if refined else float("nan")
        label = (f"{name}  winner {selection.get('winner')}  IoU {iou:.2f}  "
                 f"extents {np.round(provenance.get('final_extents_m', []), 2).tolist()} m"
                 + ("  REVIEW" if provenance.get("requires_visual_review") else ""))
        ImageDraw.Draw(row).text((6, 6), label, fill=(255, 255, 0))
        rows.append(np.asarray(row))
    if not rows:
        raise SystemExit("no installed generated meshes found")
    out = ROOT / "outputs" / "object_meshes" / "review_sheet.png"
    Image.fromarray(np.vstack(rows)).save(out)
    print(f"PASS: wrote {out} ({len(rows)} objects)")


if __name__ == "__main__":
    main()
