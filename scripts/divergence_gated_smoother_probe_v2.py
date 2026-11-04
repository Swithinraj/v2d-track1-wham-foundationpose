"""Robust Divergence-Gated Smoother for FoundationPose -- probe v2 copy of tools/divergence_gated_smoother.py.

Changes vs the original (gating thresholds and logic unchanged): (1) zero-pose guard: never fall back to a raw pose that is
FoundationPose's failure output (zero translation), has z < 0.05 m, or comes from an empty/missing mask -- keep the
smoothed pose and log GUARD_KEPT_SMOOTHED; (2) raw input = reports/probe_v2/step1_raw (unmodified 193382a tracker,
production settings), output = reports/probe_v2/step2_gate; (3) depth metric decodes the inverse-depth PNG as
65535/v - 1 (was /1000; reporting only); (4) all events are recorded; (5) optional --raw-outlier-guard (user-approved
2026-11-04): never fall back when the raw silhouette IoU is < 0.05 or below the smoothed IoU (a raw outlier such as
ep16 f120); (6) --out-base.


Prevents RTS backward smoother divergence (e.g. 2.22m drift on black_pan, 6.9m on bowl)
by enforcing a divergence gate between raw measurements and smoothed output:

Gating Policy:
1. At each frame k, compute discrepancy between smoothed pose and raw pose:
   delta_t = ||t_smooth - t_raw||_2
   delta_R = geodesic_angle(R_smooth, R_raw)
2. Compute silhouette IoU of both against observed SAM2 mask: iou_smooth, iou_raw
3. If delta_t > max_trans_drift (e.g. 15 cm) and (iou_smooth < iou_raw - 0.05 or iou_raw > 0.40):
   -> The RTS smoother has diverged!
   -> Re-anchor / clamp the pose to raw measurement (or local bounded EMA filter).

Outputs:
- Generates corrected poses in reports/divergence_gated_smoother/{tag}/
- Computes before-vs-after comparison table on silhouette IoU, depth residual, and drift.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import trimesh

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path('/work/smoses2s/thesis_abl')
REPO = ROOT / 'repo'
RAW_BASE = ROOT / 'reports/probe_v2/step1_raw'
OUT_BASE = ROOT / 'reports/probe_v2/step2_gate'


def compute_geodesic_angle(R1, R2):
    R_diff = R1.T @ R2
    trace_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_val)))


def render_silhouette_iou(mesh, scene, K, Ki, R, t, mask):
    h, w = mask.shape
    V = np.asarray(mesh.vertices)
    posed = V @ R.T + t
    good = np.isfinite(posed).all(axis=1) & (posed[:, 2] > 0.05)
    if not good.any():
        return 0.0
    p = posed[good] @ K.T
    p = p[:, :2] / p[:, 2, None]
    low = np.maximum(np.floor(p.min(axis=0) - 2).astype(int), [0, 0])
    high = np.minimum(np.ceil(p.max(axis=0) + 2).astype(int), [w, h])
    x0, y0 = low
    x1, y1 = high
    if x1 <= x0 or y1 <= y0:
        return 0.0

    y, x = np.mgrid[y0:y1, x0:x1]
    directions = np.stack([x + 0.5, y + 0.5, np.ones_like(x)], axis=-1) @ Ki.T
    origin = (-t) @ R
    directions = directions @ R
    rays = np.concatenate([np.broadcast_to(origin, directions.shape), directions], axis=-1).astype(np.float32)

    hits = np.isfinite(scene.cast_rays(o3d.core.Tensor(rays), nthreads=2)['t_hit'].numpy())
    silhouette = np.zeros((h, w), dtype=bool)
    silhouette[y0:y1, x0:x1] = hits

    intersection = np.logical_and(silhouette, mask).sum()
    union = np.logical_or(silhouette, mask).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_depth_residual_cm(mesh, scene, K, Ki, R, t, depth_map, mask):
    # Unproject masked depth points
    y, x = np.where(mask & np.isfinite(depth_map) & (depth_map > 0.1) & (depth_map < 10.0))
    if len(x) < 20:
        return None
    d = depth_map[y, x]
    # Pick sample if too large
    if len(x) > 1000:
        idx = np.random.choice(len(x), 1000, replace=False)
        x, y, d = x[idx], y[idx], d[idx]

    pts = (np.column_stack([x, y, np.ones(len(x))]) @ Ki.T) * d[:, None]
    local = (pts - t) @ R
    closest = scene.compute_closest_points(o3d.core.Tensor(local.astype(np.float32)), nthreads=2)['points'].numpy()
    dist = np.linalg.norm(local - closest, axis=1) * 100.0  # cm
    return float(np.median(dist))


RAW_OUTLIER_GUARD = False


def run_gated_smoothing_episode(ep, max_trans_drift_cm=15.0, min_iou_gain=0.05):
    tag = f'episode_{ep:06d}'
    base = REPO / 'outputs/object' / tag
    raw_dir = RAW_BASE / tag / 'raw'
    smoothed_dir = base / 'foundationpose/smoothed'

    if not raw_dir.exists():
        logger.error(f"Raw poses not found in {raw_dir}")
        return dict(episode=ep, error="Raw poses missing")

    meta_rows = [json.loads(l) for l in (REPO / 'data/track_1/meta/episodes_metadata.jsonl').read_text().splitlines() if l.strip()]
    metadata = {r['episode_index']: r for r in meta_rows}
    obj = metadata[ep]['object']

    ref = int((base / 'prompt/reference_frame.txt').read_text().strip())
    intr = json.loads((base / f'intrinsics/{ref:06d}.json').read_text())
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.0]], dtype=float)
    Ki = np.linalg.inv(K)

    mesh = trimesh.load(REPO / f'inputs/object_meshes/{obj}.glb', force='mesh', process=False)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32))
    ))

    raw_files = sorted(raw_dir.glob('*.json'))
    total_frames = len(raw_files)
    logger.info(f"Processing {total_frames} frames for {tag} ({obj}) with divergence gate threshold {max_trans_drift_cm} cm")

    out_ep_dir = OUT_BASE / tag
    gated_dir = out_ep_dir / 'gated_poses'
    gated_dir.mkdir(parents=True, exist_ok=True)

    divergence_events = []
    guard_events = []
    before_ious = []
    after_ious = []
    before_depth_errs = []
    after_depth_errs = []
    deltas_smooth_raw = []
    deltas_gated_raw = []

    # Scored / sampled frames for fast metrics
    scored_path = REPO / 'outputs/hand_fix/episodes_bmq_base_meshes/scored_frames.json'
    scored_frames = set(json.loads(scored_path.read_text()).get(str(ep), list(range(0, total_frames, 10))))

    for k in range(total_frames):
        fname = f'{k:06d}.json'
        raw_f = raw_dir / fname
        sm_f = smoothed_dir / fname
        if not raw_f.exists() or not sm_f.exists():
            continue

        p_raw = json.loads(raw_f.read_text())
        p_sm = json.loads(sm_f.read_text())

        t_rw = np.array(p_raw['translation'], dtype=float)
        t_sm = np.array(p_sm['translation'], dtype=float)

        q_rw = p_raw['rotation']
        q_sm = p_sm['rotation']
        R_rw = Rotation.from_quat([q_rw[1], q_rw[2], q_rw[3], q_rw[0]]).as_matrix()
        R_sm = Rotation.from_quat([q_sm[1], q_sm[2], q_sm[3], q_sm[0]]).as_matrix()

        d_t = float(np.linalg.norm(t_sm - t_rw) * 100.0)  # cm
        d_R = compute_geodesic_angle(R_rw, R_sm)

        deltas_smooth_raw.append(d_t)

        mask_f = base / f'sam2_masks/1/{k:06d}.png'
        mask = cv2.imread(str(mask_f), 0) > 0 if mask_f.exists() else None
        depth_f = base / f'depth/{k:06d}.png'
        _raw_d = cv2.imread(str(depth_f), -1).astype(float) if depth_f.exists() else None
        depth_map = np.where(_raw_d > 0, 65535.0 / np.maximum(_raw_d, 1.0) - 1.0, np.nan) if _raw_d is not None else None

        # Check IoU if evaluation frame or if drift is large
        needs_eval = (k in scored_frames) or (d_t > max_trans_drift_cm)
        iou_rw, iou_sm = None, None
        if needs_eval and mask is not None and mask.any():
            iou_rw = render_silhouette_iou(mesh, scene, K, Ki, R_rw, t_rw, mask)
            iou_sm = render_silhouette_iou(mesh, scene, K, Ki, R_sm, t_sm, mask)

        # Gating Decision
        # Smoother diverged if:
        # 1. Translation drift > 2x threshold (e.g. > 30 cm) -> unconditionally diverged
        # 2. Translation drift > threshold (15 cm) AND (silhouette IoU collapsed, raw is substantially better, or no mask)
        is_diverged = False
        if d_t > max_trans_drift_cm * 2.0:
            is_diverged = True
        elif d_t > max_trans_drift_cm:
            if iou_rw is not None and iou_sm is not None and (iou_rw > 0.05 or iou_sm > 0.05):
                if iou_sm < iou_rw - min_iou_gain or (iou_sm < 0.35 and iou_rw > 0.45):
                    is_diverged = True
            else:
                is_diverged = True

        raw_invalid = bool(np.abs(t_rw).max() < 1e-6 or t_rw[2] < 0.05 or mask is None or not mask.any())
        guard_kept = False
        if is_diverged and raw_invalid:
            # zero-pose guard: the raw pose is a tracker failure / unobserved frame -> never substitute it
            is_diverged = False
            guard_kept = True
            guard_events.append(dict(frame=k, delta_t_cm=round(d_t, 2), delta_R_deg=round(d_R, 2),
                                     raw_t=[round(float(x), 4) for x in t_rw],
                                     mask_pixels=int(mask.sum()) if mask is not None else None,
                                     action='GUARD_KEPT_SMOOTHED', reason='raw_invalid'))
        if (is_diverged and RAW_OUTLIER_GUARD and iou_rw is not None and iou_sm is not None
                and (iou_rw < 0.05 or iou_rw < iou_sm)):
            # raw-outlier guard: the raw pose explains the observed silhouette no better than the smoothed one
            is_diverged = False
            guard_kept = True
            guard_events.append(dict(frame=k, delta_t_cm=round(d_t, 2), delta_R_deg=round(d_R, 2),
                                     iou_raw=round(iou_rw, 4), iou_smooth=round(iou_sm, 4),
                                     action='GUARD_KEPT_SMOOTHED', reason='raw_not_better'))

        if is_diverged:
            # Fall back to raw pose
            t_final = t_rw.copy()
            R_final = R_rw.copy()
            q_final = q_rw
            action = "FALLBACK_TO_RAW"
            divergence_events.append(dict(
                frame=k,
                delta_t_cm=round(d_t, 2),
                delta_R_deg=round(d_R, 2),
                iou_raw=round(iou_rw, 4) if iou_rw is not None else None,
                iou_smooth=round(iou_sm, 4) if iou_sm is not None else None,
                action=action,
            ))
        else:
            # Keep smoothed pose
            t_final = t_sm.copy()
            R_final = R_sm.copy()
            q_final = q_sm
            action = "GUARD_KEPT_SMOOTHED" if guard_kept else "ACCEPTED_SMOOTHED"

        # Record metrics on scored frames
        if k in scored_frames and mask is not None and mask.any():
            iou_final = iou_rw if is_diverged else (iou_sm if iou_sm is not None else render_silhouette_iou(mesh, scene, K, Ki, R_final, t_final, mask))
            iou_orig_sm = iou_sm if iou_sm is not None else render_silhouette_iou(mesh, scene, K, Ki, R_sm, t_sm, mask)
            before_ious.append(iou_orig_sm)
            after_ious.append(iou_final)

            if depth_map is not None:
                d_orig = compute_depth_residual_cm(mesh, scene, K, Ki, R_sm, t_sm, depth_map, mask)
                d_final = compute_depth_residual_cm(mesh, scene, K, Ki, R_final, t_final, depth_map, mask)
                if d_orig is not None and d_final is not None:
                    before_depth_errs.append(d_orig)
                    after_depth_errs.append(d_final)

        # Save gated pose
        p_gated = dict(
            translation=t_final.tolist(),
            rotation=q_final if isinstance(q_final, list) else [float(x) for x in q_final],
            scale=p_raw.get('scale', [1.0]),
            gating_action=action,
            original_drift_cm=round(d_t, 2),
        )
        (gated_dir / fname).write_text(json.dumps(p_gated, indent=2))
        deltas_gated_raw.append(float(np.linalg.norm(t_final - t_rw) * 100.0))

    summary = dict(
        episode=ep,
        object=obj,
        total_frames=total_frames,
        divergence_events_count=len(divergence_events),
        divergence_event_fraction=round(len(divergence_events) / max(1, total_frames), 4),
        max_translation_drift_before_cm=round(float(np.max(deltas_smooth_raw)), 2) if deltas_smooth_raw else None,
        max_translation_drift_after_cm=round(float(np.max(deltas_gated_raw)), 2) if deltas_gated_raw else None,
        silhouette_iou=dict(
            before_smoothed_median=round(float(np.median(before_ious)), 4) if before_ious else None,
            after_gated_median=round(float(np.median(after_ious)), 4) if after_ious else None,
            delta_iou=round(float(np.median(after_ious) - np.median(before_ious)), 4) if before_ious else None,
        ),
        depth_residual_cm=dict(
            before_smoothed_median=round(float(np.median(before_depth_errs)), 2) if before_depth_errs else None,
            after_gated_median=round(float(np.median(after_depth_errs)), 2) if after_depth_errs else None,
            delta_depth_cm=round(float(np.median(after_depth_errs) - np.median(before_depth_errs)), 2) if before_depth_errs else None,
        ),
        divergence_events=divergence_events,
        guard_events_count=len(guard_events),
        guard_events=guard_events,
    )

    out_file = out_ep_dir / 'gated_smoothing_summary.json'
    out_file.write_text(json.dumps(summary, indent=2))
    logger.info(f"Saved gated smoothing summary to {out_file}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, nargs='+', default=[12, 3, 24])
    p.add_argument('--max-trans-drift-cm', type=float, default=15.0)
    p.add_argument('--out-summary', type=Path, default=None)
    p.add_argument('--out-base', type=Path, default=None)
    p.add_argument('--raw-outlier-guard', action='store_true')
    p.add_argument('--raw-base', type=Path, default=None)
    args = p.parse_args()
    global OUT_BASE, RAW_OUTLIER_GUARD, RAW_BASE
    if args.raw_base is not None:
        RAW_BASE = args.raw_base
    if args.out_base is not None:
        OUT_BASE = args.out_base
    RAW_OUTLIER_GUARD = args.raw_outlier_guard
    if args.out_summary is None:
        args.out_summary = OUT_BASE / 'summary.json'
    print(f'raw_base={RAW_BASE} out_base={OUT_BASE} raw_outlier_guard={RAW_OUTLIER_GUARD}', flush=True)

    results = []
    print(f"Running Divergence-Gated Smoother on episodes {args.episodes} (max drift threshold: {args.max_trans_drift_cm} cm)...", flush=True)
    for ep in args.episodes:
        res = run_gated_smoothing_episode(ep, max_trans_drift_cm=args.max_trans_drift_cm)
        results.append(res)
        iou = res.get('silhouette_iou', {})
        dep = res.get('depth_residual_cm', {})
        delta_iou_str = f"{iou.get('delta_iou'):+.4f}" if iou.get('delta_iou') is not None else "N/A"
        delta_dep_str = f"{dep.get('delta_depth_cm'):+.2f}" if dep.get('delta_depth_cm') is not None else "N/A"
        print(f"\n=== Episode {ep:2d} ({res.get('object')}): Gated Smoother Results ===", flush=True)
        print(f"  Divergence events: {res.get('divergence_events_count')} / {res.get('total_frames')} frames ({res.get('divergence_event_fraction')*100:.1f}%)", flush=True)
        print(f"  Max Translation Drift: {res.get('max_translation_drift_before_cm')} cm -> {res.get('max_translation_drift_after_cm')} cm", flush=True)
        print(f"  Silhouette IoU:       {iou.get('before_smoothed_median')} -> {iou.get('after_gated_median')} (delta = {delta_iou_str})", flush=True)
        print(f"  Depth Residual:       {dep.get('before_smoothed_median')} cm -> {dep.get('after_gated_median')} cm (delta = {delta_dep_str} cm)", flush=True)

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(results, indent=2))
    print(f"\nAll done! Saved summary to {args.out_summary}", flush=True)


if __name__ == '__main__':
    main()
