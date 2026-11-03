#!/usr/bin/env python3
"""Convert kit-layout candidate npz dirs (pose [T_full,136], scales, shape, object arrays, _object.glb)
to the scored-frames layout used by ens_reframe / patch_parquet npz (frames, mhr_pose, mhr_scales,
mhr_shape, object_*). Scored frames come from a reference scored-frames dir. The object mesh is taken
from the sibling P4tab102 dir (same tracker mesh): it is only used by the ens_reframe diagnostic;
builds patch human arrays only (mhr_pose/scales/shape), never the mesh.
usage: kit2calib.py SRC_KIT_DIR REF_SCORED_DIR OUT_DIR"""
import sys, numpy as np
from pathlib import Path
src, ref, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
for e in range(30):
    d = dict(np.load(src / f"episode_{e:06d}.npz")); f = np.load(ref / f"episode_{e:06d}.npz")["frames"]
    m = np.load(ref.parent / "P4tab102" / f"episode_{e:06d}.npz")   # same tracker mesh (diagnostic only)
    np.savez(out / f"episode_{e:06d}.npz", frames=f, mhr_pose=d["pose"][f], mhr_scales=d["scales"], mhr_shape=d["shape"],
             object_rotation=d["object_rotation"][f], object_translation=d["object_translation"][f],
             object_scale=d["object_scale"], object_mesh_vertices=m["object_mesh_vertices"], object_mesh_faces=m["object_mesh_faces"])
print("converted", src.name)
