#!/usr/bin/env python3
"""Experimental constant object translation from independently observed contacts.

This is a geometric proposal, not a learned correction or a validated Kaggle recipe.
Inputs are in camera coordinates/metres (mesh vertices remain in mesh-local units).
Contact samples must come from independent, pre-contact hands. Fitting hands already
anchored to the same object is circular. An accepted fit still needs independent image
validation and a downstream contact solve before it becomes a submission candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree


def fit_translation(vertices, normals, rotation, translation, scale, hands, contact,
                    *, max_shift=.15, min_frames=12, tolerance=.015):
    """Return a constant offset and diagnostics, abstaining on weak/inconsistent evidence.

    vertices/normals [V,3], rotation [T,3,3], translation [T,3], hands [T,P,3],
    contact [T,P] boolean (selected independently). Robust point-to-plane ICP uses
    surface normals for observability and disjoint temporal halves for agreement.
    Rejected proposals always return a zero applied offset.
    """
    vertices, normals, rotation, translation, hands = (
        np.asarray(a, float) for a in (vertices, normals, rotation, translation, hands))
    contact = np.asarray(contact, bool)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
        raise ValueError("vertices must have shape [V,3], V >= 3")
    if normals.shape != vertices.shape:
        raise ValueError("normals must match vertices")
    if translation.ndim != 2 or translation.shape[1:] != (3,):
        raise ValueError("translation must have shape [T,3]")
    T = len(translation)
    if rotation.shape != (T,3,3) or hands.ndim != 3 or hands.shape[0] != T or hands.shape[2] != 3:
        raise ValueError("frame/coordinate shape mismatch")
    if contact.shape != hands.shape[:2]:
        raise ValueError("contact mask must match hand samples")
    if not all(np.isfinite(a).all() for a in (vertices,normals,rotation,translation,hands)):
        raise ValueError("geometry must be finite")
    if not np.isfinite(scale) or scale <= 0 or max_shift <= 0 or tolerance <= 0 or min_frames < 4:
        raise ValueError("invalid scale or solver bounds")
    if not np.allclose(rotation @ rotation.swapaxes(1,2), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotation),1,atol=1e-4):
        raise ValueError("rotation must be proper orthonormal matrices")
    length = np.linalg.norm(normals,axis=1)
    if (length < 1e-8).any():
        raise ValueError("surface normals must be nonzero")
    normals = normals / length[:,None]
    report = dict(accepted=False, reason="insufficient contact frames", translation_m=[0.,0.,0.])
    frames, points = np.nonzero(contact)
    unique = np.unique(frames)
    report['contact_frames'] = len(unique)
    if len(unique) < min_frames:
        return report
    H, R, t = hands[frames,points], rotation[frames], translation[frames]
    tree = cKDTree(vertices)

    def correspondence(shift, selection):
        local = np.einsum('ni,nij->nj',H[selection]-t[selection]-shift,R[selection]) / scale
        distance, idx = tree.query(local)
        N = np.einsum('nij,nj->ni',R[selection],normals[idx])
        surface = scale*np.einsum('nij,nj->ni',R[selection],vertices[idx])+t[selection]
        residual = np.sum((H[selection]-surface-shift)*N,axis=1)
        return distance*scale, N, residual

    def optimize(selection):
        shift = np.zeros(3)
        for _ in range(60):
            distance,N,residual = correspondence(shift,selection)
            # Reject gross non-contact observations; Huber weights limit the rest.
            keep = distance <= max(.03, np.quantile(distance,.8))
            weights = np.minimum(1., .005/np.maximum(np.abs(residual),1e-12)) * keep
            root = np.sqrt(weights)
            delta = np.linalg.lstsq(N*root[:,None],residual*root,rcond=1e-6)[0]
            shift += delta
            if np.linalg.norm(shift) > max_shift:
                return shift
            if np.linalg.norm(delta) < 1e-7:
                break
        return shift

    all_points = np.ones(len(frames),bool)
    proposal = optimize(all_points)
    before,_,_ = correspondence(np.zeros(3),all_points)
    after,N,_ = correspondence(proposal,all_points)
    report.update(proposal_m=proposal.tolist(), residual_before_m=float(np.median(before)),
                  residual_after_m=float(np.median(after)))
    if not np.isfinite(proposal).all() or np.linalg.norm(proposal) > max_shift:
        report['reason']='offset exceeds bound'; return report
    support = after <= tolerance
    report['support_fraction']=float(np.mean(support))
    eigen = np.linalg.eigvalsh(N[support].T @ N[support]) if support.any() else np.zeros(3)
    ratio = eigen[0]/max(eigen[-1],1e-12)
    report['normal_information_ratio']=float(ratio)
    if ratio < .02:
        report['reason']='translation is not observable from contact normals'; return report
    if support.mean() < .6:
        report['reason']='insufficient surface agreement'; return report
    first = frames < unique[len(unique)//2]
    halves = [optimize(first),optimize(~first)]
    disagreement = float(np.linalg.norm(halves[0]-halves[1]))
    report['temporal_disagreement_m']=disagreement
    if disagreement > .02:
        report['reason']='inconsistent offsets across temporal halves'; return report
    if np.median(before)-np.median(after) < .002:
        report['reason']='no meaningful contact improvement'; return report
    report.update(accepted=True,reason='geometric proposal; independent image validation still required',
                  translation_m=proposal.tolist())
    return report


def render_mask(vertices, faces, rotation, translation, scale, camera, downsample=4):
    """Rasterize a positive-depth mesh silhouette at reduced resolution."""
    import cv2
    X=scale*np.asarray(vertices) @ np.asarray(rotation).T+translation
    if not np.isfinite(X).all() or (X[:,2] <= .05).any():
        raise ValueError('mesh crosses or lies behind the camera')
    height,width=int(camera['height'])//downsample,int(camera['width'])//downsample
    if min(height,width)<=0: raise ValueError('invalid camera image dimensions')
    uv=np.c_[camera['fx']*X[:,0]/X[:,2]+camera['cx'],camera['fy']*X[:,1]/X[:,2]+camera['cy']]/downsample
    if not np.isfinite(uv).all() or np.max(np.abs(uv))>1e7: raise ValueError('invalid camera projection')
    triangles=np.round(uv[np.asarray(faces,int)]).astype(np.int32)
    mask=np.zeros((height,width),np.uint8)
    for triangle in triangles:
        cv2.fillConvexPoly(mask,triangle,1)
    return mask.astype(bool)


def silhouette_gate(vertices,faces,rotation,translation,scale,shift,camera,masks,occluders=None):
    """Require positive held-out image evidence, not just a lower contact objective.

    masks/occluders map frame indices to boolean masks at render_mask resolution.
    Human-occluded pixels are excluded from both prediction and observation when
    available. Small/empty masks and badly mismatched initial shapes cannot validate.
    """
    scores=[]
    for frame,observed in sorted(masks.items()):
        if frame < 0 or frame >= len(translation): raise ValueError('mask frame is outside trajectory')
        old=render_mask(vertices,faces,rotation[frame],translation[frame],scale,camera)
        new=render_mask(vertices,faces,rotation[frame],translation[frame]+shift,scale,camera)
        observed=np.asarray(observed,bool)
        if old.shape!=observed.shape: raise ValueError('mask resolution does not match camera')
        visible=np.ones_like(old) if not occluders or frame not in occluders else ~np.asarray(occluders[frame],bool)
        observed=observed & visible; old &= visible; new &= visible
        if observed.sum()<16: continue
        iou=lambda predicted: float((predicted & observed).sum()/max((predicted | observed).sum(),1))
        scores.append((iou(old),iou(new)))
    if len(scores)<8: return dict(accepted=False,reason='insufficient independent image frames',frames=len(scores))
    scores=np.asarray(scores); before,after=np.median(scores,axis=0)
    improvement=scores[:,1]-scores[:,0]
    accepted=bool(before>=.3 and after-before>=.005 and np.mean(improvement>=-.01)>=.8)
    return dict(accepted=accepted,frames=len(scores),iou_before=float(before),iou_after=float(after),
                nonregressing_fraction=float(np.mean(improvement>=-.01)))


def process_directory(base, contacts, output, *, image_validator=None):
    """Write a separate NPZ/GLB candidate, applying only independently gated proposals.

    contacts/<tag>.npz must contain hands [T,P,3] and contact [T,P]. Missing
    contact evidence leaves an episode unchanged. No validator means report-only.
    A validator receives (tag, episode_dict, mesh, offset) and returns a JSON dict
    with accepted=True only when image evidence supports that exact offset.
    """
    import trimesh
    base,contacts,output=map(Path,(base,contacts,output))
    if output.exists() or output.resolve().is_relative_to(base.resolve()):
        raise ValueError('output must be a new directory outside the input directory')
    paths=sorted(base.glob('episode_??????.npz'))
    if not paths: raise ValueError('input contains no episodes')
    # Validate dependencies before any files are written.
    for path in paths:
        if not (base/f'{path.stem}_object.glb').is_file():
            raise ValueError(f'missing object mesh for {path.stem}')
    shutil.copytree(base,output)
    reports={}
    for path in paths:
        tag=path.stem; evidence=contacts/path.name
        if not evidence.is_file():
            reports[tag]=dict(applied=False,reason='missing independent contact evidence'); continue
        with np.load(path,allow_pickle=False) as archive: data=dict(archive)
        with np.load(evidence,allow_pickle=False) as archive: obs=dict(archive)
        mesh=trimesh.load(base/f'{tag}_object.glb',force='mesh',process=False)
        report=fit_translation(mesh.vertices,mesh.vertex_normals,data['object_rotation'],
                               data['object_translation'],float(data['object_scale']),obs['hands'],obs['contact'])
        report['input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        report['contacts_sha256']=hashlib.sha256(evidence.read_bytes()).hexdigest()
        report['applied']=False
        if report['accepted'] and image_validator is not None:
            gate=image_validator(tag,data,mesh,np.asarray(report['translation_m']))
            report['image_gate']=gate
            if gate.get('accepted') is True:
                old=data['object_translation']
                updated=(old.astype(float)+report['translation_m']).astype(old.dtype)
                report['acceleration_change_m']=float(np.max(np.abs(np.diff(updated.astype(float),n=2,axis=0)-
                                                                          np.diff(old.astype(float),n=2,axis=0)))) if len(old)>2 else 0.
                if report['acceleration_change_m'] < 1e-5:
                    data['object_translation']=updated
                    np.savez(output/path.name,**data); report['applied']=True
        reports[tag]=report
    (output/'constant_contact_report.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--contacts',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--camera-dir',type=Path)
    parser.add_argument('--mask-root',type=Path)
    parser.add_argument('--person-root',type=Path)
    parser.add_argument('--apply',action='store_true',help='apply only proposals passing independent silhouettes')
    args=parser.parse_args()
    validator=None
    if args.apply:
        if not args.camera_dir or not args.mask_root or not args.person_root:
            parser.error('--apply requires --camera-dir, --mask-root and --person-root')
        def validator(tag,data,mesh,offset):
            from PIL import Image
            K=json.loads((args.camera_dir/tag/'intrinsics.json').read_text())
            masks={}; occluders={}
            # Odd frames are reserved for image checks; contact construction should
            # use even frames to avoid selecting the same observations twice.
            frames=np.unique(np.linspace(1,len(data['pose'])-1,32).astype(int)//2*2+1)
            for frame in frames:
                path=args.mask_root/tag/'sam2_masks/1'/f'{frame:06d}.png'
                person=args.person_root/tag/'sam2_masks/0'/f'{frame:06d}.png'
                if frame>=len(data['pose']) or not path.is_file() or not person.is_file(): continue
                size=(int(K['width'])//4,int(K['height'])//4)
                with Image.open(path) as im: masks[int(frame)]=np.asarray(im.resize(size,Image.Resampling.NEAREST))>127
                with Image.open(person) as im: occluders[int(frame)]=np.asarray(im.resize(size,Image.Resampling.NEAREST))>127
            return silhouette_gate(mesh.vertices,mesh.faces,data['object_rotation'],data['object_translation'],
                                   float(data['object_scale']),offset,K,masks,occluders)
    reports=process_directory(args.input,args.contacts,args.out,image_validator=validator)
    print(json.dumps({'episodes':len(reports),'proposals':sum(r.get('accepted',False) for r in reports.values()),
                      'applied':sum(r['applied'] for r in reports.values()),
                      'mode':'image-gated' if args.apply else 'report-only; independent image gate required'}))


if __name__=='__main__':
    main()
