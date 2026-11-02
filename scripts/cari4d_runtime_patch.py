#!/usr/bin/env python3
"""Create a narrow, version-checked CARI4D exporter patch for container binding.

Silhouette centroid distance is undefined when either visible silhouette is empty.
The selection algorithm uses infinity to reject that observation; the exported
diagnostics schema instead uses NaN for an undefined measurement. Convert only at
the export boundary, preserving the selector, poses, reliability and all validators.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def patch_foundationpose_export(source: str) -> str:
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_selected_metric_values']
    if len(functions)!=1:
        raise ValueError('unrecognized upstream exporter: missing selected-metric function')
    function=functions[0]
    lines=source.splitlines(keepends=True)
    text=''.join(lines[function.lineno-1:function.end_lineno])
    old='float(metrics.centroid_distance[index])'
    if text.count(old)!=1 or 'metrics.observed_visible_pixels' not in text or 'metrics.rendered_visible_pixels[index]' not in text:
        raise ValueError('unrecognized upstream selected-metric implementation')
    new='(float(metrics.centroid_distance[index]) if int(metrics.observed_visible_pixels) > 0 and int(metrics.rendered_visible_pixels[index]) > 0 else float("nan"))'
    lines[function.lineno-1:function.end_lineno]=[text.replace(old,new)]
    patched=''.join(lines)
    ast.parse(patched)
    return patched


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    original_bytes=args.source.read_bytes()
    original=original_bytes.decode('utf-8')
    patched=patch_foundationpose_export(original)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(patched)
    manifest={'source':str(args.source),'source_sha256':hashlib.sha256(original_bytes).hexdigest(),
              'patched_sha256':hashlib.sha256(patched.encode()).hexdigest(),
              'change':'undefined visible-centroid diagnostic -> NaN; selection and geometry untouched'}
    args.out.with_suffix('.patch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__': main()
