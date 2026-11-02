"""Data identity contracts for reused MHR joint caches."""
import hashlib
import numpy as np


def motion_fingerprint(data):
    digest=hashlib.sha256()
    for key in ('pose','scales','shape'):
        value=np.ascontiguousarray(data[key])
        digest.update(f'{key}:{value.dtype.str}:{value.shape}\n'.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def require_motion_match(cache,data):
    if 'source_motion_sha256' not in cache:
        raise ValueError('joint cache has no source motion provenance; regenerate it')
    if str(cache['source_motion_sha256'].item())!=motion_fingerprint(data):
        raise ValueError('joint cache belongs to a different source motion candidate')
