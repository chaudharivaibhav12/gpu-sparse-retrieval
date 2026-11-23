"""Validate before narrowing; preserve exact uint8 impacts, including zero/255."""
import numpy as np

def checked_u8(values):
    a=np.asarray(values)
    if a.dtype.kind not in 'uif' or not np.isfinite(a).all():
        raise ValueError('Document impacts must be finite numeric integers')
    if np.any(a<0) or np.any(a>255) or np.any(a!=np.floor(a)):
        raise ValueError('Document impacts outside exact uint8 domain')
    return a.astype(np.uint8,copy=False)
