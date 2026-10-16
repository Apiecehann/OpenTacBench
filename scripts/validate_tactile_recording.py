"""Decode every RGB image and check frame/numeric integrity in a recording."""
import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def audit_recording(path):
    errors=[]
    streams={}
    numeric=0
    with h5py.File(path,'r') as data:
        clock='physics_step' if 'physics_step' in data else 'step'
        steps=data[clock][:]
        frames=len(steps)
        if frames<2 or np.any(np.diff(steps)<=0):errors.append('invalid observation clock')
        def check(name,field):
            nonlocal numeric
            if not isinstance(field,h5py.Dataset):return
            if not field.shape or field.shape[0]!=frames:
                errors.append('inconsistent frame count: '+name)
                return
            if field.dtype.kind=='f':
                numeric+=1
                for start in range(0,frames,64):
                    if not np.isfinite(field[start:start+64]).all():
                        errors.append('nonfinite values: '+name)
                        break
            if not name.endswith(('/rgb','/rgb_marker')):return
            raw=field.ndim==4
            expected=None
            decoded=nonempty=0
            try:
                for value in field:
                    if raw:
                        image=np.asarray(value)
                        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3:
                            raise ValueError('expected uint8 HWC RGB')
                    else:
                        with Image.open(io.BytesIO(value.tobytes())) as opened:
                            opened.load()  # Decode pixels, including the JPEG scan.
                            if opened.mode!='RGB':raise ValueError('expected encoded RGB')
                            image=np.asarray(opened)
                    if expected is None:expected=image.shape
                    if image.shape!=expected:raise ValueError('image size changes within stream')
                    nonempty+=int(np.ptp(image,axis=(0,1)).max()>0)
                    decoded+=1
            except (OSError,ValueError) as error:
                errors.append(f'{name}[{decoded}]: {error}')
            if not nonempty:errors.append('entire RGB stream is empty: '+name)
            streams[name]=dict(shape=expected,decoded_images=decoded,nonempty_images=nonempty)
        data.visititems(check)
        if not streams:errors.append('no RGB image stream')
    return dict(valid=not errors,frames=frames,numeric_fields=numeric,
        decoded_images=sum(s['decoded_images'] for s in streams.values()),streams=streams,errors=errors)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',nargs='+',type=Path)
    args=parser.parse_args()
    reports=[dict(episode=str(path),**audit_recording(path)) for path in args.episodes]
    print(json.dumps(reports,indent=2))
    raise SystemExit(0 if all(r['valid'] for r in reports) else 1)
