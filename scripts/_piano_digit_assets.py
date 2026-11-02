"""Fetch original Meta DIGIT CAD/background, pinned and checksum verified."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

REVISION='a21d0c4626d74a546d94859226c2fea348babb6a'
ASSETS={
    'meshes/digit.STL':'7ea789c949a371d68b2e8b33cc1c999bc495680b15c5f84a3765aec5f96c9f06',
    'examples/conf/bg_digit_240_320.jpg':'b386c851500f063369b2c2e43d8308da0a35736c3d922e1b3c6d8b0f33b91631',
}


def fetch_json(url):
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'OpenTacBench'})
            with urllib.request.urlopen(request,timeout=30) as response:
                return json.load(response)
        except OSError:
            if attempt==2: raise
            time.sleep(1)


def main(root):
    for relative,digest in ASSETS.items():
        path=root/relative
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest: continue
        info=fetch_json(f'https://api.github.com/repos/facebookresearch/tacto/contents/{relative}?ref={REVISION}')
        if not info.get('content'): info=fetch_json(info['git_url'])
        raw=base64.b64decode(info['content'])
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise RuntimeError(f'Original DIGIT asset checksum mismatch: {relative}')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
        print(f'Installed original DIGIT asset: {relative}',flush=True)


if __name__=='__main__': main(Path(sys.argv[1]))
