"""Upload a validated episode with the official ModelScope Hub API."""
import argparse
import hashlib
import json
from pathlib import Path


PATTERNS = ('hdf5/*.hdf5', 'tactile.h5', 'metadata.json', 'result.json',
            'config.yml', 'piano_steps.jsonl', 'piano_steps/*.jsonl', 'checksums.json',
            'work/*/trace.jsonl', 'work/*/public_actions_*.jsonl')


def episode_files(source):
    source = Path(source).resolve()
    result = json.loads((source/'result.json').read_text())
    success = result.get('result') == 'success' or result.get('success') is True
    validation = result.get('validation', result.get('data_validation', {}))
    if not success or validation.get('valid') is not True:
        raise ValueError('Only successful, validated episodes may be uploaded')
    files = sorted({p for pattern in PATTERNS for p in source.glob(pattern)
                    if p.is_file() and p.name != 'checksums.json'})
    if not any(p.suffix in ('.h5', '.hdf5') for p in files):
        raise ValueError('Validated episode has no HDF5 data')
    return files


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def upload_episode(source, repo_id, path_in_repo, *, roundtrip=False):
    from modelscope_hub import HubApi
    source = Path(source).resolve()
    files = episode_files(source)
    checks = {str(p.relative_to(source)):dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in files}
    (source/'checksums.json').write_text(json.dumps(checks, indent=2))
    api = HubApi()
    api.upload_folder(repo_id, 'dataset', source, path_in_repo=path_in_repo,
        allow_patterns=list(PATTERNS), max_workers=2, disable_tqdm=True,
        commit_message='Add validated tactile episode '+path_in_repo)
    if roundtrip:
        name = next(name for name in checks if name.endswith(('.h5','.hdf5')))
        downloaded = api.download_file(repo_id, 'dataset', path_in_repo+'/'+name,
            local_dir=source/'.upload_readback', force=True, expected_sha256=checks[name]['sha256'])
        if sha256(downloaded) != checks[name]['sha256']:
            raise ValueError('Uploaded HDF5 failed independent download/hash verification')
        downloaded.unlink()
    # Small-file readback also verifies the final repository commit, including
    # names and hashes, without downloading every large episode a second time.
    downloaded = api.download_file(repo_id, 'dataset', path_in_repo+'/checksums.json',
        local_dir=source/'.upload_readback', force=True)
    if json.loads(downloaded.read_text()) != checks:
        raise ValueError('Remote checksum manifest differs from local episode')
    downloaded.unlink()
    return dict(repo_id=repo_id, path=path_in_repo, files=checks, roundtrip=roundtrip)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('repo_id')
    parser.add_argument('path_in_repo')
    parser.add_argument('--roundtrip', action='store_true')
    args = parser.parse_args()
    result = upload_episode(args.source, args.repo_id, args.path_in_repo, roundtrip=args.roundtrip)
    print('UPLOAD_PASS '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
