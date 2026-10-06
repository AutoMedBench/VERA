"""Retrieve the externally maintained pinned legacy skills into private host state."""
import argparse
import os
from pathlib import Path
import urllib.parse
import urllib.request
from .integrity import byte_digest, digest, relative_path, strict_json, write_json

REVISION = '79dd2a31f5f3e5018608bf20f982ad70d2e5dafa'
REPOSITORY = 'operator/rlevo-Med-RL-data'


def prepare(manifest_path, output):
    manifest = strict_json(Path(manifest_path).read_bytes())
    if manifest['manifest_blake3'] != digest({k: v for k, v in manifest.items() if k != 'manifest_blake3'}):
        raise ValueError('legacy_manifest_commitment')
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True, mode=0o700)
    required = {'README.md': (None, manifest['source_readme_blake3'])}
    catalogs = set()
    for skill in manifest['skills']:
        for path in skill['source_paths']:
            required[path] = (skill['bytes'], skill['content_blake3'])
        catalogs.update(skill['catalog_paths'])
    for path in catalogs: required.setdefault(path, (None, None))
    token = os.environ.get('HF_TOKEN')
    files = []
    for name, (size, expected) in sorted(required.items()):
        relative_path(name)
        path = root / name
        if path.is_file():
            raw = path.read_bytes()
        else:
            url = 'https://huggingface.co/datasets/' + REPOSITORY + '/resolve/' + REVISION + '/' + urllib.parse.quote(name, safe='/')
            headers = {'Authorization': 'Bearer ' + token} if token else {}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                raw = response.read(1024 * 1024)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        if (size is not None and len(raw) != size) or (expected is not None and byte_digest(raw) != expected):
            raise ValueError('legacy_skill_source_commitment')
        files.append({'path': name, 'bytes': len(raw), 'blake3': byte_digest(raw), 'canonical_content_commitment_present': expected is not None})
    receipt = {'schema': 'eva.medresearch-external-legacy-skills.v1', 'repository': REPOSITORY, 'revision': REVISION,
        'manifest_blake3': manifest['manifest_blake3'], 'redistribution': manifest['redistribution'],
        'unique_contents': manifest['unique_content_count'], 'occurrences': manifest['occurrence_count'],
        'files': files, 'credentials_saved': False}
    write_json(root / 'download-receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    receipt = prepare(args.manifest, args.output)
    print('Verified external skill occurrences:', receipt['occurrences'])
