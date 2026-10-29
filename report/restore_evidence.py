"""Restore the published observations without overwriting local experiments."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore(root=ROOT):
    archive = root / 'evidence/phase_b.zip'
    manifest = json.loads((root / 'evidence/manifest.json').read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Published evidence archive checksum differs')
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != set(manifest['files']):
            raise ValueError('Published evidence inventory differs')
        for name, expected in manifest['files'].items():
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError('Evidence member escapes the repository')
            data = bundle.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Evidence checksum differs: ' + name)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                    raise ValueError('Existing observations differ; use a fresh clone: ' + name)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    return manifest


if __name__ == '__main__':
    restored = restore()
    print(f"Published evidence verified: {len(restored['files'])} files")
