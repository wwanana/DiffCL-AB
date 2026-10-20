"""Restore checksummed data and model files from portable compressed Git assets."""
from __future__ import annotations
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def safe_path(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Asset path leaves repository')
    return p

def prepare(root=ROOT):
    root=Path(root);manifest=json.loads((root/'assets/manifest.json').read_text());restored=[]
    for entry in manifest['files']:
        target=safe_path(root,entry['target'])
        if target.exists():
            if digest(target)!=entry['sha256']:raise ValueError(f'Existing file differs from the deposited asset: {entry["target"]}')
            continue
        pieces=[]
        for part in entry['parts']:
            path=safe_path(root,part['path'])
            if digest(path)!=part['sha256']:raise ValueError(f'Corrupt asset part: {part["path"]}')
            payload=path.read_bytes()
            if part.get('encoding')=='base64':payload=base64.b64decode(payload,validate=True)
            pieces.append(payload)
        content=gzip.decompress(b''.join(pieces))
        if len(content)!=entry['size'] or hashlib.sha256(content).hexdigest()!=entry['sha256']:
            raise ValueError(f'Corrupt reconstructed file: {entry["target"]}')
        target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.assembling')
        temporary.write_bytes(content);temporary.replace(target);restored.append(entry['target'])
    return restored

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    restored=prepare();print(f'Assets verified; restored {len(restored)} files')
