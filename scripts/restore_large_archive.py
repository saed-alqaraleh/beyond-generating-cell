#!/usr/bin/env python3
"""Reconstruct the exact frozen score archive from verified parts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'data/large_archive_parts/manifest.json'
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
    d=json.loads(MANIFEST.read_text());dest=ROOT/d['target']
    if dest.is_file():
        b=dest.read_bytes()
        if len(b)==d['bytes'] and digest(b)==d['sha256']:
            print('Frozen score archive verified:',dest.relative_to(ROOT));return
        raise ValueError('An existing frozen score archive failed checksum verification')
    blocks=[]
    for item in d['parts']:
        p=ROOT/d['parts_dir']/item['filename'];b=p.read_bytes()
        if len(b)!=item['bytes'] or digest(b)!=item['sha256']:
            raise ValueError(f'Archive part failed verification: {p}')
        blocks.append(b)
    restored=b''.join(blocks)
    if len(restored)!=d['bytes'] or digest(restored)!=d['sha256']:
        raise ValueError('Reconstructed score archive differs from expected SHA-256')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(restored)
    print('Restored frozen score archive:',dest.relative_to(ROOT))
if __name__=='__main__':main()
