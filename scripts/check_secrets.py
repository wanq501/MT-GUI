"""Scan the repository for API keys, tokens and server credentials before every push.
Usage: python scripts/check_secrets.py   (exit code 1 if anything is found)"""
import os, re, sys, gzip
PAT = re.compile(rb'(sk-or-v1-[A-Za-z0-9]{16,}|sk-ant-[A-Za-z0-9_-]{16,}|sk-proj-[A-Za-z0-9_-]{16,}|\bsk-[A-Za-z0-9]{32,}|AIza[0-9A-Za-z_-]{30,}'
                 rb'|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|\bhf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._-]{24,})')
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); found = 0
for d, dirs, files in os.walk(root):
    dirs[:] = [x for x in dirs if x != '.git']
    for f in files:
        p = os.path.join(d, f); data = open(p, 'rb').read()
        if p.endswith('.gz'): data = gzip.decompress(data)
        for m in PAT.finditer(data):
            found += 1; print(f'{os.path.relpath(p, root)}: {m.group(0)[:10].decode("latin1")}...')
print('no secrets found' if not found else f'{found} secret-like strings found, remove them before pushing'); sys.exit(1 if found else 0)
