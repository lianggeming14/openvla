"""Create a NEW private repository and push already committed source only."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def run(*command, check=True):
    return subprocess.run(command, text=True, capture_output=True, check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    if not shutil.which('gh') or not shutil.which('git'):
        raise RuntimeError('Install GitHub CLI and Git and authenticate gh first')
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise RuntimeError('Run this command from the repository root')
    login = run('gh', 'api', 'user', '--jq', '.login').stdout.strip()
    if login.lower() != args.owner.lower():
        raise RuntimeError(f'Logged in as {login}; requested owner is {args.owner}')
    if run('git','status','--porcelain').stdout.strip():
        raise RuntimeError('Review and commit source changes before publishing')
    run('git','rev-parse','HEAD')
    if run('git','remote').stdout.strip():
        raise RuntimeError('A remote already exists; refusing to modify it')
    # Authenticated repo listing provides a positive preflight; network errors abort.
    import json
    listing = run('gh','api','--paginate','--slurp','user/repos?per_page=100&affiliation=owner')
    pages = json.loads(listing.stdout)
    target = f'{args.owner}/openvla'
    if any(repo['full_name'].lower() == target.lower() for page in pages for repo in page):
        raise RuntimeError(f'{target} already exists; refusing to overwrite')
    created = run('gh','repo','create',target,'--private','--source','.', '--remote','origin','--push')
    print(created.stdout)
    print(run('gh','repo','view',target,'--json','url,visibility').stdout)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f'Publication failed: {exc}',file=sys.stderr)
        raise SystemExit(1)
