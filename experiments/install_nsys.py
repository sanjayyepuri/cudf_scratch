"""Extract a checksum-pinned NVIDIA Nsight Systems package locally; no root needed."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import urllib.request

VERSION = '2026.3.2'
PACKAGE = 'nsight-systems-2026.3.2_2026.3.2.476-1_amd64.deb'
URL = 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/' + PACKAGE
SHA256 = 'e9bf0ff9c613ff5545095a4177676a9579cb666e78c8621c3792f3559e2c1690'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=Path('/tmp/cudf-profiling/nsight'))
    parser.add_argument('--cache', type=Path, default=Path('/tmp/cudf-profiling'))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    archive = args.cache / PACKAGE
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        with urllib.request.urlopen(URL) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError(f'Checksum mismatch: remove {archive} and retry')
    executable = args.prefix / f'opt/nvidia/nsight-systems/{VERSION}/bin/nsys'
    marker = args.prefix / '.package-sha256'
    if not marker.exists() or marker.read_text().strip() != SHA256:
        args.prefix.mkdir(parents=True, exist_ok=True)
        subprocess.run(['dpkg-deb', '-x', str(archive), str(args.prefix)], check=True)
        marker.write_text(SHA256 + '\n')
    subprocess.run([str(executable), '--version'], check=True)
    print(f'Executable: {executable.resolve()}')
    print('Only the local profiler is extracted; system packages and GPU drivers are unchanged.')


if __name__ == '__main__':
    main()
