"""Download the exact online test video with SHA256 verification (stdlib only)."""
from pathlib import Path
import hashlib
import urllib.request

URL = 'https://raw.githubusercontent.com/abdullahtarek/tennis_analysis/main/input_videos/input_video.mp4'
EXPECTED_SHA256 = '5f2584a2010864d5d8624c325d7f0571d48bd0756164657ef603dc8a97bab22d'


def main():
    target = Path(__file__).resolve().parents[1] / 'data/online/tennis.mp4'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED_SHA256:
            print('Video already present:', target)
            return
        raise RuntimeError('Existing video hash mismatch; move it aside before retrying')
    temporary = target.with_suffix('.download')
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
            while chunk := response.read(1024*1024):
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED_SHA256:
            raise RuntimeError('Downloaded video hash mismatch; upstream may have changed')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('Downloaded:', target)


if __name__ == '__main__':
    main()
