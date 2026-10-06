"""Pinned public LPIPS dependency, verified offline without importing Torch."""
from pathlib import Path
import stat

from blake3 import blake3


ALEXNET_URL = "https://service.example.invalid/models/alexnet-owt-7be5be79.pth"
ALEXNET_RELATIVE = "hub/checkpoints/alexnet-owt-7be5be79.pth"
ALEXNET_BYTES = 244_408_911
ALEXNET_BLAKE3 = "f244ce4ed7579e1b514fa6eb68d742e7a1188767872f860e11d804b0a87ff739"
# Upstream torchvision's published filename carries the first eight digits;
# the full download checksum was independently observed before pinning BLAKE3.
ALEXNET_UPSTREAM_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def verify_lpips_cache(torch_home: Path) -> dict:
    """Allow only the exact published AlexNet artifact; never fetch at scoring."""
    root = Path(torch_home).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("lpips_cache_directory_invalid")
    path = root
    for part in Path(ALEXNET_RELATIVE).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("lpips_cache_symlink_forbidden")
    info = path.stat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_size != ALEXNET_BYTES or info.st_mode & 0o222):
        raise ValueError("lpips_cache_file_metadata_differs")
    digest = blake3()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ALEXNET_BLAKE3:
        raise ValueError("lpips_cache_file_digest_differs")
    return {"schema": "eva.automedbench-lpips-cache.v1", "torch_home": str(root.resolve()),
            "relative_path": ALEXNET_RELATIVE, "bytes": ALEXNET_BYTES, "blake3": ALEXNET_BLAKE3,
            "upstream_url": ALEXNET_URL, "device": "cpu", "network_downloads_allowed": False}
