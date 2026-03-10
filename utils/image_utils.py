"""Shared utilities for image handling."""
import base64
import os

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def load_image(path: str) -> str:
    """Load an image file and return it as a base64 data URL for API consumption."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lower()
    mime = MIME_TYPES.get(ext, "image/jpeg")
    return f"data:{mime};base64,{data}"
