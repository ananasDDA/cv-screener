"""Photo lookup with a deterministic placeholder so rendering never blocks on image generation."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..config import PHOTOS_DIR

_SIZE = 512
_PALETTE = ["#4F6D7A", "#7A5C61", "#5B6E4F", "#6B5B95", "#8A6D3B", "#3B6E8A", "#7A4F4F", "#4F7A6A"]


def find_photo(candidate_id: str, photos_dir: Path = PHOTOS_DIR) -> Path | None:
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = photos_dir / f"{candidate_id}.{ext}"
        if p.exists():
            return p
    return None


def placeholder(full_name: str, candidate_id: str) -> Image.Image:
    """Flat colored square with initials; color and initials are derived from the candidate."""
    digest = hashlib.sha256(candidate_id.encode()).digest()
    color = _PALETTE[digest[0] % len(_PALETTE)]
    initials = "".join(part[0] for part in full_name.split()[:2]).upper()
    img = Image.new("RGB", (_SIZE, _SIZE), color)
    draw = ImageDraw.Draw(img)
    font = _font(int(_SIZE * 0.42))
    box = draw.textbbox((0, 0), initials, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.text(
        ((_SIZE - w) / 2 - box[0], (_SIZE - h) / 2 - box[1]), initials, fill="white", font=font
    )
    return img


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("Helvetica.ttc", "Arial.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        for folder in (
            "/System/Library/Fonts",
            "/Library/Fonts",
            "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts/truetype/liberation",
        ):
            p = Path(folder) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def photo_data_uri(candidate_id: str, full_name: str, photos_dir: Path = PHOTOS_DIR) -> str:
    """Base64 data URI for the template: the real photo if present, else a placeholder."""
    path = find_photo(candidate_id, photos_dir)
    if path:
        mime = (
            "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else f"image/{path.suffix[1:]}"
        )
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    buf = BytesIO()
    placeholder(full_name, candidate_id).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
