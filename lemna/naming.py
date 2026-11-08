"""Parsing of photo file names such as G1_1_Gun0.jpg (group, replicate, day)."""

import re
import unicodedata
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Accepted forms: G1_1_Gun0.jpg, G1-1-Gün0.jpg, g1_1_gun_0.JPG, G1_1_Day0.png, G1_1_D0.jpg
_PATTERN = re.compile(
    r"^(?P<group>G\d+)[_\-\s]+(?P<rep>\d+)[_\-\s]+(?:gun|day|d)?[_\-\s]*(?P<day>\d+)$",
    re.IGNORECASE,
)


def _ascii_fold(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("ı", "i").replace("İ", "I")
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def parse_name(path):
    """Return (group, replicate, day) or None if the name does not match."""
    stem = _ascii_fold(Path(path).stem).strip()
    match = _PATTERN.match(stem)
    if not match:
        return None
    return match.group("group").upper(), int(match.group("rep")), int(match.group("day"))


def make_name(group, rep, day, ext=".jpg"):
    return f"{group.upper()}_{int(rep)}_Gun{int(day)}{ext.lower()}"


def list_images(folder):
    folder = Path(folder)
    files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
    return sorted(files)
