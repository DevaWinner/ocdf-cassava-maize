# ---- OCDF F2 inventory helpers (Notebook 01) ----
import hashlib, io, os, csv, json
import numpy as np
from PIL import Image, ImageOps

WORKING_MAX_SIDE = 512        # working copies: longest side <= 512 px, aspect kept
WORKING_JPEG_QUALITY = 95
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_DCT32 = None
def _dct_matrix(n=32):
    # Unnormalised DCT-II, identical to scipy.fftpack.dct (as used by imagehash.phash)
    k = np.arange(n)[:, None]; i = np.arange(n)[None, :]
    return 2.0 * np.cos(np.pi * (2 * i + 1) * k / (2 * n))

def phash_hex(img):
    """64-bit DCT perceptual hash (same method as imagehash.phash, hash_size=8)."""
    global _DCT32
    if _DCT32 is None:
        _DCT32 = _dct_matrix(32)
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = _DCT32 @ g @ _DCT32.T
    low = d[:8, :8]
    bits = (low > np.median(low)).flatten()
    return "%016x" % int("".join("1" if b else "0" for b in bits), 2)

def hamming(h1, h2):
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def inspect_image(raw_bytes, working_out=None):
    """Decode one original image; return inventory fields. Optionally write a working copy.

    The original bytes are never modified. The working copy is EXIF-orientation corrected,
    RGB, longest side <= WORKING_MAX_SIDE, saved as JPEG.
    """
    rec = {"sha256": sha256_bytes(raw_bytes), "bytes": len(raw_bytes), "decode_ok": False,
           "format": "", "width": "", "height": "", "mode": "", "phash": "", "decode_error": "",
           "working_path": "", "working_width": "", "working_height": ""}
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            rec["format"] = im.format or ""
            rec["width"], rec["height"] = im.size
            rec["mode"] = im.mode
            im.load()                                   # full decode = integrity check
            im = ImageOps.exif_transpose(im)
            rgb = im.convert("RGB")
        rec["phash"] = phash_hex(rgb)
        rec["decode_ok"] = True
        if working_out:
            w = rgb.copy()
            w.thumbnail((WORKING_MAX_SIDE, WORKING_MAX_SIDE), Image.LANCZOS)
            os.makedirs(os.path.dirname(working_out), exist_ok=True)
            w.save(working_out, "JPEG", quality=WORKING_JPEG_QUALITY)
            rec["working_path"] = working_out
            rec["working_width"], rec["working_height"] = w.size
    except Exception as e:                              # corrupt / unreadable file
        rec["decode_error"] = f"{type(e).__name__}: {e}"[:200]
    return rec

INVENTORY_FIELDS = ["source_id", "release", "original_relpath", "original_label", "research_label",
                    "crop", "selected", "exclusion_reason", "sha256", "bytes", "decode_ok", "decode_error",
                    "format", "width", "height", "mode", "phash", "working_path", "working_width",
                    "working_height", "group_hint", "published_checksum", "checksum_match"]

def write_inventory(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def count_table(rows):
    """Per source x original label: downloaded, readable, selected (research label), excluded."""
    from collections import defaultdict
    t = defaultdict(lambda: {"downloaded": 0, "readable": 0, "unreadable": 0, "selected": 0, "excluded": 0})
    for r in rows:
        k = (r["source_id"], r["original_label"], r["research_label"] or "(excluded)")
        t[k]["downloaded"] += 1
        ok = str(r["decode_ok"]) == "True"
        t[k]["readable" if ok else "unreadable"] += 1
        sel = str(r["selected"]) == "True"          # CSV round-trip turns booleans into text
        t[k]["selected" if (sel and ok) else "excluded"] += 1
    return [dict(source_id=k[0], original_label=k[1], research_label=k[2], **v) for k, v in sorted(t.items())]
