#!/usr/bin/env python3
"""Build static preview pages for composition datasets.

Reads source datasets from /mnt/.../composition and writes only into this repo's docs/ tree.
"""
from __future__ import annotations

import ast
import base64
import binascii
import csv
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

csv.field_size_limit(1024 * 1024 * 64)

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SOURCE_ROOT = Path("/mnt/vlm-ks3/FoundationModel/dataset/pose_intriduct/composition")
DATA_DIR = DOCS / "data"
DATASET_DIR = DOCS / "datasets"
ASSET_DIR = DOCS / "assets"

MAX_SAMPLES = 6
READ_AHEAD_ROWS = 80
MAX_FIELD_CHARS = 5000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_CANDIDATE_FILES = 120
MAX_COUNTED_FILES = 20000
MAX_IMAGE_INDEX_FILES = 20000
TEXT_EXTS = {".json", ".jsonl", ".csv", ".tsv", ".txt"}
STRUCTURED_EXTS = TEXT_EXTS | {".parquet"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SKIP_DIRS = {".git", "__pycache__", ".cache"}
SKIP_FILE_NAMES = {"readme", "license", "summary", "eval_summary"}


def slugify(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if text:
        return f"composition-{text}"
    return "composition-" + re.sub(r"-+", "-", name.encode("punycode").decode("ascii").lower()).strip("-")


def find_limited(root: Path, patterns: list[str], limit: int) -> list[Path]:
    cmd = ["find", str(root)]
    for skip in SKIP_DIRS:
        cmd.extend(["-path", f"*/{skip}/*", "-prune", "-o"])
    cmd.append("-type")
    cmd.append("f")
    if patterns:
        cmd.append("(")
        for i, pattern in enumerate(patterns):
            if i:
                cmd.append("-o")
            cmd.extend(["-iname", pattern])
        cmd.append(")")
    cmd.extend(["-print", "-quit"] if limit == 1 else ["-print"])
    try:
        result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=90)
    except subprocess.TimeoutExpired:
        return []
    rows = []
    for line in result.stdout.splitlines():
        rows.append(Path(line))
        if len(rows) >= limit:
            break
    return rows


def find_candidates(root: Path) -> list[Path]:
    groups = [
        ["*.jsonl", "*.parquet", "*.csv", "*.tsv"],
        ["*label*.json", "*annotation*.json", "*train*.json", "*test*.json", "*val*.json", "*.json"],
        ["*label*.txt", "*score*.txt", "*.txt"],
    ]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for patterns in groups:
        for p in find_limited(root, patterns, MAX_CANDIDATE_FILES):
            if p not in seen:
                candidates.append(p)
                seen.add(p)
            if len(candidates) >= MAX_CANDIDATE_FILES:
                return candidates
    return candidates


def count_files_limited(root: Path) -> int:
    files = find_limited(root, [], MAX_COUNTED_FILES)
    return len(files) if len(files) < MAX_COUNTED_FILES else MAX_COUNTED_FILES


def build_image_index(root: Path) -> dict[str, Any]:
    patterns = [f"*{ext}" for ext in IMAGE_EXTS]
    index: dict[str, Any] = {}
    for p in find_limited(root, patterns, MAX_IMAGE_INDEX_FILES):
        try:
            if p.stat().st_size > MAX_IMAGE_BYTES:
                continue
        except OSError:
            continue
        keys = {p.name.lower(), p.stem.lower()}
        for key in keys:
            index.setdefault(key, []).append(p)
    return index


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def compact(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    if isinstance(value, (bytes, bytearray)):
        return f"<binary image/data: {len(value)} bytes>"
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + "\n... [truncated]"
    return text


def flatten_fields(value: Any, prefix: str = "", out: list[tuple[str, Any]] | None = None) -> list[tuple[str, Any]]:
    if out is None:
        out = []
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                rendered = compact(v, 1600)
                out.append((key, rendered))
            else:
                out.append((key, v))
    elif isinstance(value, list):
        for i, v in enumerate(value[:12]):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            if isinstance(v, (dict, list)):
                out.append((key, compact(v, 1600)))
            else:
                out.append((key, v))
    else:
        out.append((prefix or "value", value))
    return out


def find_image_strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, str) and (Path(v).suffix.lower() in IMAGE_EXTS or "image" in str(k).lower() or "img" in str(k).lower()):
                found.append(v)
            else:
                found.extend(find_image_strings(v))
    elif isinstance(value, list):
        for item in value:
            found.extend(find_image_strings(item))
    elif isinstance(value, str):
        if Path(value).suffix.lower() in IMAGE_EXTS:
            found.append(value)
        found.extend(re.findall(r"[^\\s,;:'\"()<>]+\\.(?:jpg|jpeg|png|webp|bmp)", value, flags=re.I))
    return found


def find_image_by_name(dataset_root: Path, names: list[str], cache: dict[str, Path | None]) -> Path | None:
    patterns = [name for name in names if name]
    cache_key = "\0".join(patterns)
    if cache_key in cache:
        return cache[cache_key]
    for p in find_limited(dataset_root, patterns, 1):
        try:
            if p.suffix.lower() in IMAGE_EXTS and p.stat().st_size <= MAX_IMAGE_BYTES:
                cache[cache_key] = p
                return p
        except OSError:
            continue
    cache[cache_key] = None
    return None


def token_image_candidates(text: str, allow_numeric: bool) -> list[str]:
    out: list[str] = []
    tokens = re.split(r"[\s,;:=|]+", text.strip())
    for token in tokens[:4]:
        token = token.strip().strip('"\'')
        if Path(token).suffix.lower() in IMAGE_EXTS:
            out.append(token)
        elif allow_numeric and re.fullmatch(r"\d{1,12}", token):
            out.extend([f"{token}{ext}" for ext in IMAGE_EXTS])
    return out


def extra_image_candidates(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k).lower()
            if isinstance(v, list):
                for item in v:
                    out.extend(extra_image_candidates(item))
                continue
            if isinstance(v, dict):
                out.extend(extra_image_candidates(v))
                continue
            text = str(v).strip().strip('"\'')
            if Path(text).suffix.lower() in IMAGE_EXTS:
                out.append(text)
            elif key in {"image_id", "imageid", "img_id", "id"} and re.fullmatch(r"\d{1,12}", text):
                out.extend([f"{text}{ext}" for ext in IMAGE_EXTS])
            elif key in {"text", "line", "images", "image", "img"}:
                out.extend(token_image_candidates(text, allow_numeric=(key in {"text", "line"})))
    elif isinstance(value, str):
        out.extend(token_image_candidates(value, allow_numeric=True))
    return out


def resolve_image(path_text: str, dataset_root: Path, annotation_file: Path, image_index: dict[str, Any]) -> Path | None:
    raw = path_text.strip().strip('"\'')
    if not raw:
        return None
    candidates = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    common_image_dirs = [
        dataset_root,
        dataset_root / "images",
        dataset_root / "image",
        dataset_root / "imgs",
        dataset_root / "images_21904",
        dataset_root / "AVA_dataset" / "image",
        dataset_root / "datasetImages_originalSize",
        dataset_root / "AADB_newtest_originalSize",
        dataset_root / "40K",
        dataset_root / "images" / "EVA_together",
        annotation_file.parent,
        annotation_file.parent.parent / "images",
        annotation_file.parent.parent / "image",
        annotation_file.parent.parent / "imgs",
    ]
    candidates.extend([
        dataset_root / raw,
        annotation_file.parent / raw,
        dataset_root / Path(raw).name,
    ])
    for d in common_image_dirs:
        candidates.append(d / Path(raw).name)
    imgs_dir = dataset_root / "imgs"
    if imgs_dir.exists():
        try:
            for child in list(imgs_dir.iterdir())[:300]:
                if child.is_dir():
                    candidates.append(child / Path(raw).name)
        except OSError:
            pass
    stem = Path(raw).stem
    if stem:
        for base in [annotation_file.parent, dataset_root]:
            for ext in IMAGE_EXTS:
                candidates.append(base / f"{stem}{ext}")
    for c in candidates:
        try:
            if c.exists() and c.is_file() and c.suffix.lower() in IMAGE_EXTS and c.stat().st_size <= MAX_IMAGE_BYTES:
                return c
        except OSError:
            continue
    lookup_names = []
    for key in [Path(raw).name.lower(), Path(raw).stem.lower()]:
        lookup_names.append(key)
        for match in image_index.get(key, []):
            try:
                if match.exists() and match.stat().st_size <= MAX_IMAGE_BYTES:
                    return match
            except OSError:
                continue
    return None


def find_neighbor_image(annotation_file: Path, dataset_root: Path, used: set[Path], image_index: dict[str, Any]) -> Path | None:
    stem = annotation_file.stem
    search_dirs = [annotation_file.parent, annotation_file.parent.parent, annotation_file.parent.parent / "images", annotation_file.parent.parent / "image", annotation_file.parent.parent / "imgs", dataset_root]
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in IMAGE_EXTS:
            for name in [f"{stem}{ext}", f"{stem.upper()}{ext}"]:
                p = d / name
                try:
                    if p.exists() and p not in used and p.stat().st_size <= MAX_IMAGE_BYTES:
                        return p
                except OSError:
                    pass
    for match in image_index.get(stem.lower(), []):
        if match not in used:
            return match
    return None


def copy_image(src: Path, slug: str, index: int) -> dict[str, str] | None:
    out_dir = ASSET_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"s{index}-{src.name}"
    dst = out_dir / out_name
    try:
        shutil.copy2(src, dst)
    except OSError:
        return None
    return {"label": "image", "src": f"../../assets/{slug}/{out_name}", "alt": str(src)}


def image_bytes(value: Any) -> bytes | None:
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    elif isinstance(value, str) and value.startswith("b'"):
        try:
            data = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    elif isinstance(value, str) and len(value) > 200 and re.fullmatch(r"[A-Za-z0-9+/=\n\r]+", value):
        try:
            data = base64.b64decode(value, validate=False)
        except (binascii.Error, ValueError):
            return None
    else:
        return None
    if data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG") or data.startswith(b"RIFF"):
        return data
    return None


def write_image_bytes(data: bytes, slug: str, index: int, label: str) -> dict[str, str] | None:
    out_dir = ASSET_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if data.startswith(b"\xff\xd8\xff") else ".png" if data.startswith(b"\x89PNG") else ".webp"
    safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-") or "image"
    out_name = f"s{index}-{safe_label}{ext}"
    dst = out_dir / out_name
    try:
        dst.write_bytes(data)
    except OSError:
        return None
    return {"label": label, "src": f"../../assets/{slug}/{out_name}", "alt": label}


def embedded_images(value: Any) -> list[tuple[str, bytes]]:
    found: list[tuple[str, bytes]] = []
    if isinstance(value, dict):
        for k, v in value.items():
            data = image_bytes(v)
            if data:
                found.append((str(k), data))
            else:
                found.extend(embedded_images(v))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            data = image_bytes(v)
            if data:
                found.append((f"image_{i}", data))
            else:
                found.extend(embedded_images(v))
    return found


def make_sample(title: str, subtitle: str, annotation_file: Path, dataset_root: Path, value: Any, slug: str, index: int, used_images: set[Path], image_index: dict[str, Any]) -> dict[str, Any]:
    fields = [{"label": "Annotation file", "value": rel(annotation_file, dataset_root)}]
    flat = flatten_fields(value)
    for k, v in flat[:40]:
        fields.append({"label": str(k), "value": compact(v)})
    images = []
    for img_text in (find_image_strings(value) + extra_image_candidates(value))[:16]:
        img_path = resolve_image(img_text, dataset_root, annotation_file, image_index)
        if img_path and img_path not in used_images:
            copied = copy_image(img_path, slug, index)
            if copied:
                images.append(copied)
                used_images.add(img_path)
    if not images:
        for label, data in embedded_images(value)[:4]:
            copied = write_image_bytes(data, slug, index, label)
            if copied:
                images.append(copied)
    if not images:
        neighbor = find_neighbor_image(annotation_file, dataset_root, used_images, image_index)
        if neighbor:
            copied = copy_image(neighbor, slug, index)
            if copied:
                images.append(copied)
                used_images.add(neighbor)
    return {"title": title, "subtitle": subtitle, "text_fields": fields, "images": images}


def read_json_samples(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[:READ_AHEAD_ROWS]
    if isinstance(data, dict):
        for key in ["data", "samples", "annotations", "items", "records", "train", "test", "val"]:
            if isinstance(data.get(key), list):
                return data[key][:READ_AHEAD_ROWS]
        return [data]
    return [data]


def read_jsonl_samples(path: Path) -> list[Any]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(line)
            if len(rows) >= READ_AHEAD_ROWS:
                break
    return rows


def read_table_samples(path: Path) -> list[Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if first_line.count("=") > first_line.count(",") and first_line.count("=") >= 2:
            delimiter = "="
    except (OSError, IndexError):
        pass
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames:
            for row in reader:
                rows.append(dict(row))
                if len(rows) >= READ_AHEAD_ROWS:
                    break
        else:
            f.seek(0)
            raw = csv.reader(f, delimiter=delimiter)
            for row in raw:
                rows.append(row)
                if len(rows) >= READ_AHEAD_ROWS:
                    break
    return rows


def parse_txt_line(path: Path, line: str, line_number: int) -> dict[str, Any]:
    tokens = line.split()
    row: dict[str, Any] = {"line_number": line_number, "raw_text": line}
    if not tokens:
        return row
    first = tokens[0]
    if Path(first).suffix.lower() in IMAGE_EXTS:
        row["image"] = first
        if len(tokens) == 2:
            row["score"] = tokens[1]
        elif len(tokens) > 2:
            row["values"] = tokens[1:]
        return row
    lower_name = path.name.lower()
    if lower_name == "ava.txt" and len(tokens) >= 15:
        row.update({
            "image_id": tokens[0],
            "semantic_tag_id_1": tokens[1],
            "rating_1_count": tokens[2],
            "rating_2_count": tokens[3],
            "rating_3_count": tokens[4],
            "rating_4_count": tokens[5],
            "rating_5_count": tokens[6],
            "rating_6_count": tokens[7],
            "rating_7_count": tokens[8],
            "rating_8_count": tokens[9],
            "rating_9_count": tokens[10],
            "rating_10_count": tokens[11],
            "semantic_tag_id_2": tokens[12],
            "challenge_id": tokens[13],
            "source_id": tokens[14],
        })
        return row
    if all(re.fullmatch(r"-?\d+(?:\.\d+)?", token) for token in tokens):
        row["label_format"] = "numeric_tokens"
        row["image_stem"] = path.stem
        if len(tokens) == 4:
            row["bbox_x1"] = tokens[0]
            row["bbox_y1"] = tokens[1]
            row["bbox_x2"] = tokens[2]
            row["bbox_y2"] = tokens[3]
        elif len(tokens) >= 5:
            row["class_id"] = tokens[0]
            row["bbox_x1"] = tokens[1]
            row["bbox_y1"] = tokens[2]
            row["bbox_x2"] = tokens[3]
            row["bbox_y2"] = tokens[4]
        elif len(tokens) > 1:
            row["values"] = tokens
        return row
    row["text"] = line
    return row


def read_txt_samples(path: Path) -> list[Any]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rows.append(parse_txt_line(path, line, i + 1))
            if len(rows) >= READ_AHEAD_ROWS:
                break
    return rows


def read_parquet_samples(path: Path) -> list[Any]:
    if pq is not None:
        table = pq.read_table(path, use_threads=False)
        rows = table.slice(0, READ_AHEAD_ROWS).to_pylist()
        return rows
    if pd is None:
        return []
    df = pd.read_parquet(path)
    return json.loads(df.head(READ_AHEAD_ROWS).to_json(orient="records", force_ascii=False))


def candidate_score(path: Path) -> tuple[int, int, str]:
    lower = path.name.lower()
    score = 0
    if path.suffix.lower() == ".jsonl": score -= 50
    if path.suffix.lower() == ".parquet": score -= 45
    if path.suffix.lower() in {".csv", ".tsv"}: score -= 40
    if path.suffix.lower() == ".json": score -= 35
    if path.suffix.lower() == ".txt": score -= 20
    rel_lower = str(path).lower()
    if any(name in lower for name in SKIP_FILE_NAMES): score += 100
    if lower in {"styles.txt", "tags.txt", "challenges.txt", "users.csv", "region_index.csv", "image_content_category.csv"}: score += 80
    if lower in {"ava.txt", "votes.csv", "votes_filtered.csv"}: score -= 80
    if lower.endswith("_score.txt") or "giaa" in lower: score -= 35
    if "metadata-" in lower: score -= 20
    if "label" in lower or "annotation" in lower or "train" in lower or "test" in lower or "val" in lower: score -= 10
    if "/images/" in rel_lower and path.suffix.lower() == ".json": score += 30
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return (score, size, str(path))


def build_dataset(dataset_root: Path) -> dict[str, Any] | None:
    name = dataset_root.name
    slug = slugify(name)
    candidates = [p for p in find_candidates(dataset_root) if p.suffix.lower() in STRUCTURED_EXTS]
    candidates.sort(key=candidate_score)
    if not candidates:
        return None
    candidate_samples = []
    warnings = []
    used_images: set[Path] = set()
    image_index: dict[str, Any] = {}
    data_labels = []
    kinds = []
    for path in candidates[:80]:
        if len([s for s in candidate_samples if s.get("images")]) >= MAX_SAMPLES:
            break
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                rows = read_json_samples(path)
            elif suffix == ".jsonl":
                rows = read_jsonl_samples(path)
            elif suffix in {".csv", ".tsv"}:
                rows = read_table_samples(path)
            elif suffix == ".txt":
                rows = read_txt_samples(path)
            elif suffix == ".parquet":
                rows = read_parquet_samples(path)
            else:
                rows = []
        except Exception as exc:
            warnings.append(f"无法读取 {rel(path, dataset_root)}: {exc}")
            continue
        if not rows:
            continue
        if not data_labels:
            data_labels.append(rel(path, dataset_root))
        if suffix.lstrip(".") not in kinds:
            kinds.append(suffix.lstrip("."))
        for row in rows:
            idx = len(candidate_samples)
            sample = make_sample(f"Sample {idx + 1}", rel(path, dataset_root), path, dataset_root, row, slug, idx, used_images, image_index)
            candidate_samples.append(sample)
            if len([s for s in candidate_samples if s.get("images")]) >= MAX_SAMPLES:
                break
    samples = [s for s in candidate_samples if s.get("images")][:MAX_SAMPLES]
    if len(samples) < MAX_SAMPLES:
        samples.extend([s for s in candidate_samples if not s.get("images")][:MAX_SAMPLES - len(samples)])
    for i, sample in enumerate(samples):
        sample["title"] = f"Sample {i + 1}"
    if not samples:
        return None
    total_files = count_files_limited(dataset_root)
    return {
        "name": name,
        "slug": slug,
        "kind": "+".join(kinds) if kinds else candidates[0].suffix.lstrip("."),
        "data_label": data_labels[0] if data_labels else rel(candidates[0], dataset_root),
        "warnings": warnings[:8],
        "samples": samples,
        "total_files": total_files,
        "sample_count": len(samples),
        "source_dir": str(dataset_root),
    }


def page_html(name: str, data_file: str) -> str:
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(name)}</title>
  <link rel="stylesheet" href="../../assets/site.css" />
</head>
<body>
  <header class="site-header">
    <div class="wrap">
      <div>
        <div class="eyebrow">Composition2 数据浏览器</div>
        <h1 id="pageTitle">{html.escape(name)}</h1>
        <p id="pageSubtitle">源数据集预览</p>
      </div>
      <a class="home-link" href="../../index.html">← 总览</a>
    </div>
  </header>
  <main class="wrap">
    <div id="warningArea"></div>
    <section class="panel">
      <div class="panel-head"><div><h2>概览</h2><p id="overviewText"></p></div></div>
      <div id="metaRow" class="meta-row"></div>
    </section>
    <section>
      <div class="section-head"><h2>样本</h2><p>从源数据集中抽取的样本。</p></div>
      <div id="sampleGrid" class="sample-grid"></div>
    </section>
  </main>
  <script>
  const DATA_URL = '../../data/{data_file}';
  const warningArea = document.getElementById('warningArea');
  const overviewText = document.getElementById('overviewText');
  const metaRow = document.getElementById('metaRow');
  const sampleGrid = document.getElementById('sampleGrid');
  function esc(s){{return String(s ?? '').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
  function renderField(f){{return `<div class="kv"><div class="k">${{esc(f.label)}}</div><div class="v">${{esc(f.value)}}</div></div>`;}}
  function renderSample(s, i){{
    const text = (s.text_fields || []).map(renderField).join('') || '<div class="muted">未提取到文本字段。</div>';
    const imgs = (s.images || []).map(img => `<figure class="image-card"><div class="image-label">${{esc(img.label)}}</div><img src="${{esc(img.src)}}" alt="${{esc(img.alt || img.label || '预览')}}" /></figure>`).join('') || '<div class="muted">未提取到可预览图片。</div>';
    return `<article class="card"><div class="card-head"><div><div class="sample-index">样本 ${{i + 1}}</div><h3>${{esc(s.title || '样本')}}</h3><p>${{esc(s.subtitle || '')}}</p></div></div><div class="card-body"><div class="text-col">${{text}}</div><div class="image-col">${{imgs}}</div></div></article>`;
  }}
  fetch(DATA_URL).then(r=>r.json()).then(data=>{{
    document.title = `${{data.name}} · 数据浏览器`;
    document.getElementById('pageTitle').textContent = data.name;
    document.getElementById('pageSubtitle').textContent = data.name;
    overviewText.textContent = `${{data.kind}} · ${{data.samples.length}} 条预览样本 · 扫描了 ${{data.total_files}} 个文件`;
    metaRow.innerHTML = [
      ['数据文件', data.data_label || 'n/a'],
      ['源目录', data.source_dir || data.name],
      ['预览类型', data.kind],
    ].map(([k,v])=>`<div class="meta-pill"><span>${{esc(k)}}</span><strong>${{esc(v)}}</strong></div>`).join('');
    if (data.warnings && data.warnings.length) {{
      warningArea.innerHTML = `<div class="warning"><strong>警告</strong><ul>${{data.warnings.map(w=>`<li>${{esc(w)}}</li>`).join('')}}</ul></div>`;
    }}
    sampleGrid.innerHTML = (data.samples || []).map(renderSample).join('') || '<div class="warning">未提取到样本。</div>';
  }}).catch(err => {{
    warningArea.innerHTML = `<div class="warning"><strong>警告</strong><div>加载预览数据失败：${{esc(err)}}</div></div>`;
  }});
  </script>
</body>
</html>
'''


ANNOTATION_SUMMARIES = {
    "composition-ava": "AVA.txt：序号、image_id、1–10 分各档人数（10列）、最多2个语义标签、challenge_id；可由评分分布计算 MOS。附加划分文件还提供通用/内容相关美学训练测试列表，以及 14 类摄影风格标签（训练单标签、测试多标签）。",
    "composition-para": "PIAA 图像表：session_id、图像名、user_id、总美学分、质量/构图/色彩/景深/内容/光照分、主体突出、情绪类别、判断难度、内容偏好、分享意愿、场景类别。用户表：年龄段、性别、学历、艺术/摄影经验、Big Five。GIAA 表还给出各分项的评分分布、均值/方差及人口属性分布。",
    "composition-aadb": "图像名/ID、评分者 ID、整体美学评分及聚合分；11 个属性：有趣内容、主体突出、良好光照、色彩和谐、鲜艳色彩、浅景深、运动模糊、三分法、平衡元素、重复、对称。官方 AADBinfo.mat 同时保存划分与属性/评分矩阵。",
    "composition-tad66k": "发布训练格式包含图像路径/ID、主题类别、聚合美学分（MOS）以及主题相关属性/准则标签；论文还描述每图至少约 1,200 个有效审美意见。注意：转换后的 Hugging Face 版本通常提供聚合标签，不一定保留全部逐人原始投票。",
    "composition-ickk17k": "ID：相对图像路径/样本 ID；color：主观色彩美学分；MOS：整体图像美学均分；probability：标注置信度。目录路径还编码 15 类单色和 15 类互补/多色组合。",
    "composition-aesmmit": "Hugging Face 字段：id、image、conversations（多轮列表，每项含 from 与 value）、type、qtype。对话内容中保存问题、答案、选项及与审美维度相关的解释。",
    "composition-portraitcraft": "Track 1 JSON：image_path、criteria、total_score；每个准则含 score、reason。13 项为色彩和谐、风格一致、清晰度、光影塑造、创意、曝光、经典构图、景深层次、视觉中心稳定、视觉流引导、结构支撑稳定、留白适切、主体完整。Track 2 另含结构化描述/VQA 标注。",
    "realqa": "图像/路径与多轮 messages 或 conversation；10 个属性分/文字解释：吸引力、构图、主体完整、主体杂乱、背景完整、背景杂乱、水平保持、清晰度、曝光、饱和度；另有综合分，以及按发布版本组织的直接回答、CoT/分析等对话字段。",
    "artimuse-10k": "图像、主类别、子类别、整体美学分（0–100）及 8 维专家文本：构图与设计、视觉元素与结构、技术执行、原创性与创意、主题与表达、情绪与观众反应、整体格式塔、综合评价。当前 HF 公开内容以测试集为主。",
    "composition-flickr-aes": "Flickr-AES：图像 ID、评分用户 ID、个人 1–5 分、由多人分数聚合并归一化到 0.2–1.0 的通用美学分及划分。REAL-CUR：图像/相册 ID、所有者/用户 ID、所有者个人评分及归一化分。",
}


FIELD_DESCRIPTIONS = {
    "line_number": "源标注文件中的行号",
    "raw_text": "源标注文件中的原始行内容",
    "image": "图像文件名或图像内容字段",
    "images": "样本关联的图像路径或图像列表",
    "image_stem": "由标注文件名推断出的图像文件主名",
    "image_id": "图像 ID，可用于关联本地图片或评分记录",
    "id": "样本唯一 ID",
    "index": "样本序号或表格索引",
    "score": "美学评分或该样本的目标分数",
    "scores": "多个维度或多个来源的评分集合",
    "raw_score": "转换前的原始评分",
    "raw_scores": "转换前的多个原始评分",
    "score_scale": "评分取值范围或归一化尺度",
    "label_format": "文本标注行被解析出的格式类型",
    "bbox_2d": "二维边界框坐标",
    "bbox_x1": "边界框左上角 x 坐标",
    "bbox_y1": "边界框左上角 y 坐标",
    "bbox_x2": "边界框右下角 x 坐标",
    "bbox_y2": "边界框右下角 y 坐标",
    "class_id": "类别 ID 或标签 ID",
    "question": "围绕图像提出的问题或评价任务",
    "answers": "问题对应的答案集合",
    "answer": "正确答案或标注答案",
    "A": "选择题 A 选项",
    "B": "选择题 B 选项",
    "C": "选择题 C 选项",
    "D": "选择题 D 选项",
    "category": "类别、任务类别或图像内容类别",
    "classification": "图像分类或内容类别",
    "split": "数据划分，如 train/val/test",
    "messages": "多轮对话消息，通常包含用户问题和助手回答",
    "conversations": "多轮视觉问答/指令微调对话内容",
    "source_dataset": "样本来源数据集名称",
    "channel": "训练样本所属通道或任务类型",
    "prompt": "文本提示词或生成/评价指令",
    "caption": "图像描述或生成提示文本",
    "path1": "第一张候选图像路径",
    "path2": "第二张候选图像路径",
    "preferred_uid": "偏好比较中被选中的图像 ID",
    "rejected_uid": "偏好比较中未被选中的图像 ID",
    "best_image_uid": "偏好标注中最佳图像 ID",
    "preferred_image": "偏好比较中胜出的图像",
    "rejected_image": "偏好比较中落败的图像",
    "preferred_position": "偏好图像在候选列表中的位置",
    "preference_target": "偏好学习目标或胜负标签",
    "pair_id": "成对比较样本 ID",
    "target": "成对比较或训练任务的目标标签",
    "score_gap": "高低质量/偏好图像之间的分数差",
    "high": "成对比较中高分或更优样本的信息",
    "low": "成对比较中低分或较差样本的信息",
    "are_different": "两张候选图是否被认为存在差异",
    "has_label": "该样本是否具备有效人工标签",
    "image_0_uid": "第一张候选图像 ID",
    "image_0_url": "第一张候选图像 URL",
    "image_1_uid": "第二张候选图像 ID",
    "image_1_url": "第二张候选图像 URL",
    "jpg_0": "第一张候选图像的内嵌 JPEG 二进制内容",
    "jpg_1": "第二张候选图像的内嵌 JPEG 二进制内容",
    "label_0": "第一张候选图像的偏好/标签值",
    "label_1": "第二张候选图像的偏好/标签值",
    "model_0": "生成第一张候选图像的模型名称",
    "model_1": "生成第二张候选图像的模型名称",
    "prompt_id": "提示词 ID",
    "image_path": "图像相对路径",
    "image_amount_in_total": "同一提示词或分组下的候选图像总数",
    "rank": "图像在候选集合中的排序名次",
    "overall_rating": "总体评分",
    "image_text_alignment_rating": "图文一致性评分",
    "fidelity_rating": "保真度或真实感评分",
    "difficulty": "标注者感知的判断难度",
    "visual": "视觉表现维度评分",
    "composition": "构图维度评分",
    "quality": "图像质量维度评分",
    "semantic": "语义相关维度评分",
    "vote_time": "投票或标注耗时",
    "device": "标注或采集设备信息",
    "user_id": "评分用户或标注者 ID",
    "created_at": "样本创建时间",
    "type": "样本或问题类型",
    "qtype": "问题子类型",
    "source_parquet": "来源 parquet 文件",
    "source_row": "来源文件中的行号",
    "source_json": "来源 JSON/JSONL 文件",
    "source_index": "来源文件中的样本索引",
    "metadata": "补充元数据",
    "iaa_targets": "图像美学评价的辅助目标标签",
    "iaa_target_texts": "辅助目标标签对应的文本说明",
    "roc_targets": "排序/相对比较目标标签",
    "roc_distributions": "相对比较或评分分布信息",
    "aesthetic_target": "美学评价目标",
    "icaa_target": "色彩美学评价目标",
    "annotation_confidence": "标注置信度",
    "confidence_normalization_mean": "置信度归一化使用的均值",
    "archive": "压缩包路径或来源归档文件",
    "image_root": "图片根目录",
    "newly_extracted": "本次新解压图片数量",
    "reused": "复用已有图片数量",
    "构图方法": "人工标注的构图方法类别",
    "构图描述": "对图像构图方式的文字描述",
    "annotations": "原始标注对象或标注列表",
    "Image": "图像文件名、图像路径或内嵌图像内容",
    "Description": "图像描述、审美分析说明或样本文本描述",
    "File": "当前可预览文件的文件名或路径",
    "Image id": "图像 ID，用于关联输入/输出图像和编辑记录",
    "Edit instruction": "图像编辑任务的自然语言指令",
    "Edit type": "编辑任务类型或编辑所依赖的视觉条件类型",
    "Input": "编辑前的输入图像描述或源图像内容",
    "Output": "编辑后的目标图像描述或结果图像内容",
    "Example id": "样本 ID",
    "Split": "数据划分，如 train/val/test/dev",
    "Image file": "图像文件名",
    "Task id": "任务 ID",
    "Prompt": "文本提示词、生成指令或编程题题面",
    "Canonical solution": "参考标准答案或标准解法代码",
    "Test": "用于验证答案的测试代码或测试条件",
    "Entry point": "待实现函数或程序入口名称",
    "Edit": "正向图像编辑指令",
    "Inverse edit": "反向编辑指令，用于把目标图像还原到输入状态",
    "File name": "样本对应的文件名或输出图像路径",
    "Sample id": "样本 ID",
    "Input style ref": "输入风格参考图像路径",
    "Instruction": "图像编辑、生成或风格迁移的自然语言指令",
    "Dist img": "失真图像文件名",
    "Ref img": "参考原图文件名",
    "Dmos": "差分平均意见分，表示失真图像相对参考图的主观质量下降程度",
    "Var": "主观评分方差，反映评分者一致性或不确定性",
    "Image name": "图像文件名",
    "M O S zscore": "MOS 的 z-score 标准化分数",
    "M O S": "平均意见分，表示图像主观质量/美学质量",
    "Excerpt": "从 README、说明文档或文本文件中截取的内容片段",
    "Img id": "图像 ID",
    "Turn index": "多轮编辑任务中的轮次序号",
    "Challenge": "提示词所属挑战类型或评测难点",
    "Note": "对提示词、类别或样本的补充说明",
    "Sub split": "数据子划分或细分测试 split",
    "Img name": "图像文件名",
    "Target text": "目标文本描述，表示希望图像编辑后匹配的内容",
    "Clip sim source": "编辑结果与源图像/源文本的 CLIP 相似度",
    "Clip sim target": "编辑结果与目标文本的 CLIP 相似度",
    "Clip sim dir": "CLIP 方向一致性分数，衡量编辑方向是否符合指令",
    "Clip sim image": "图像间 CLIP 相似度，衡量编辑前后视觉保持程度",
    "Dinov2 sim": "DINOv2 特征相似度，衡量图像内容/结构保持程度",
    "Ssim": "结构相似性指标，衡量编辑前后低层视觉结构相似度",
    "Id": "样本 ID",
    "Domain": "评测领域或任务大类",
    "Criterion": "评价准则或评价维度",
    "Qa template": "问答模板类型，用于生成评价问题",
}


def describe_field(label: str) -> str:
    if label in FIELD_DESCRIPTIONS:
        return FIELD_DESCRIPTIONS[label]
    lower = label.lower().replace(" ", "_")
    if lower in FIELD_DESCRIPTIONS:
        return FIELD_DESCRIPTIONS[lower]
    if lower in {"1", "2", "3", "4"}:
        return f"第 {label} 个离散选项/标签的取值或计数"
    if "prompt" in lower:
        return "文本提示词或任务指令"
    if "image" in lower or "img" in lower:
        return "图像路径、图像 ID 或图像内容"
    if "score" in lower or "rating" in lower or "mos" in lower or "dmos" in lower:
        return "评分、均分或质量评价值"
    if "category" in lower or "class" in lower or "type" in lower:
        return "类别或任务类型"
    if "id" in lower:
        return "样本、图像或记录的 ID"
    return "数据集中保留的原始标注字段，用于承载该样本的标签、说明或元数据"


def summarize_annotation_fields(data: dict[str, Any]) -> str:
    if data["slug"] in ANNOTATION_SUMMARIES:
        return ANNOTATION_SUMMARIES[data["slug"]]
    labels: list[str] = []
    for sample in data.get("samples", [])[:6]:
        for field in sample.get("text_fields", []):
            label = str(field.get("label", "")).strip()
            if label and label != "Annotation file" and label not in labels:
                labels.append(label)
            if len(labels) >= 10:
                break
        if len(labels) >= 10:
            break
    if labels:
        return "；".join(f"{label}：{describe_field(label)}" for label in labels) + "。"
    return "当前入口未发现结构化标注字段；卡片仅记录可预览文件、目录或资源占位信息。"


def update_index(new_datasets: list[dict[str, Any]]) -> None:
    index_path = DOCS / "index.html"
    text = index_path.read_text(encoding="utf-8")
    match = re.search(r"const DATASETS = (\[.*?\]);", text, re.S)
    if not match:
        raise RuntimeError("Could not find DATASETS array in docs/index.html")
    existing = json.loads(match.group(1))
    by_slug = {d["slug"]: d for d in existing if not str(d.get("slug", "")).startswith("composition-")}
    for data in new_datasets:
        by_slug[data["slug"]] = {
            "name": data["name"],
            "slug": data["slug"],
            "kind": data["kind"],
            "data_label": data["data_label"],
            "warnings": len(data.get("warnings", [])),
            "sample_count": data["sample_count"],
            "total_files": data["total_files"],
            "annotation_summary": summarize_annotation_fields(data),
        }
    merged = sorted(by_slug.values(), key=lambda d: str(d["name"]).lower())
    rendered = "const DATASETS = " + json.dumps(merged, ensure_ascii=False) + ";"
    text = text[:match.start()] + rendered + text[match.end():]
    index_path.write_text(text, encoding="utf-8")


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"source directory does not exist: {SOURCE_ROOT}")
    for old_asset in ASSET_DIR.glob("composition-*"):
        if old_asset.is_dir():
            shutil.rmtree(old_asset)
    built = []
    skipped = []
    for dataset_root in sorted([p for p in SOURCE_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        data = build_dataset(dataset_root)
        if data is None:
            skipped.append(dataset_root.name)
            continue
        slug = data["slug"]
        data_path = DATA_DIR / f"{slug}.json"
        data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        page_dir = DATASET_DIR / slug
        page_dir.mkdir(parents=True, exist_ok=True)
        (page_dir / "index.html").write_text(page_html(data["name"], f"{slug}.json"), encoding="utf-8")
        built.append(data)
    update_index(built)
    print(f"built {len(built)} composition dataset previews")
    if skipped:
        print("skipped without readable annotations: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
