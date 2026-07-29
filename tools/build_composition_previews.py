#!/usr/bin/env python3
"""Build static preview pages for composition datasets.

Reads source datasets from /mnt/.../composition and writes only into this repo's docs/ tree.
"""
from __future__ import annotations

import csv
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

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
MAX_FIELD_CHARS = 5000
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_CANDIDATE_FILES = 120
MAX_COUNTED_FILES = 20000
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


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def compact(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
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
    elif isinstance(value, str) and Path(value).suffix.lower() in IMAGE_EXTS:
        found.append(value)
    return found


def resolve_image(path_text: str, dataset_root: Path, annotation_file: Path) -> Path | None:
    raw = path_text.strip().strip('"\'')
    if not raw:
        return None
    candidates = []
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)
    candidates.extend([
        dataset_root / raw,
        annotation_file.parent / raw,
        dataset_root / Path(raw).name,
    ])
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
    return None


def find_neighbor_image(annotation_file: Path, dataset_root: Path, used: set[Path]) -> Path | None:
    stem = annotation_file.stem
    search_dirs = [annotation_file.parent, annotation_file.parent.parent, dataset_root]
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


def make_sample(title: str, subtitle: str, annotation_file: Path, dataset_root: Path, value: Any, slug: str, index: int, used_images: set[Path]) -> dict[str, Any]:
    fields = [{"label": "Annotation file", "value": rel(annotation_file, dataset_root)}]
    flat = flatten_fields(value)
    for k, v in flat[:40]:
        fields.append({"label": str(k), "value": compact(v)})
    images = []
    for img_text in find_image_strings(value)[:4]:
        img_path = resolve_image(img_text, dataset_root, annotation_file)
        if img_path and img_path not in used_images:
            copied = copy_image(img_path, slug, index)
            if copied:
                images.append(copied)
                used_images.add(img_path)
    if not images:
        neighbor = find_neighbor_image(annotation_file, dataset_root, used_images)
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
        return data[:MAX_SAMPLES]
    if isinstance(data, dict):
        for key in ["data", "samples", "annotations", "items", "records", "train", "test", "val"]:
            if isinstance(data.get(key), list):
                return data[key][:MAX_SAMPLES]
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
            if len(rows) >= MAX_SAMPLES:
                break
    return rows


def read_table_samples(path: Path) -> list[Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames:
            for row in reader:
                rows.append(dict(row))
                if len(rows) >= MAX_SAMPLES:
                    break
        else:
            f.seek(0)
            raw = csv.reader(f, delimiter=delimiter)
            for row in raw:
                rows.append(row)
                if len(rows) >= MAX_SAMPLES:
                    break
    return rows


def read_txt_samples(path: Path) -> list[Any]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rows.append({"line_number": i + 1, "text": line})
            if len(rows) >= MAX_SAMPLES:
                break
    return rows


def read_parquet_samples(path: Path) -> list[Any]:
    if pq is not None:
        table = pq.read_table(path, use_threads=False)
        rows = table.slice(0, MAX_SAMPLES).to_pylist()
        return rows
    if pd is None:
        return []
    df = pd.read_parquet(path)
    return json.loads(df.head(MAX_SAMPLES).to_json(orient="records", force_ascii=False))


def candidate_score(path: Path) -> tuple[int, int, str]:
    lower = path.name.lower()
    score = 0
    if path.suffix.lower() == ".jsonl": score -= 50
    if path.suffix.lower() == ".parquet": score -= 45
    if path.suffix.lower() in {".csv", ".tsv"}: score -= 40
    if path.suffix.lower() == ".json": score -= 35
    if path.suffix.lower() == ".txt": score -= 20
    if any(name in lower for name in SKIP_FILE_NAMES): score += 100
    if "label" in lower or "annotation" in lower or "train" in lower or "test" in lower or "val" in lower: score -= 10
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
    samples = []
    warnings = []
    used_images: set[Path] = set()
    data_labels = []
    kinds = []
    for path in candidates[:80]:
        if len(samples) >= MAX_SAMPLES:
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
            if len(samples) >= MAX_SAMPLES:
                break
            idx = len(samples)
            samples.append(make_sample(f"Sample {idx + 1}", rel(path, dataset_root), path, dataset_root, row, slug, idx, used_images))
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
        }
    merged = sorted(by_slug.values(), key=lambda d: str(d["name"]).lower())
    rendered = "const DATASETS = " + json.dumps(merged, ensure_ascii=False) + ";"
    text = text[:match.start()] + rendered + text[match.end():]
    index_path.write_text(text, encoding="utf-8")


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"source directory does not exist: {SOURCE_ROOT}")
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
