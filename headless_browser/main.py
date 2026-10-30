"""EML-first HTML provenance dataset; optional unlabelled text/CTA/link export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import lzma
import math
import re
import sys
import stat
import tarfile
import zipfile
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


@dataclass
class Span:
    start: int
    end: int

    def validate(self, length):
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start < self.end <= length:
            raise ValueError(f"Invalid Unicode span [{self.start}, {self.end}) for length {length}")


@dataclass
class HTMLRecord:
    id: str
    group_id: str
    html: str
    label: int | None = None
    ai_spans: list[Span] | None = None
    language: str | None = None
    email_type: str | None = None
    generation_method: str | None = None
    generator_model: str | None = None
    annotation_source: str | None = None
    split: str | None = None
    parent_id: str | None = None
    source: str | None = None
    prediction: dict | None = None

    def validate(self):
        for name in ("id", "group_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.html, str) or not self.html.strip():
            raise ValueError("html must be a nonempty decoded string")
        if self.label is not None and (type(self.label) is not int or self.label not in (0, 1)):
            raise ValueError("label must be 0, 1 or null")
        if self.label is not None and (not isinstance(self.annotation_source, str) or not self.annotation_source.strip()):
            raise ValueError("Known HTML labels require nonempty annotation_source")
        if self.label is None and self.ai_spans is not None:
            raise ValueError("Unknown provenance requires ai_spans=null")
        if self.label == 0 and self.ai_spans != []:
            raise ValueError("Human HTML requires ai_spans=[]")
        if self.label == 1 and self.ai_spans == []:
            raise ValueError("Positive HTML requires nonempty ai_spans or null for unknown boundaries")
        if self.ai_spans is not None:
            if not isinstance(self.ai_spans, list):
                raise ValueError("ai_spans must be an array or null")
            for span in self.ai_spans:
                if not isinstance(span, Span):
                    raise ValueError("Each ai_spans item must have start/end")
                span.validate(len(self.html))
        if self.split not in (None, "train", "validation", "test"):
            raise ValueError("split must be train, validation, test or null")
        for name in ("language", "email_type", "generation_method", "generator_model",
                     "annotation_source", "parent_id", "source"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string or null")
        if self.prediction is not None:
            if not isinstance(self.prediction, dict) or set(self.prediction) != {"html_ai_probability", "html_ai_label"}:
                raise ValueError("prediction may contain only html_ai_probability and html_ai_label")
            probability = self.prediction["html_ai_probability"]
            label = self.prediction["html_ai_label"]
            if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid prediction probability")
            if type(label) is not int or label not in (0, 1):
                raise ValueError("Invalid predicted HTML label")

    def to_dict(self):
        self.validate()
        data = asdict(self)
        return {key: value for key, value in data.items()
                if value is not None or key in ("label", "ai_spans")}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Each record must be a JSON object")
        if "raw_html" in data and "email_id" in data:
            data = migrate_legacy(data)
        data = dict(data)
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown record fields: {sorted(unknown)}")
        spans = data.get("ai_spans")
        if spans is not None:
            if not isinstance(spans, list):
                raise ValueError("ai_spans must be an array or null")
            try:
                data["ai_spans"] = [Span(**span) for span in spans]
            except (TypeError, KeyError) as error:
                raise ValueError("Invalid ai_spans item") from error
        try:
            record = cls(**data)
        except TypeError as error:
            raise ValueError(f"Invalid record schema: {error}") from error
        record.validate()
        return record


def migrate_legacy(data):
    """Only a confirmed whole-document HTML zone can supply the document label."""
    raw = data["raw_html"]
    if not isinstance(raw, str):
        raise ValueError("Legacy raw_html must be a string")
    confirmed = []
    for zone in data.get("zones", []):
        if not isinstance(zone, dict) or zone.get("zone_type") != "html":
            continue
        label, origin = zone.get("label_ai"), zone.get("annotation_source")
        if type(label) is int and label in (0, 1) and isinstance(origin, str) and origin.strip():
            if (zone.get("source_field") == "raw_html" and zone.get("start") == 0
                    and zone.get("end") == len(raw) and zone.get("zone_text") == raw):
                confirmed.append(zone)
    labels = {zone["label_ai"] for zone in confirmed}
    if len(labels) > 1:
        raise ValueError("Conflicting legacy whole-document HTML labels")
    chosen = confirmed[0] if confirmed else None
    label = chosen["label_ai"] if chosen else None
    if label == 0 and chosen.get("ai_spans"):
        raise ValueError("Legacy human HTML cannot have AI spans")
    if label == 0 and any(
        isinstance(zone, dict) and zone.get("zone_type") == "html"
        and type(zone.get("label_ai")) is int and zone["label_ai"] == 1
        and isinstance(zone.get("annotation_source"), str) and zone["annotation_source"].strip()
        for zone in data.get("zones", [])
    ):
        raise ValueError("Legacy human document conflicts with a confirmed AI HTML zone")
    converted = {"id": data["email_id"], "group_id": data.get("group_id") or data["email_id"],
                 "html": raw, "label": label,
                 "ai_spans": [] if label == 0 else (chosen.get("ai_spans") or None) if label == 1 else None}
    if chosen:
        converted["annotation_source"] = chosen["annotation_source"]
    for name in ("language", "email_type", "generation_method", "generator_model", "split"):
        if data.get(name) is not None:
            converted[name] = data[name]
    for new, old in (("parent_id", "parent_email_id"), ("source", "source_path")):
        if data.get(old) is not None:
            converted[new] = data[old]
    
    return converted


def dataset_entries(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".json":
            data = json.load(stream)
            if not isinstance(data, list):
                raise ValueError("JSON dataset must be an array")
            for number, item in enumerate(data, 1):
                yield f"{path}:record {number}", item
        else:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                    except ValueError as error:
                        item = ValueError(f"Invalid JSON: {error}")
                    yield f"{path}:line {number}", item


def read_records(path):
    records = []
    for location, data in dataset_entries(path):
        try:
            if isinstance(data, Exception):
                raise data
            records.append(HTMLRecord.from_dict(data))
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{location}: {error}") from error
    return records


def write_data(path, data):
    if path.suffix.lower() not in (".json", ".jsonl"):
        raise ValueError("Output must have .json or .jsonl extension")
    content = (json.dumps(data, ensure_ascii=False, indent=2) + "\n" if path.suffix.lower() == ".json"
               else "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_records(path, records):
    write_data(path, [record.to_dict() for record in records])


def identity_keys(record):
    keys = {"id:" + record.id, "group:" + record.group_id,
            "hash:" + hashlib.sha256(record.html.encode("utf-8")).hexdigest()}
    if record.parent_id:
        keys.add("id:" + record.parent_id)
    return keys


def components(records):
    parents, seen = {}, set()

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for record in records:
        record.validate()
        if record.id in seen:
            raise ValueError(f"Duplicate id: {record.id}")
        seen.add(record.id)
        keys = sorted(identity_keys(record))
        for key in keys[1:]:
            left, right = root(keys[0]), root(key)
            parents[max(left, right)] = min(left, right)
    result = {}
    for record in records:
        result.setdefault(root("id:" + record.id), []).append(record)
    return list(result.values())


def assert_no_leakage(records):
    if any(record.split is None for record in records):
        raise ValueError("Every record must have a split")
    for group in components(records):
        if len({record.split for record in group}) != 1:
            raise ValueError("Related groups/parents/duplicate HTML occur in different splits")


def split_records(records, seed=42):
    for group in components(records):
        identity = min(record.group_id for record in group)
        value = int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "big") / 2**64
        split = "train" if value < .70 else "validation" if value < .85 else "test"
        for record in group:
            record.split = split
    return records


VOID = set("area base br col embed hr img input link meta param source track wbr".split())
HIDDEN = {"script", "style", "head", "template"}
BLOCKS = {"p", "div", "td", "li", "section", "article", "table"}
CTA = re.compile(
    r"(?<!\w)(?:ответьте|позвоните|подтвердите|узнать|узнайте|подробнее|"
    r"купить|купите|открыть|откройте|перейдите|скачать|скачайте|"
    r"зарегистрируйтесь|посмотреть|подписаться|"
    r"reply|call|confirm|learn more|read more|buy|shop|open|click|"
    r"download|sign up|register|subscribe|view)(?!\w)", re.I
)
ENTITY = re.compile(r"&(?:#[xX][0-9a-fA-F]+;?|#[0-9]+;?|[A-Za-z][A-Za-z0-9]+;?)")


@dataclass
class Node:
    tag: str
    attrs: dict
    start: int
    open_end: int
    end: int
    path: str
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)


class SourceParser(HTMLParser):
    def __init__(self, raw: str):
        super().__init__(convert_charrefs=False)
        self.raw = raw
        self.lines = [0] + [m.end() for m in re.finditer("\n", raw)]
        self.nodes: list[Node] = []
        self.stack: list[Node] = []
        self.pieces: list[str] = []
        self.mapping: list[list[int] | None] = []
        self.feed(raw)
        self.close()
        self.text = "".join(self.pieces)

    def source_offset(self) -> int:
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if tag in BLOCKS or tag in {"br", "tr", "hr", "h1", "h2", "h3"}:
            self.separator()
        start = self.source_offset()
        parent = self.stack[-1] if self.stack else None
        siblings = parent.children if parent else [n for n in self.nodes if n.parent is None]
        index = 1 + sum(n.tag == tag for n in siblings)
        path = (parent.path if parent else "") + f"/{tag}[{index}]"
        node = Node(tag, dict(attrs), start, start + len(self.get_starttag_text()),
                    len(self.raw), path, parent)
        self.nodes.append(node)
        if parent:
            parent.children.append(node)
        if tag in VOID:
            node.end = node.open_end
        else:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        node = self.nodes[-1]
        node.end = node.open_end
        if self.stack and self.stack[-1] is node:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in BLOCKS or tag in {"tr", "h1", "h2", "h3"}:
            self.separator()
        end = self.raw.find(">", self.source_offset())
        end = len(self.raw) if end < 0 else end + 1
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                for node in self.stack[index:]:
                    node.end = end
                del self.stack[index:]
                break

    def separator(self):
        if self.mapping and self.mapping[-1] is not None:
            self.pieces.append("\n")
            self.mapping.append(None)

    def append_text(self, decoded: str, mappings: list[list[int]]):
        if not decoded or any(node.tag in HIDDEN for node in self.stack):
            return
        self.pieces.append(decoded)
        self.mapping.extend(mappings)

    def handle_data(self, data):
        start = self.source_offset()
        self.append_text(data, [[start + i, start + i + 1] for i in range(len(data))])

    def handle_entityref(self, name):
        self.entity()

    def handle_charref(self, name):
        self.entity()

    def entity(self):
        start = self.source_offset()
        match = ENTITY.match(self.raw, start)
        if match:
            decoded = html.unescape(match.group())
            
            if not any(node.tag in HIDDEN for node in self.stack):
                self.pieces.append(decoded)
                self.mapping.extend([[start, match.end()]] * len(decoded))



def extract_text_data(record):
    """Team handoff only: no labels and no features for the HTML classifier."""
    parser = SourceParser(record.html)
    links, candidates = [], []
    for node in parser.nodes:
        if node.tag not in ("a", "button"):
            continue
        positions = [index for index, span in enumerate(parser.mapping)
                     if span and node.open_end <= span[0] < node.end]
        first, last = (positions[0], positions[-1] + 1) if positions else (0, 0)
        while first < last and parser.text[first].isspace():
            first += 1
        while last > first and parser.text[last - 1].isspace():
            last -= 1
        anchor = parser.text[first:last]
        if node.tag == "a":
            links.append({"anchor": anchor, "href": node.attrs.get("href")})
        if anchor and (node.tag == "button" or CTA.search(anchor)):
            candidates.append((first, last))
    candidates.extend(match.span() for match in re.finditer(r"[^\n.!?]+[.!?]?", parser.text)
                      if not any(match.start() < end and match.end() > start for start, end in candidates))
    ctas = [parser.text[start:end].strip() for start, end in sorted(set(candidates))
            if CTA.search(parser.text[start:end]) or any(
                node.tag == "button" and node.open_end <= (parser.mapping[start] or [len(record.html)])[0] < node.end
                for node in parser.nodes)]
    return {"id": record.id, "text": parser.text, "ctas": ctas, "links": links}


class SkippedSource(ValueError):
    pass


def make_record(raw, source=None, email_type=None):
    identity = hashlib.sha256(((source or "") + "\0" + raw).encode("utf-8")).hexdigest()[:20]
    return HTMLRecord("email_" + identity, "email_" + identity, raw,
                      source=source, email_type=email_type)


def load_eml_bytes(payload, source):
    message = BytesParser(policy=policy.default).parsebytes(payload)
    body = message.get_body(preferencelist=("html",))
    if body is None or body.get_content_type() != "text/html":
        raise SkippedSource("EML без HTML: пропущено для модели разметки")
    raw = body.get_content()
    if not isinstance(raw, str) or not raw.strip():
        raise SkippedSource("Пустая HTML-часть EML: пропущено")
    record = make_record(raw, source)
    record.validate()
    return record


def load_email(path):
    if path.suffix.lower() == ".eml":
        return load_eml_bytes(path.read_bytes(), str(path.resolve()))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw = stream.read()
    record = make_record(raw, str(path.resolve()), "webpage")
    record.validate()
    return record


ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
ARCHIVE_ERRORS = (OSError, EOFError, RuntimeError, NotImplementedError,
                  zipfile.BadZipFile, tarfile.TarError, zlib.error, lzma.LZMAError)


def is_archive(path):
    return path.name.lower().endswith(ARCHIVE_SUFFIXES)


def safe_member(name):
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts and not (path.parts and ":" in path.parts[0])


def archive_entries(path):
    """Yield EML bytes/errors without ever extracting archive members to disk."""
    found = False
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16) or not member.filename.lower().endswith(".eml"):
                        continue
                    found = True
                    location = f"{path.resolve()}!/{member.filename}"
                    if not safe_member(member.filename):
                        yield location, ValueError("Unsafe archive member path")
                        continue
                    try:
                        with archive.open(member) as stream:
                            payload = stream.read()
                    except ARCHIVE_ERRORS as error:
                        yield location, ValueError(f"Cannot read EML archive member: {error}")
                        continue
                    yield location, payload
        else:
            with tarfile.open(path, "r:*") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.lower().endswith(".eml"):
                        continue
                    found = True
                    location = f"{path.resolve()}!/{member.name}"
                    if not safe_member(member.name):
                        yield location, ValueError("Unsafe archive member path")
                        continue
                    try:
                        stream = archive.extractfile(member)
                        if stream is None:
                            raise ValueError("Archive member has no file data")
                        with stream:
                            payload = stream.read()
                    except (*ARCHIVE_ERRORS, ValueError) as error:
                        yield location, ValueError(f"Cannot read EML archive member: {error}")
                        continue
                    yield location, payload
    except ARCHIVE_ERRORS as error:
        raise ValueError(f"Cannot open/read EML archive {path}: {error}") from error
    if not found:
        raise ValueError(f"No .eml files in archive: {path}")



def is_url(value):
    return urlsplit(value).scheme.lower() in ("http", "https")


def load_url(url, timeout=30):
    if not is_url(url) or not urlsplit(url).hostname:
        raise ValueError("Expected http/https URL")
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError as error:
        raise RuntimeError("Install requirements.txt and run python -m playwright install chromium") from error
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(accept_downloads=False)
                response = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                if response is None or response.status >= 400:
                    raise ValueError(f"HTTP {response.status if response else 'no response'}: {url}")
                content_type = response.headers.get("content-type", "").lower()
                if content_type and "html" not in content_type:
                    raise ValueError(f"Expected HTML, received {content_type}")
                page.wait_for_timeout(1000)
                raw, final_url = page.content(), page.url
            finally:
                browser.close()
    except Error as error:
        raise RuntimeError(f"Chromium could not load {url}: {error}") from error
    record = make_record(raw, url, "webpage")
    record.source = final_url
    record.validate()
    return record


@dataclass
class Source:
    source: str
    metadata: dict = field(default_factory=dict)


def parse_label(value):
    if value is None or str(value).strip().lower() in ("", "null", "unknown"):
        return None
    try:
        return {"0": 0, "1": 1, "human": 0, "ai": 1}[str(value).strip().lower()]
    except KeyError:
        raise ValueError("label must be 0, 1 or unknown") from None


def parse_sources(inputs):
    sources = []
    allowed = {item.name for item in fields(HTMLRecord)} - {"html", "prediction", "source"}
    for value in inputs:
        if is_url(value):
            sources.append(Source(value))
            continue
        path = Path(value)
        if path.is_dir():
            sources.extend(Source(str(item.resolve())) for item in sorted(path.rglob("*"))
                           if item.is_file() and (is_archive(item) or item.suffix.lower() in (".eml", ".html", ".htm")))
        elif path.is_file() and (is_archive(path) or path.suffix.lower() in (".eml", ".html", ".htm", ".json", ".jsonl")):
            sources.append(Source(str(path.resolve())))
        elif path.is_file() and path.suffix.lower() == ".txt":
            for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if not is_url(line):
                    raise ValueError(f"{path}:{number}: expected http/https URL")
                sources.append(Source(line))
        elif path.is_file() and path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                legacy_text = {"cta_label", "anchor_label", "url_label"}
                headers = set(reader.fieldnames or [])
                if not headers.intersection({"source", "url"}) or headers - allowed - legacy_text - {"source", "url", "html_label"}:
                    raise ValueError(f"{path}: unsupported CSV columns or missing source/url")
                for number, row in enumerate(reader, 2):
                    if None in row:
                        raise ValueError(f"{path}:{number}: too many CSV columns")
                    source = (row.get("source") or row.get("url") or "").strip()
                    if not source:
                        if any((cell or "").strip() for cell in row.values()):
                            raise ValueError(f"{path}:{number}: source is empty")
                        continue
                    if not is_url(source):
                        source = str((path.parent / source).resolve())
                    metadata = {key: value.strip() for key, value in row.items()
                                if key in allowed and value is not None and value.strip()}
                    if "html_label" in row and (row["html_label"] or "").strip():
                        alias = parse_label(row["html_label"])
                        if "label" in metadata and parse_label(metadata["label"]) != alias:
                            raise ValueError(f"{path}:{number}: conflicting label/html_label")
                        metadata["label"] = alias
                    elif "label" in row:
                        metadata["label"] = parse_label(row.get("label"))
                    if "ai_spans" in metadata:
                        metadata["ai_spans"] = json.loads(metadata["ai_spans"])
                    sources.append(Source(source, metadata))
        else:
            raise ValueError(f"Unsupported or missing source: {value}")
    unique = {}
    for source in sources:
        if source.source in unique and source != unique[source.source]:
            raise ValueError(f"Conflicting metadata for {source.source}")
        unique[source.source] = source
    if not unique:
        raise ValueError("No input sources")
    return list(unique.values())


def apply_metadata(record, metadata):
    data = record.to_dict()
    metadata = dict(metadata)
    if "label" in metadata:
        label = parse_label(metadata["label"])
        if "ai_spans" not in metadata:
            metadata["ai_spans"] = ([] if label == 0 else
                                    data["ai_spans"] if label == 1 and data["label"] == 1 else None)
        metadata["label"] = label
    data.update(metadata)
    return HTMLRecord.from_dict(data)


def build_dataset(inputs, output, *, text_export=None, model=None, metadata=None, timeout=30):
    outputs = [output] + ([text_export] if text_export else [])
    if any(path.suffix.lower() not in (".json", ".jsonl") for path in outputs):
        raise ValueError("Outputs must have .json or .jsonl extension")
    sources = parse_sources(inputs)
    protected = {Path(value).resolve() for value in inputs if not is_url(value)}
    protected.update(Path(item.source).resolve() for item in sources if not is_url(item.source))
    if model:
        protected.add(model.resolve())
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.resolve() in protected for path in outputs):
        raise ValueError("Outputs must differ from each other and all input files/model")
    if model and (model.is_dir() or model.name == "metadata.json"):
        model_root = model.resolve() if model.is_dir() else model.resolve().parent
        if any(path.resolve().is_relative_to(model_root) for path in outputs):
            raise ValueError("Outputs must not overwrite checkpoint files")
    predictor = None
    if model:
        if __package__:
            from .train import HTMLClassifier
        else:
            from train import HTMLClassifier
        predictor = HTMLClassifier.load(model)
    records, handoff, issues, ids = [], [], [], set()

    def consume(record, spec, location):
        combined = {**spec.metadata, **(metadata or {})}
        record = apply_metadata(record, combined)
        if record.id in ids:
            raise ValueError(f"Duplicate id: {record.id}")
        text = extract_text_data(record) if text_export else None
        if predictor:
            record.prediction = predictor.predict(record)
        record.validate()
        ids.add(record.id)
        records.append(record)
        if text is not None:
            handoff.append(text)
        print(f"\nИсточник: {location}\n  id: {record.id}\n  HTML: {len(record.html)} символов; label: {record.label}")
        if text is not None:
            print(f"  Экспорт текста: CTA — {len(text['ctas'])}, ссылок — {len(text['links'])}")
        if record.prediction is not None:
            print(f"  Оценка HTML: P(AI)={record.prediction['html_ai_probability']:.3f}; "
                  f"вердикт={record.prediction['html_ai_label']}")

    def issue(location, error):
        status = "skipped" if isinstance(error, SkippedSource) else "error"
        issues.append({"source": location, "status": status, "error": str(error)})
        print(f"{'Пропущено' if status == 'skipped' else 'Ошибка'}: {location}: {error}", file=sys.stderr)

    for spec in sources:
        try:
            path = None if is_url(spec.source) else Path(spec.source)
            if path is not None and path.suffix.lower() in (".json", ".jsonl"):
                for location, data in dataset_entries(path):
                    try:
                        if isinstance(data, Exception):
                            raise data
                        consume(HTMLRecord.from_dict(data), spec, location)
                    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                        issue(location, error)
            elif path is not None and is_archive(path):
                for location, payload in archive_entries(path):
                    try:
                        if isinstance(payload, Exception):
                            raise payload
                        consume(load_eml_bytes(payload, location), spec, location)
                    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
                        issue(location, error)
            else:
                consume(load_url(spec.source, timeout) if path is None else load_email(path), spec, spec.source)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            issue(spec.source, error)
    if not records:
        raise ValueError("No HTML records produced; existing outputs were not changed")
    components(records)
    data = [record.to_dict() for record in records]
    write_data(output, data)
    if text_export:
        write_data(text_export, handoff)
    print(f"\nГотово. HTML-документов: {len(records)}; пропусков/ошибок: {len(issues)}")
    print(f"Датасет: {output.resolve()}")
    if text_export:
        print(f"Текстовый экспорт: {text_export.resolve()}")
    return records, issues


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description="EML → датасет происхождения HTML-разметки; также HTML и URL.")
    parser.add_argument("source", help="EML, папка писем, ZIP/TAR, HTML, URL, urls.txt, sources.csv или JSON/JSONL")
    parser.add_argument("output", type=Path, help="Выходной HTML-датасет .json/.jsonl")
    parser.add_argument("--text-export", type=Path, help="Отдельный неразмеченный текст/CTA/ссылки для участника команды")
    parser.add_argument("--model", type=Path, help="Каталог DistilBERT из train.py")
    parser.add_argument("--label", "--html-label", dest="label", type=parse_label, default=argparse.SUPPRESS)
    for name in ("language", "email_type", "group_id", "annotation_source", "generation_method", "generator_model"):
        parser.add_argument("--" + name.replace("_", "-"), default=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    metadata = {name: getattr(args, name) for name in
                ("label", "language", "email_type", "group_id", "annotation_source", "generation_method", "generator_model")
                if hasattr(args, name)}
    try:
        _, issues = build_dataset([args.source], args.output, text_export=args.text_export,
                                  model=args.model, metadata=metadata, timeout=args.timeout)
        return 2 if issues else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    sys.modules.setdefault("headless_browser.main" if __package__ else "main", sys.modules[__name__])
    raise SystemExit(main())
