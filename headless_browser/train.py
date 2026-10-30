"""DistilBERT classifier for normalized HTML markup."""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
from collections import Counter
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from time import perf_counter

if __package__:
    from .main import (HTMLRecord, VOID, assert_no_leakage, configure_console,
                       identity_keys, read_records, split_records, components)
else:
    from main import (HTMLRecord, VOID, assert_no_leakage, configure_console,
                      identity_keys, read_records, split_records, components)

FEATURE_NAMES = ("tags", "unique_tags", "max_depth", "attributes", "inline_styles",
                 "style_tags", "tables", "links", "images", "comments", "text_nodes")
PROCESSING = {"version": 1, "text": "[TEXT]", "urls": "[URL]", "identifiers": "[ID]",
              "scripts": "[SCRIPT]", "comments": "[COMMENT]", "css": "canonical_declarations_v1",
              "structural_features": list(FEATURE_NAMES)}
URL_ATTRS = {"href", "src", "srcset", "action", "formaction", "poster", "background",
             "cite", "longdesc", "xlink:href", "ping", "manifest"}
TEXT_ATTRS = {"title", "alt", "aria-label", "placeholder", "value", "label", "content"}
ID_ATTRS = {"id", "class", "name", "for", "headers", "aria-labelledby", "aria-describedby"}
SEMANTIC_ATTRS = {"role", "type", "dir", "lang", "charset", "http-equiv", "rel", "target",
                  "align", "valign", "colspan", "rowspan", "width", "height", "border",
                  "cellpadding", "cellspacing", "bgcolor", "color"}
KEYWORDS = set(("none auto inherit initial unset normal bold bolder lighter italic oblique "
                "block inline inline-block inline-flex inline-grid flex grid table table-row table-cell "
                "contents static relative absolute fixed sticky left right center justify top bottom middle "
                "baseline both hidden visible collapse solid dotted dashed double transparent currentcolor "
                "border-box content-box wrap nowrap row column stretch start end space-between space-around "
                "button link presentation img heading paragraph list listitem navigation main article section "
                "text password checkbox radio submit reset email number tel url search file "
                "ltr rtl utf-8 text/html stylesheet noopener noreferrer _blank _self ru en").split())


def normalized_value(value):
    value = re.sub(r"url\s*\([^)]*\)", "[URL]", value, flags=re.I)
    value = re.sub(r"(['\"]).*?\1", "[STR]", value, flags=re.S)
    value = re.sub(r"#[0-9a-fA-F]{3,8}\b", "[COLOR]", value)
    tokens = re.findall(r"\[[A-Z]+\]|[-+]?(?:\d*\.)?\d+(?:[a-zA-Z%]+)?|[a-zA-Z_][-a-zA-Z_]*|[^\w\s]", value)
    result = []
    for token in tokens:
        lower = token.lower()
        if token.startswith("["):
            result.append(token)
        elif re.fullmatch(r"[-+]?(?:\d*\.)?\d+(?:[a-zA-Z%]+)?", token):
            unit = re.sub(r"^[-+]?(?:\d*\.)?\d+", "", lower)
            result.append("[N]" + (unit if unit in {"px", "pt", "em", "rem", "%", "vh", "vw", "ch", "ex", "cm", "mm", "in", "pc", "s", "ms", "deg"} else ""))
        elif lower in KEYWORDS:
            result.append(lower)
        elif re.fullmatch(r"[a-zA-Z_][-a-zA-Z_]*", token):
            result.append("[VALUE]")
        else:
            result.append(token)
    return " ".join(result)


def normalize_css(value):
    value = re.sub(r"/\*.*?\*/", "", value, flags=re.S)
    declarations = []
    for name, content in re.findall(r"([-\w]+)\s*:\s*([^;{}]+)", value):
        name = "--CUSTOM" if name.startswith("--") else name.lower()
        declarations.append(name + ":" + ("[TEXT]" if name == "content" else normalized_value(content)))
    return ";".join(sorted(declarations)) or "[CSS]"


class MarkupParser(HTMLParser):
    def __init__(self, raw):
        super().__init__(convert_charrefs=False)
        self.tokens, self.stack, self.tags, self.counts = [], [], set(), Counter()
        self.feed(raw)
        self.close()

    def literal(self, marker):
        if not self.tokens or self.tokens[-1] != marker:
            self.tokens.append(marker)

    def handle_starttag(self, tag, attrs):
        self.counts["tags"] += 1
        self.tags.add(tag)
        self.counts["attributes"] += len(attrs)
        self.counts["tables"] += tag == "table"
        self.counts["links"] += tag == "a"
        self.counts["images"] += tag == "img"
        self.counts["style_tags"] += tag == "style"
        normalized = []
        for key, value in sorted(attrs, key=lambda pair: pair[0]):
            if value is None:
                normalized.append(key)
            elif key in URL_ATTRS:
                normalized.append(key + "=[URL]")
            elif key in TEXT_ATTRS:
                normalized.append(key + "=[TEXT]")
            elif key in ID_ATTRS or key.startswith("data-") or key.startswith("on"):
                normalized.append(key + "=[ID]")
            elif key == "style":
                self.counts["inline_styles"] += 1
                normalized.append(key + "=" + normalize_css(value))
            elif key in SEMANTIC_ATTRS:
                normalized.append(key + "=" + normalized_value(value.lower()))
            else:
                normalized.append(key + "=[VALUE]")
        self.tokens.append("<" + tag + (" " + " ".join(normalized) if normalized else "") + ">")
        self.counts["max_depth"] = max(self.counts["max_depth"], len(self.stack) + 1)
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        self.tokens.append("</" + tag + ">")
        if tag in self.stack:
            index = len(self.stack) - 1 - self.stack[::-1].index(tag)
            del self.stack[index:]

    def handle_data(self, value):
        if not value.strip():
            return
        if "script" in self.stack:
            self.literal("[SCRIPT]")
        elif "style" in self.stack:
            self.tokens.append(normalize_css(value))
        else:
            if not self.tokens or self.tokens[-1] != "[TEXT]":
                self.counts["text_nodes"] += 1
            self.literal("[TEXT]")

    def handle_entityref(self, name):
        self.handle_data(html.unescape("&" + name + ";"))

    def handle_charref(self, name):
        self.handle_data(html.unescape("&#" + name + ";"))

    def handle_comment(self, value):
        self.counts["comments"] += 1
        self.tokens.append("[COMMENT]")

    def handle_decl(self, value):
        self.tokens.append("<!DOCTYPE>")

    def representation(self):
        return " ".join(self.tokens)

    def features(self):
        self.counts["unique_tags"] = len(self.tags)
        return [math.log1p(self.counts[name]) for name in FEATURE_NAMES]


def markup_features(raw):
    parser = MarkupParser(raw)
    return parser.representation(), parser.features()



def metrics(labels, scores, threshold):
    if not labels:
        return {"count": 0, "roc_auc": None, "f1": None, "fpr": None}
    from sklearn.metrics import f1_score, roc_auc_score
    predictions = [int(score >= threshold) for score in scores]
    negative = [pred for label, pred in zip(labels, predictions) if label == 0]
    return {"count": len(labels),
            "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "fpr": sum(negative) / len(negative) if negative else None}


def select_threshold(labels, scores):
    candidates = sorted({0.0, math.nextafter(max(scores), math.inf), *map(float, scores)})
    choices = []
    for threshold in candidates:
        result = metrics(labels, scores, threshold)
        if result["fpr"] is not None and result["fpr"] <= .20:
            choices.append((result["f1"], threshold))
    return max(choices)[1]


def ensure_splits(records, seed=42):
    if not records:
        raise ValueError("Dataset is empty")
    if all(record.split is None for record in records):
        split_records(records, seed)
    elif any(record.split is None and record.label is not None for record in records):
        raise ValueError("Assign splits to all labelled records or to none")
    else:
        for group in components(records):
            assigned = {record.split for record in group if record.split is not None}
            if len(assigned) > 1:
                raise ValueError("Related records occur in different splits")
            if assigned:
                split = next(iter(assigned))
                for record in group:
                    if record.split is None:
                        record.split = split
            else:
                split_records(group, seed)
    assert_no_leakage(records)




DEFAULT_CHECKPOINT = "distilbert/distilbert-base-multilingual-cased"
BACKEND = "html_distilbert"
FORMAT_VERSION = 3
PROCESSING["neural_input"] = "canonical_markup_with_log_structural_counts_v1"


def dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install dependencies: python -m pip install -r requirements.txt") from error
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def resolve_device(name):
    torch, *_ = dependencies()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available; use --device cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available; use --device cpu")
    return torch.device(name)


def model_input(record):
    markup, counts = markup_features(record.html)
    structure = " ".join(f"{name}={value:.4f}" for name, value in zip(FEATURE_NAMES, counts))
    return "[STRUCTURE] " + structure + " [MARKUP] " + markup


def encode_document(record, tokenizer, settings):
    encoded = tokenizer(
        model_input(record), max_length=settings["max_length"], stride=settings["stride"],
        truncation=True, return_overflowing_tokens=True, return_token_type_ids=False,
        padding=False,
    )
    return [{"input_ids": ids, "attention_mask": mask}
            for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])]


def document_logit(model, tokenizer, windows, chunk_batch_size, device):
    torch, *_ = dependencies()
    margins = []
    for start in range(0, len(windows), chunk_batch_size):
        batch = tokenizer.pad(windows[start:start + chunk_batch_size], padding=True, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()
                 if key in ("input_ids", "attention_mask")}
        logits = model(**batch).logits
        margins.append(logits[:, 1] - logits[:, 0])
    values = torch.cat(margins)
    return torch.logsumexp(values, dim=0) - math.log(len(windows))


def validate_metadata(artifact):
    if artifact.get("format_version") != FORMAT_VERSION or artifact.get("backend") != BACKEND:
        raise ValueError("Unsupported/legacy model; retrain with DistilBERT")
    if artifact.get("processing") != PROCESSING:
        raise ValueError("Unsupported HTML preprocessing configuration")
    inference = artifact["inference"]
    if inference.get("aggregation") != "logmeanexp_margin":
        raise ValueError("Unsupported window aggregation")
    if (type(inference["max_length"]) is not int or inference["max_length"] < 8
            or type(inference["stride"]) is not int or not 0 <= inference["stride"] < inference["max_length"] - 2
            or type(inference["chunk_batch_size"]) is not int or inference["chunk_batch_size"] < 1):
        raise ValueError("Invalid window settings")
    threshold = artifact["threshold"]
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 <= threshold <= math.nextafter(1.0, math.inf)):
        raise ValueError("Invalid decision threshold")


class HTMLClassifier:
    def __init__(self, model, tokenizer, artifact, device="cpu"):
        validate_metadata(artifact)
        if model.config.model_type != "distilbert" or model.config.num_labels != 2:
            raise ValueError("Expected a DistilBERT classifier with two outputs")
        if artifact["inference"]["max_length"] > model.config.max_position_embeddings:
            raise ValueError("max_length exceeds the model position limit")
        self.model, self.tokenizer, self.artifact = model, tokenizer, artifact
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, path, device="auto"):
        path = Path(path)
        metadata = path / "metadata.json" if path.is_dir() else path
        artifact = json.loads(metadata.read_text(encoding="utf-8-sig"))
        validate_metadata(artifact)
        root = metadata.parent
        _, AutoModelForSequenceClassification, AutoTokenizer = dependencies()
        tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True, local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            root, local_files_only=True, use_safetensors=True, trust_remote_code=False,
            attn_implementation="eager",
        )
        return cls(model, tokenizer, artifact, device)

    def save(self, path):
        path = Path(path)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError("Model output must be a new or empty directory")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        (path / "metadata.json").write_text(
            json.dumps(self.artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def predict(self, record):
        record.validate()
        torch, *_ = dependencies()
        settings = self.artifact["inference"]
        windows = encode_document(record, self.tokenizer, settings)
        self.model.eval()
        with torch.inference_mode():
            logit = document_logit(self.model, self.tokenizer, windows,
                                   settings["chunk_batch_size"], self.device)
            probability = float(torch.sigmoid(logit).item())
        return {"html_ai_probability": probability,
                "html_ai_label": int(probability >= self.artifact["threshold"])}


def train_model(records, *, checkpoint=DEFAULT_CHECKPOINT, revision=None, epochs=3,
                learning_rate=2e-5, batch_size=2, max_length=512, stride=64,
                chunk_batch_size=4, seed=42, device="auto", local_files_only=False):
    ensure_splits(records, seed)
    training = [record for record in records if record.split == "train" and record.label is not None]
    validation = [record for record in records if record.split == "validation" and record.label is not None]
    if any({record.label for record in subset} != {0, 1} for subset in (training, validation)):
        raise ValueError("HTML training requires both confirmed classes in train and validation")
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
            or type(chunk_batch_size) is not int or chunk_batch_size < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or type(max_length) is not int or max_length < 8
            or type(stride) is not int or not 0 <= stride < max_length - 2):
        raise ValueError("Invalid training parameters")
    torch, AutoModelForSequenceClassification, AutoTokenizer = dependencies()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, revision=revision, use_fast=True, local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, revision=revision, num_labels=2, local_files_only=local_files_only,
        trust_remote_code=False, use_safetensors=True, attn_implementation="eager",
        id2label={0: "HUMAN_HTML", 1: "AI_HTML"}, label2id={"HUMAN_HTML": 0, "AI_HTML": 1},
    )
    settings = {"max_length": max_length, "stride": stride, "chunk_batch_size": chunk_batch_size,
                "aggregation": "logmeanexp_margin"}
    artifact = {
        "format_version": FORMAT_VERSION, "backend": BACKEND,
        "processing": deepcopy(PROCESSING), "threshold": .5,
        "base_model": str(checkpoint), "base_revision": getattr(model.config, "_commit_hash", None) or revision,
        "inference": settings,
        "training_params": {"seed": seed, "epochs": epochs, "learning_rate": learning_rate,
                            "batch_size": batch_size, "weight_decay": .01,
                            "encoder_trainable": True, "threshold_selection": "validation_only",
                            "probability_calibration": "none"},
        "fit_identity_keys": sorted(set().union(*(identity_keys(record) for record in training + validation))),
    }
    classifier = HTMLClassifier(model, tokenizer, artifact, device)
    prepared = [encode_document(record, tokenizer, settings) for record in training]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    positives = sum(record.label == 1 for record in training)
    positive_weight = torch.tensor((len(training) - positives) / positives, device=classifier.device)
    history = []
    for epoch in range(epochs):
        order = list(range(len(training)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            group = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                logit = document_logit(model, tokenizer, prepared[index], chunk_batch_size, classifier.device)
                target = torch.tensor(float(training[index].label), device=classifier.device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit, target, pos_weight=positive_weight)
                total_loss += float(loss.detach().item())
                (loss / len(group)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(training)})
        print(f"Epoch {epoch + 1}/{epochs}: HTML train loss={history[-1]['train_loss']:.4f}")
    model.eval()
    scores = [classifier.predict(record)["html_ai_probability"] for record in validation]
    labels = [record.label for record in validation]
    artifact["threshold"] = float(select_threshold(labels, scores))
    artifact["validation_metrics"] = metrics(labels, scores, artifact["threshold"])
    artifact["training_history"] = history
    artifact["training_seconds"] = perf_counter() - started
    return classifier


def evaluate_model(records, classifier):
    ensure_splits(records, classifier.artifact["training_params"]["seed"])
    selected = [record for record in records if record.split == "test" and record.label is not None]
    fitted = set(classifier.artifact["fit_identity_keys"])
    for record in selected:
        if fitted.intersection(identity_keys(record)):
            raise ValueError("Test record overlaps fitting/validation data")
    scores = [classifier.predict(record)["html_ai_probability"] for record in selected]
    threshold = classifier.artifact["threshold"]
    result = {"all": metrics([record.label for record in selected], scores, threshold)}
    subsets = {}
    for index, record in enumerate(selected):
        if record.language:
            subsets.setdefault("language:" + record.language, []).append(index)
        if record.email_type == "marketing":
            subsets.setdefault("marketing", []).append(index)
        if record.generation_method in ("human", "partial_rewrite"):
            subsets.setdefault("partial_vs_human", []).append(index)
    for name, indices in subsets.items():
        result[name] = metrics([selected[index].label for index in indices],
                               [scores[index] for index in indices], threshold)
    return result


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description="Обучить DistilBERT для происхождения HTML-разметки")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--evaluate", type=Path, metavar="MODEL_DIR")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--chunk-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or Path("metrics.json" if args.evaluate else "artifacts/html_distilbert")
    if (output.resolve() == args.dataset.resolve() or args.dataset.resolve().is_relative_to(output.resolve())
            or (args.evaluate and output.resolve().is_relative_to(args.evaluate.resolve()))):
        parser.error("Output must differ from and not contain input data/model")
    try:
        records = read_records(args.dataset)
        if args.evaluate:
            classifier = HTMLClassifier.load(args.evaluate, args.device)
            artifact = {"test_metrics": evaluate_model(records, classifier)}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            if output.suffix.lower() in (".json", ".jsonl"):
                raise ValueError("DistilBERT output is a directory, not a JSON file")
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError("Choose a new or empty model directory")
            classifier = train_model(
                records, checkpoint=args.checkpoint, revision=args.revision, epochs=args.epochs,
                learning_rate=args.learning_rate, batch_size=args.batch_size,
                max_length=args.max_length, stride=args.stride, chunk_batch_size=args.chunk_batch_size,
                seed=args.seed, device=args.device, local_files_only=args.local_files_only,
            )
            artifact = classifier.artifact
            artifact["test_metrics"] = evaluate_model(records, classifier)
            classifier.save(output)
            print("HTML validation:", json.dumps(artifact["validation_metrics"]))
        print("HTML test:", json.dumps(artifact["test_metrics"]))
        print(f"Сохранено: {output.resolve()}")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
