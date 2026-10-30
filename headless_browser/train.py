"""One HTML-markup classifier: normalized markup TF-IDF + structure + logistic regression."""
from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter
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
    # Only declarations survive: selector IDs/classes and literal content disappear.
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


def dependencies():
    try:
        import numpy as np
        from scipy.sparse import csr_matrix, hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, roc_auc_score
    except ImportError as error:
        raise RuntimeError("Install ML dependencies: python -m pip install -r requirements.txt") from error
    return np, csr_matrix, hstack, TfidfVectorizer, LogisticRegression, f1_score, roc_auc_score


def matrix(records, vectorizer, fit=False):
    np, csr_matrix, hstack, *_ = dependencies()
    prepared = [markup_features(record.html) for record in records]
    text = [item[0] for item in prepared]
    tfidf = vectorizer.fit_transform(text) if fit else vectorizer.transform(text)
    return hstack([tfidf, csr_matrix(np.asarray([item[1] for item in prepared]))]).tocsr()


def metrics(labels, scores, threshold):
    if not labels:
        return {"count": 0, "roc_auc": None, "f1": None, "fpr": None}
    *_, f1_score, roc_auc_score = dependencies()
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


def train_model(records, max_features=20000, seed=42):
    ensure_splits(records, seed)
    training = [record for record in records if record.split == "train" and record.label is not None]
    validation = [record for record in records if record.split == "validation" and record.label is not None]
    if any({record.label for record in subset} != {0, 1} for subset in (training, validation)):
        raise ValueError("HTML training requires both confirmed classes in train and validation")
    np, _, _, TfidfVectorizer, LogisticRegression, *_ = dependencies()
    started = perf_counter()
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, max_features=max_features)
    features = matrix(training, vectorizer, fit=True)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    classifier.fit(features, [record.label for record in training])
    scores = classifier.predict_proba(matrix(validation, vectorizer))[:, 1]
    labels = [record.label for record in validation]
    threshold = select_threshold(labels, scores)
    return {
        "format_version": 2, "backend": "html_tfidf_logistic", "processing": PROCESSING,
        "vectorizer": {"analyzer": "char", "ngram_range": [2, 5], "lowercase": True, "norm": "l2",
                       "vocabulary": {key: int(value) for key, value in vectorizer.vocabulary_.items()},
                       "idf": vectorizer.idf_.tolist()},
        "classifier": {"coefficients": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0])},
        "threshold": float(threshold), "validation_metrics": metrics(labels, scores, threshold),
        "training_params": {"seed": seed, "max_features": max_features, "class_weight": "balanced",
                            "threshold_selection": "validation_only", "probability_calibration": "none"},
        "fit_identity_keys": sorted(set().union(*(identity_keys(record) for record in training + validation))),
        "training_seconds": perf_counter() - started,
    }


class HTMLClassifier:
    def __init__(self, artifact):
        if artifact.get("format_version") != 2 or artifact.get("backend") != "html_tfidf_logistic":
            raise ValueError("Unsupported/legacy model; retrain the HTML-only classifier")
        if artifact.get("processing") != PROCESSING:
            raise ValueError("Unsupported HTML preprocessing configuration; retrain")
        self.artifact = artifact
        parameters = artifact["vectorizer"]
        vocabulary, idf = parameters["vocabulary"], parameters["idf"]
        coefficients = artifact["classifier"]["coefficients"]
        if (sorted(vocabulary.values()) != list(range(len(vocabulary))) or len(idf) != len(vocabulary)
                or len(coefficients) != len(vocabulary) + len(FEATURE_NAMES)):
            raise ValueError("Invalid feature dimensions")
        values = [*idf, *coefficients, artifact["classifier"]["intercept"], artifact["threshold"]]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in values) or any(value <= 0 for value in idf):
            raise ValueError("Nonfinite/invalid model parameters")
        if {key: parameters[key] for key in ("analyzer", "ngram_range", "lowercase", "norm")} != {
            "analyzer": "char", "ngram_range": [2, 5], "lowercase": True, "norm": "l2"
        }:
            raise ValueError("Unsupported vectorizer configuration")
        self.vectorizer = None

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text(encoding="utf-8-sig")))

    def predict(self, record):
        record.validate()
        np, _, _, TfidfVectorizer, *_ = dependencies()
        if self.vectorizer is None:
            parameters = self.artifact["vectorizer"]
            self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                             lowercase=True, vocabulary=parameters["vocabulary"])
            self.vectorizer.idf_ = np.asarray(parameters["idf"])
        features = matrix([record], self.vectorizer)
        logit = float((features @ np.asarray(self.artifact["classifier"]["coefficients"]))[0])
        logit += self.artifact["classifier"]["intercept"]
        probability = 1 / (1 + math.exp(-max(-700, min(700, logit))))
        # Classification only: no predicted spans or reused annotation intervals.
        return {"html_ai_probability": probability,
                "html_ai_label": int(probability >= self.artifact["threshold"])}


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
    parser = argparse.ArgumentParser(description="Обучить один классификатор происхождения HTML-разметки")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--evaluate", type=Path, metavar="MODEL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=20000)
    args = parser.parse_args(argv)
    output = args.output or Path("metrics.json" if args.evaluate else "model.json")
    if output.resolve() in {args.dataset.resolve(), *([args.evaluate.resolve()] if args.evaluate else [])}:
        parser.error("Output must differ from the input dataset/model")
    try:
        records = read_records(args.dataset)
        if args.evaluate:
            classifier = HTMLClassifier.load(args.evaluate)
            ensure_splits(records, classifier.artifact["training_params"]["seed"])
            artifact = {"test_metrics": evaluate_model(records, classifier)}
        else:
            ensure_splits(records, args.seed)
            artifact = train_model(records, args.max_features, args.seed)
            artifact["test_metrics"] = evaluate_model(records, HTMLClassifier(artifact))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if "validation_metrics" in artifact:
            print("HTML validation:", json.dumps(artifact["validation_metrics"]))
        print("HTML test:", json.dumps(artifact["test_metrics"]))
        print(f"Сохранено: {output.resolve()}")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
