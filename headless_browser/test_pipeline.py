"""Run tests or validate a new/legacy JSON/JSONL HTML dataset."""
import argparse
import base64
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import threading
import tarfile
import zipfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__:
    from . import train as training_module
    from .main import (HTMLRecord, Span, SkippedSource, assert_no_leakage, build_dataset,
                       components, configure_console, extract_text_data, load_email,
                       read_records, split_records, write_records)
    from .train import (HTMLClassifier, ensure_splits, evaluate_model, markup_features, train_model)
else:
    import train as training_module
    from main import (HTMLRecord, Span, SkippedSource, assert_no_leakage, build_dataset,
                      components, configure_console, extract_text_data, load_email,
                      read_records, split_records, write_records)
    from train import (HTMLClassifier, ensure_splits, evaluate_model, markup_features, train_model)

PACKAGE = Path(__file__).resolve().parent


def record(raw="<p>Reply 😀</p>", label=None, identity="one", group=None, spans=None, split=None):
    return HTMLRecord(identity, group or identity, raw, label,
                      [] if label == 0 and spans is None else spans,
                      annotation_source="software_test" if label is not None else None, split=split)


def fixture_records():
    
    records = []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            for index in range(2):
                identity = f"{split}-{label}-{index}"
                raw = ("<table style='width:100%'><tr><td><a href='/event'>Read more</a></td></tr></table>"
                       if label else "<p><a href='/event'>Reply to this email</a></p>")
                item = record(f"<!--{identity}-->" + raw, label, identity, split=split)
                item.language = "en"
                item.email_type = "marketing"
                records.append(item)
    return records


def legacy(label=None, origin=None, spans=None):
    raw = "<p>Привет 😀</p>"
    return {"email_id": "old", "group_id": "family", "raw_html": raw,
            "zones": [{"zone_id": "html_0000", "zone_type": "html", "source_field": "raw_html",
                       "start": 0, "end": len(raw), "zone_text": raw, "label_ai": label,
                       "annotation_source": origin, "ai_spans": spans or []},
                      {"zone_id": "cta_0001", "zone_type": "cta", "label_ai": 1,
                       "annotation_source": "text_generator"},
                      {"zone_id": "link_0002", "zone_type": "link", "anchor_label_ai": 1,
                       "url_label_ai": 1, "annotation_source": "text_generator"}],
            "predictions": [{"zone_type": "html", "label_ai": 1}]}


class SchemaTests(unittest.TestCase):
    def test_unicode_spans_and_roundtrip(self):
        item = record("<p>Привет 😀</p>", label=1, spans=[Span(0, 3), Span(10, 11)])
        item.validate()
        self.assertEqual(item.html[10:11], "😀")
        self.assertEqual(HTMLRecord.from_dict(item.to_dict()).to_dict(), item.to_dict())
        for suffix in ("json", "jsonl"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / ("data." + suffix)
                write_records(path, [item])
                self.assertEqual(read_records(path)[0].to_dict(), item.to_dict())

    def test_null_human_positive_rules_and_origin(self):
        for label, spans in ((None, None), (0, []), (1, None), (1, [Span(0, 3)])):
            record(label=label, spans=spans).validate()
        for label, spans in ((None, []), (None, [Span(0, 1)]), (0, [Span(0, 1)]), (1, [])):
            with self.subTest(label=label, spans=spans), self.assertRaises(ValueError):
                record(label=label, spans=spans).validate()
        data = record(label=0).to_dict()
        data["ai_spans"] = None
        with self.assertRaises(ValueError):
            HTMLRecord.from_dict(data)
        for origin in ("", "   ", None):
            item = record(label=1)
            item.annotation_source = origin
            with self.assertRaisesRegex(ValueError, "annotation_source"):
                item.validate()

    def test_invalid_spans_and_labels(self):
        for span in (Span(-1, 1), Span(0, 1000), Span(2, 2), Span(True, 3), Span(0, 1.5)):
            with self.subTest(span=span), self.assertRaises(ValueError):
                record(label=1, spans=[span]).validate()
        for label in (True, False, 2, "1"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                record(label=label).validate()

    def test_prediction_has_no_localization(self):
        item = record(label=1, spans=[Span(0, 3)])
        item.prediction = {"html_ai_probability": .81, "html_ai_label": 1}
        item.validate()
        self.assertEqual(item.to_dict()["ai_spans"], [{"start": 0, "end": 3}])
        item.prediction["ai_spans"] = [{"start": 0, "end": 3}]
        with self.assertRaises(ValueError):
            item.validate()


class InputExportTests(unittest.TestCase):
    def test_eml_decoding_and_html_crlf_preserved(self):
        raw = "<!doctype html>\r\n<p>Привет 😀 &amp; мир</p>\r\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eml = root / "mail.eml"
            eml.write_bytes(b"MIME-Version: 1.0\r\nContent-Type: text/html; charset=utf-8\r\n"
                            b"Content-Transfer-Encoding: base64\r\n\r\n" +
                            base64.b64encode(raw.encode("utf-8")) + b"\r\n")
            source = root / "page.html"
            source.write_bytes(raw.encode("utf-8"))
            self.assertEqual(load_email(eml).html, raw)
            self.assertEqual(load_email(source).html, raw)
            self.assertIsNone(load_email(eml).label)
            self.assertIsNone(load_email(eml).ai_spans)

    def test_plain_eml_skipped_without_artificial_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plain.eml").write_text("Content-Type: text/plain\n\nhello", encoding="utf-8")
            (root / "empty.html").write_text("", encoding="utf-8")
            (root / "good.html").write_text("<p>Read more</p>", encoding="utf-8")
            with self.assertRaises(SkippedSource):
                load_email(root / "plain.eml")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result, issues = build_dataset([str(root)], root / "out.json")
            self.assertEqual(len(result), 1)
            self.assertIsNone(result[0].label)
            self.assertEqual({item["status"] for item in issues}, {"skipped", "error"})

    def test_text_export_same_ids_no_labels(self):
        raw = "<style>.x{}</style><script>buy now</script><p>Привет &amp; мир 😀</p><a href='/x'>Подробнее</a><p>Ответьте на письмо.</p>"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.html"
            source.write_text(raw, encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                records, issues = build_dataset([str(source)], root / "data.jsonl",
                                               text_export=root / "text.json",
                                               metadata={"label": 0, "annotation_source": "author_log"})
            exported = json.loads((root / "text.json").read_text(encoding="utf-8"))
            self.assertFalse(issues)
            self.assertEqual(exported[0]["id"], records[0].id)
            self.assertEqual(set(exported[0]), {"id", "text", "ctas", "links"})
            self.assertIn("Привет & мир 😀", exported[0]["text"])
            self.assertNotIn("buy now", exported[0]["text"])
            self.assertNotIn(".x{}", exported[0]["text"])
            self.assertIn("Подробнее", exported[0]["ctas"])
            self.assertIn({"anchor": "Подробнее", "href": "/x"}, exported[0]["links"])
            self.assertNotIn("label", exported[0])
            self.assertEqual(records[0].html, raw)

    def test_partial_jsonl_corruption_does_not_abort_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            valid = record().to_dict()
            invalid = {**valid, "id": "invalid", "label": 0, "ai_spans": None}
            source.write_text(json.dumps(valid) + "\nnot-json\n" + json.dumps(invalid) + "\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                records, issues = build_dataset([str(source)], root / "output.json")
            self.assertEqual((len(records), len(issues)), (1, 2))
            with self.assertRaises(ValueError):
                read_records(source)

    def test_output_protection_and_all_fail_preserves_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.html"
            source.write_text("<p>Read more</p>", encoding="utf-8")
            output = root / "out.json"
            output.write_text("unchanged", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_dataset([str(source)], output, text_export=output)
            source.write_text("", encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
                build_dataset([str(source)], output)
            self.assertEqual(output.read_text(encoding="utf-8"), "unchanged")

    def test_two_argument_cli_from_folder_and_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.html"
            source.write_text("<p>Read more</p>", encoding="utf-8")
            for index, (args, cwd) in enumerate((
                (["main.py"], PACKAGE),
                (["-m", "headless_browser.main"], PACKAGE.parent),
            )):
                output = root / f"dataset result {index}.json"
                result = subprocess.run([sys.executable, "-B", *args, str(source), str(output)],
                                        cwd=cwd, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(set(json.loads(output.read_text(encoding="utf-8"))[0]) &
                                 {"id", "group_id", "html", "label", "ai_spans"},
                                 {"id", "group_id", "html", "label", "ai_spans"})


class ArchiveTests(unittest.TestCase):
    @staticmethod
    def eml(raw="<p>Привет 😀 &amp; мир</p>\r\n"):
        return (b"MIME-Version: 1.0\r\nContent-Type: text/html; charset=utf-8\r\n"
                b"Content-Transfer-Encoding: base64\r\n\r\n" +
                base64.b64encode(raw.encode("utf-8")) + b"\r\n")

    def test_zip_and_tar_formats_preserve_html_and_export_ids(self):
        raw = "<p>Привет 😀 &amp; мир</p>\r\n"
        for suffix, mode in ((".zip", None), (".tar", "w"), (".tar.gz", "w:gz"),
                             (".tgz", "w:gz"), (".tar.bz2", "w:bz2"), (".tar.xz", "w:xz")):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / ("emails" + suffix)
                items = [("nested/mail.EML", self.eml(raw)),
                         ("plain.eml", b"Content-Type: text/plain\r\n\r\nno html"),
                         ("ignored.txt", b"not an email")]
                if mode is None:
                    with zipfile.ZipFile(source, "w") as archive:
                        for name, payload in items:
                            archive.writestr(name, payload)
                else:
                    with tarfile.open(source, mode) as archive:
                        for name, payload in items:
                            member = tarfile.TarInfo(name)
                            member.size = len(payload)
                            archive.addfile(member, io.BytesIO(payload))
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    records, issues = build_dataset([str(source)], root / "html.json",
                                                   text_export=root / "text.jsonl")
                self.assertEqual((len(records), len(issues)), (1, 1))
                self.assertEqual(issues[0]["status"], "skipped")
                self.assertEqual(records[0].html, raw)
                self.assertIsNone(records[0].label)
                self.assertIsNone(records[0].ai_spans)
                self.assertIn("!/nested/mail.EML", records[0].source)
                exported = json.loads((root / "text.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(exported["id"], records[0].id)
                self.assertFalse((root / "nested").exists())

    def test_bad_archives_and_unsafe_names_do_not_abort_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "broken.zip").write_bytes(b"not a zip archive")
            (inputs / "standalone.eml").write_bytes(self.eml("<p>Read more</p>"))
            with zipfile.ZipFile(inputs / "emails.zip", "w") as archive:
                archive.writestr("../escape.eml", self.eml())
                archive.writestr("valid.eml", self.eml())
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                records, issues = build_dataset([str(inputs)], root / "html.jsonl")
            self.assertEqual((len(records), len(issues)), (2, 2))
            self.assertFalse((root / "escape.eml").exists())
            self.assertFalse((inputs / "valid.eml").exists())
            self.assertEqual(len({item.id for item in records}), 2)

    def test_empty_archive_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "empty.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("README.txt", "no mail")
            output = root / "data.json"
            output.write_text("unchanged", encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(ValueError):
                build_dataset([str(source)], output)
            self.assertEqual(output.read_text(encoding="utf-8"), "unchanged")

    def test_duplicate_html_from_archive_and_folder_stays_one_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.eml()
            (root / "one.eml").write_bytes(payload)
            with zipfile.ZipFile(root / "emails.zip", "w") as archive:
                archive.writestr("nested/one.eml", payload)
            with redirect_stdout(io.StringIO()):
                records, issues = build_dataset([str(root)], root / "data.json")
            self.assertFalse(issues)
            self.assertEqual(len(records), 2)
            split_records(records)
            self.assertEqual(records[0].split, records[1].split)

    def test_archive_cli_two_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "emails.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("one.eml", self.eml())
            output, text = root / "html.json", root / "text.jsonl"
            result = subprocess.run([sys.executable, "-B", str(PACKAGE / "main.py"),
                                     str(archive), str(output), "--text-export", str(text)],
                                    capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(validate_dataset(output)["documents"], 1)
            self.assertEqual(json.loads(text.read_text(encoding="utf-8"))["id"], read_records(output)[0].id)


class MigrationTests(unittest.TestCase):
    def test_text_labels_and_predictions_never_become_html_labels(self):
        data = legacy()
        item = HTMLRecord.from_dict(data)
        self.assertIsNone(item.label)
        self.assertIsNone(item.ai_spans)
        self.assertIsNone(item.prediction)
        self.assertNotIn("generator_model", item.to_dict())

    def test_only_confirmed_whole_document_html_label_migrates(self):
        self.assertEqual(HTMLRecord.from_dict(legacy(0, "author_log")).ai_spans, [])
        positive = HTMLRecord.from_dict(legacy(1, "generation_log", [{"start": 0, "end": 3}]))
        self.assertEqual((positive.label, positive.ai_spans), (1, [Span(0, 3)]))
        self.assertIsNone(HTMLRecord.from_dict(legacy(1, "generation_log")).ai_spans)
        self.assertIsNone(HTMLRecord.from_dict(legacy(1, None)).label)

    def test_partial_html_label_not_promoted_and_conflicts_rejected(self):
        data = legacy()
        data["zones"].append({"zone_type": "html", "label_ai": 1, "annotation_source": "log",
                              "start": 0, "end": 3, "source_field": "raw_html"})
        self.assertIsNone(HTMLRecord.from_dict(data).label)
        data["zones"][0].update(label_ai=0, annotation_source="author_log")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            HTMLRecord.from_dict(data)


class FeatureTests(unittest.TestCase):
    def test_visible_text_urls_and_identifiers_not_features(self):
        a = '<div id="campaign-123" class="brand-a"><a href="https://one.test/a?id=123" title="SECRET">Привет &amp; мир</a><img src="/one.png" alt="FIRST"></div>'
        b = '<div id="other-999" class="brand-b"><a href="https://two.test/b?id=999" title="DIFFERENT">Hello world</a><img src="/two.png" alt="SECOND"></div>'
        self.assertEqual(markup_features(a), markup_features(b))
        prepared, _ = markup_features(a)
        for forbidden in ("SECRET", "Привет", "мир", "campaign-123", "one.test", "FIRST"):
            self.assertNotIn(forbidden, prepared)
        self.assertIn("[TEXT]", prepared)
        self.assertIn("href=[URL]", prepared)

    def test_css_script_and_comment_content_do_not_leak(self):
        a = '<style>.brand123{width:123px;content:"SECRET";background:url(https://one.test/x)}</style><script>var id="secret123";</script><!--secret--><p>x</p>'
        b = '<style>.other999{width:999px;content:"OTHER";background:url(https://two.test/y)}</style><script>OTHER CODE</script><!--other--><p>Y</p>'
        self.assertEqual(markup_features(a), markup_features(b))
        self.assertNotIn("SECRET", markup_features(a)[0])
        self.assertNotEqual(markup_features('<div style="display:flex">text</div>'),
                            markup_features('<div style="display:grid">text</div>'))


class GroupTests(unittest.TestCase):
    def test_groups_parents_duplicates_and_order(self):
        a = record("<p>A</p>", identity="a", group="g1")
        b = record("<p>B</p>", identity="b", group="g2")
        b.parent_id = "a"
        c = record("<p>B</p>", identity="c", group="g3")
        split_records([a, b, c])
        self.assertEqual(len({a.split, b.split, c.split}), 1)
        before = a.split
        split_records([c, b, a])
        self.assertEqual(before, a.split)
        c.split = "test" if a.split != "test" else "train"
        with self.assertRaises(ValueError):
            assert_no_leakage([a, b, c])

    def test_validation_duplicate_ids_and_invalid_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            write_records(path, [record(), record()])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_dataset(path)
            value = record(label=1).to_dict()
            value["ai_spans"] = [{"start": 0, "end": 9999}]
            path.write_text(json.dumps([value]), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_dataset(path)


@unittest.skipUnless(importlib.util.find_spec("playwright"), "Playwright not installed")
class BrowserTests(unittest.TestCase):
    def test_batch_urls_and_failed_site(self):
        hits = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                if self.path == "/missing":
                    self.send_error(404)
                    return
                body = b"<html><body><script>document.body.innerHTML='<a href=\"/download.exe\">Read more</a>';</script></body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                urls = root / "urls.txt"
                urls.write_text(f"{base}/first\n{base}/missing\n{base}/second\n", encoding="utf-8")
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    records, issues = build_dataset([str(urls)], root / "data.jsonl",
                                                    text_export=root / "text.jsonl")
                self.assertEqual((len(records), len(issues)), (2, 1))
                self.assertEqual(len({item.id for item in records}), 2)
                text = [json.loads(line) for line in (root / "text.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual({item["id"] for item in text}, {item.id for item in records})
                self.assertIn("Read more", text[0]["ctas"])
                self.assertNotIn("/download.exe", hits)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "DistilBERT dependencies not installed")
class DistilBertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.directory = tempfile.TemporaryDirectory(prefix="distilbert-inference-test-")
        cls.root = Path(cls.directory.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<", ">", "/", "p",
                 "a", "div", "table", "tr", "td", "html", "body", "style", "display",
                 "flex", "grid", "TEXT", "URL", "ID", "VALUE", "N", "COLOR", "CSS",
                 "SCRIPT", "COMMENT", "STRUCTURE", "MARKUP", "[", "]", "=", ":", ";",
                 "width", "%", ".", "0", "1", "2", "3", "4"]
        vocab_path = cls.root / "vocab.txt"
        vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
        config = DistilBertConfig(vocab_size=len(tokenizer), max_position_embeddings=128,
                                  n_layers=1, n_heads=2, dim=24, hidden_dim=48,
                                  dropout=0, attention_dropout=0, seq_classif_dropout=0,
                                  num_labels=2)
        model = DistilBertForSequenceClassification(config)
        artifact = {
            "format_version": 3, "backend": "html_distilbert",
            "processing": deepcopy(training_module.PROCESSING), "threshold": .5,
            "base_model": "random_test_fixture_not_trained",
            "inference": {"max_length": 96, "stride": 16, "chunk_batch_size": 4,
                          "aggregation": "logmeanexp_margin"},
            "training_params": {"seed": 42},
            "fit_identity_keys": sorted(set().union(*(
                training_module.identity_keys(item) for item in fixture_records()
                if item.split in ("train", "validation")))),
        }
        cls.model_dir = cls.root / "model"
        HTMLClassifier(model, tokenizer, artifact, "cpu").save(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        import torch
        cls.directory.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def classifier(self):
        return HTMLClassifier.load(self.model_dir, "cpu")

    def test_real_distilbert_forward_and_safetensors_roundtrip(self):
        classifier = self.classifier()
        item = record()
        expected = classifier.predict(item)
        self.assertEqual(classifier.model.config.model_type, "distilbert")
        self.assertEqual(set(expected), {"html_ai_probability", "html_ai_label"})
        self.assertTrue(0 <= expected["html_ai_probability"] <= 1)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "saved"
            classifier.save(destination)
            self.assertTrue((destination / "model.safetensors").is_file())
            metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["backend"], "html_distilbert")
            self.assertNotIn("vectorizer", metadata)
            restored = HTMLClassifier.load(destination, "cpu")
            self.assertEqual(restored.predict(item), expected)

    def test_windows_cover_the_complete_markup(self):
        classifier = self.classifier()
        item = record("<div>" + "<table><tr><td>Text</td></tr></table>" * 20 + "</div>")
        settings = classifier.artifact["inference"]
        windows = training_module.encode_document(item, classifier.tokenizer, settings)
        self.assertGreater(len(windows), 1)
        reconstructed = windows[0]["input_ids"][1:-1]
        for window in windows[1:]:
            reconstructed += window["input_ids"][1:-1][settings["stride"]:]
        expected = classifier.tokenizer(training_module.model_input(item),
                                        add_special_tokens=False, truncation=False,
                                        verbose=False)["input_ids"]
        self.assertEqual(reconstructed, expected)
        result = classifier.predict(item)
        self.assertTrue(0 <= result["html_ai_probability"] <= 1)

    def test_predictions_preserve_known_labels_spans_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = record("<p>Привет 😀</p>", 1, "positive", spans=[Span(0, 3)])
            human = record("<div>Human text</div>", 0, "human")
            unknown = record("<p>Unknown text</p>", identity="unknown")
            source = root / "source.json"
            write_records(source, [positive, human, unknown])
            before = [item.to_dict() for item in (positive, human, unknown)]
            with redirect_stdout(io.StringIO()):
                result, issues = build_dataset([str(source)], root / "predictions.jsonl",
                                               text_export=root / "text.jsonl", model=self.model_dir)
            self.assertFalse(issues)
            for original, predicted in zip(before, result):
                data = predicted.to_dict()
                prediction = data.pop("prediction")
                self.assertEqual(data, original)
                self.assertEqual(set(prediction), {"html_ai_probability", "html_ai_label"})

    def test_evaluation_skips_unknowns_and_blocks_fitting_data(self):
        classifier = self.classifier()
        data = fixture_records()
        data.append(record("<p>Unknown</p>", identity="unknown", split="test"))
        self.assertEqual(evaluate_model(data, classifier)["all"]["count"], 4)
        data[0].split = "test"
        with self.assertRaisesRegex(ValueError, "overlaps"):
            evaluate_model(data, classifier)

    def test_checkpoint_protection_and_legacy_rejection(self):
        config = self.model_dir / "config.json"
        original = config.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.html"
            source.write_text("<p>Read more</p>", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_dataset([str(source)], config, model=self.model_dir)
            legacy_model = Path(directory) / "legacy.json"
            legacy_model.write_text(json.dumps({"format_version": 2, "backend": "html_tfidf_logistic"}),
                                    encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy"):
                HTMLClassifier.load(legacy_model, "cpu")
        self.assertEqual(config.read_bytes(), original)

    def test_unlabelled_training_rejected_before_loading_weights(self):
        data = fixture_records()
        for item in data:
            item.label, item.ai_spans, item.annotation_source = None, None, None
        with mock.patch.object(training_module, "dependencies",
                               side_effect=AssertionError("No model should be loaded")):
            with self.assertRaisesRegex(ValueError, "both confirmed classes"):
                train_model(data)


def validate_dataset(path):
    records = read_records(path)
    if not records:
        raise ValueError("Dataset is empty")
    components(records)
    if any(item.split is not None for item in records):
        assert_no_leakage(records)
    return {"documents": len(records),
            "labeled_html": sum(item.label is not None for item in records),
            "unknown_html": sum(item.label is None for item in records),
            "annotated_ai_spans": sum(item.label == 1 and item.ai_spans is not None for item in records)}


if __name__ == "__main__":
    configure_console()
    parser = argparse.ArgumentParser(description="Тесты и валидация HTML-датасета")
    parser.add_argument("--dataset", type=Path)
    args, remaining = parser.parse_known_args()
    if args.dataset:
        if remaining:
            parser.error("Unknown validation arguments")
        try:
            print("Валидация пройдена:", json.dumps(validate_dataset(args.dataset), ensure_ascii=False))
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.exit(1, f"Ошибка: {error}\n")
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
