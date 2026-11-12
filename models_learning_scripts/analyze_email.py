import os
import csv
import re
import argparse
import joblib
import numpy as np

from email import policy
from email.parser import BytesParser
from email.header import decode_header, make_header
from html.parser import HTMLParser

from razdel import sentenize

# НАСТРОЙКИ

MODEL_PATH = "logreg_sentence_detector.joblib"
INPUT_PATH = "emails"
OUTPUT_FILE = "email_analysis.csv"

THRESHOLD = 0.5

# HTML - TEXT
class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts)


def html_to_text(html):
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ДЕКОДИРОВАНИЕ ЗАГОЛОВКОВ

def decode_email_header(value):
    if not value:
        return ""

    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return str(value)


# ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ EMAIL

def extract_email_data(file_path):
    """
    Извлекает тему, отправителя и тело письма из .eml.
    Предпочитает text/plain, если он доступен.
    """

    with open(file_path, "rb") as file:
        message = BytesParser(policy=policy.default).parse(file)

    subject = decode_email_header(message.get("Subject", ""))
    sender = decode_email_header(message.get("From", ""))

    plain_parts = []
    html_parts = []

    if message.is_multipart():
        for part in message.walk():

            if part.get_content_disposition() == "attachment":
                continue

            content_type = part.get_content_type()

            if content_type == "text/plain":
                try:
                    plain_parts.append(part.get_content())
                except Exception:
                    pass

            elif content_type == "text/html":
                try:
                    html_parts.append(part.get_content())
                except Exception:
                    pass

    else:
        content_type = message.get_content_type()

        try:
            content = message.get_content()

            if content_type == "text/plain":
                plain_parts.append(content)

            elif content_type == "text/html":
                html_parts.append(content)

        except Exception:
            pass

    # Предпочитаем обычный текст письма.
    if plain_parts:
        body = "\n".join(plain_parts)
    elif html_parts:
        body = "\n".join(
            html_to_text(part) for part in html_parts
        )
    else:
        body = ""

    body = body.strip()

    return {
        "subject": subject,
        "sender": sender,
        "body": body
    }


# РАЗБИЕНИЕ НА ПРЕДЛОЖЕНИЯ

def split_into_sentences(text):
    if not isinstance(text, str):
        return []

    text = text.strip()

    if not text:
        return []

    sentences = [
        item.text.strip()
        for item in sentenize(text)
        if item.text.strip()
    ]

    return [
        sentence
        for sentence in sentences
        if len(sentence) >= 10
    ]


# АНАЛИЗ ПИСЬМА

def analyze_email(text, model):
    sentences = split_into_sentences(text)

    if not sentences:
        return None, []

    probabilities = model.predict_proba(sentences)[:, 1]

    sentence_results = []

    for sentence, probability in zip(sentences, probabilities):

        prediction = (
            "AI" if probability >= THRESHOLD else "Human"
        )

        sentence_results.append({
            "text": sentence,
            "ai_probability": round(float(probability), 4),
            "prediction": prediction
        })

    # Средняя оценка предложений.
    overall_probability = float(np.mean(probabilities))

    overall_prediction = (
        "AI" if overall_probability >= THRESHOLD else "Human"
    )

    result = {
        "ai_probability": round(overall_probability, 4),
        "prediction": overall_prediction
    }

    return result, sentence_results


# ОБРАБОТКА ОДНОГО EMAIL

def process_email(file_path, model):
    email_data = extract_email_data(file_path)

    result, sentence_results = analyze_email(
        email_data["body"],
        model
    )

    if result is None:
        print(f"[SKIP] Нет текста: {file_path}")
        return None

    return {
        "file": os.path.basename(file_path),
        "sender": email_data["sender"],
        "subject": email_data["subject"],
        "ai_probability": result["ai_probability"],
        "prediction": result["prediction"],
        "body": email_data["body"],
        "sentences": sentence_results
    }


# ОБРАБОТКА ПАПКИ

def main():

    print("Загружаем модель...")

    model = joblib.load(MODEL_PATH)

    print("Модель загружена.")
    print(f"Ищем письма в папке: {INPUT_PATH}")

    if os.path.isfile(INPUT_PATH):
        files = [INPUT_PATH]

    elif os.path.isdir(INPUT_PATH):
        files = [
            os.path.join(INPUT_PATH, name)
            for name in os.listdir(INPUT_PATH)
            if os.path.isfile(os.path.join(INPUT_PATH, name))
        ]

    else:
        raise FileNotFoundError(
            f"Не найдена папка или файл: {INPUT_PATH}"
        )

    if not files:
        print("Файлы .eml не найдены.")
        return

    print(f"Найдено писем: {len(files)}")

    email_results = []
    sentence_results = []

    for index, file_path in enumerate(files, start=1):

        print(
            f"[{index}/{len(files)}] "
            f"Анализируем {os.path.basename(file_path)}"
        )

        try:
            result = process_email(file_path, model)

            if result is None:
                continue

            email_results.append(result)

            for sentence in result["sentences"]:
                sentence_results.append({
                    "file": result["file"],
                    "subject": result["subject"],
                    "sentence": sentence["text"],
                    "ai_probability": sentence["ai_probability"],
                    "prediction": sentence["prediction"]
                })

            print(
                f"  Результат: {result['prediction']} "
                f"({result['ai_probability']:.2%})"
            )

        except Exception as error:
            print(f"[ERROR] {file_path}: {error}")

    # =========================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # =========================

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file",
                "sender",
                "subject",
                "ai_probability",
                "prediction",
                "body"
            ]
        )

        writer.writeheader()

        for result in email_results:
            writer.writerow({
                key: result[key]
                for key in [
                    "file",
                    "sender",
                    "subject",
                    "ai_probability",
                    "prediction",
                    "body"
                ]
            })

    with open(
        "sentence_analysis.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file",
                "subject",
                "sentence",
                "ai_probability",
                "prediction"
            ]
        )

        writer.writeheader()
        writer.writerows(sentence_results)

    print("\nАнализ завершён.")
    print(f"Письма: {OUTPUT_FILE}")
    print("Предложения: sentence_analysis.csv")


if __name__ == "__main__":
    main()
