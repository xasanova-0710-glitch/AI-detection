import os
import json
import random
import joblib
import numpy as np

from datasets import load_dataset
from razdel import sentenize

from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# =========================
# НАСТРОЙКИ
# =========================

DATASET_NAME = "iitolstykh/LLMTrace_classification"

EMAILS_FOLDER = "emails"

MAX_TRAIN_DOCUMENTS = 100_000
MAX_TEST_DOCUMENTS = 20_000

MAX_FEATURES = 200_000
NGRAM_RANGE = (1, 2)

RANDOM_SEED = 42
MODEL_PATH = "logreg_sentence_detector.joblib"

THRESHOLD = 0.5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================
# HTML -> TEXT
# =========================

class HTMLTextExtractor(HTMLParser):

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_content = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("script", "style"):
            self.skip_content = True

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style"):
            self.skip_content = False

    def handle_data(self, data):
        if not self.skip_content:
            self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts)


def html_to_text(html):
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# =========================
# ЧТЕНИЕ EMAIL
# =========================

def extract_email_text(file_path):
    """
    Извлекает только тело письма.
    Сначала ищет text/plain, затем использует HTML.
    """

    with open(file_path, "rb") as file:
        message = BytesParser(policy=policy.default).parse(file)

    plain_parts = []
    html_parts = []

    for part in message.walk():

        # Пропускаем вложения
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

    # Предпочитаем обычный текст
    if plain_parts:
        body = "\n".join(plain_parts)

    elif html_parts:
        body = "\n".join(
            html_to_text(part) for part in html_parts
        )

    else:
        body = ""

    return body.strip()


def load_generated_emails(folder):
    """
    Загружает все файлы из папки emails.
    Все письма получают метку AI = 1.
    """

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Папка с письмами не найдена: {folder}"
        )

    emails = []

    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
    ]

    print(f"\nНайдено файлов в {folder}: {len(files)}")

    for file_path in files:

        try:
            text = extract_email_text(file_path)

            if text and len(text.strip()) >= 10:
                emails.append(text)
                print(f"[OK] {os.path.basename(file_path)}")

            else:
                print(f"[SKIP] Пустое письмо: {file_path}")

        except Exception as error:
            print(f"[ERROR] {file_path}: {error}")

    print(f"\nУспешно загружено писем: {len(emails)}")

    return emails


# =========================
# ПОДГОТОВКА ДАННЫХ
# =========================

def normalize_label(value):
    """Метки human=0 и ai=1."""

    value = str(value).strip().lower()

    if value in ("human", "0"):
        return 0

    if value in ("ai", "1"):
        return 1

    raise ValueError(f"Неизвестная метка: {value}")


def split_into_sentences(text):
    """Разбивает текст на предложения."""

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


def documents_to_sentences(
    dataset,
    text_col,
    label_col,
    limit=None
):
    """
    Разбивает документы на предложения.
    Каждое предложение наследует метку документа.
    """

    if limit is not None:
        dataset = dataset.select(
            range(min(limit, len(dataset)))
        )

    texts = []
    labels = []

    for row in dataset:

        text = row[text_col]
        label = normalize_label(row[label_col])

        for sentence in split_into_sentences(text):
            texts.append(sentence)
            labels.append(label)

    return texts, np.asarray(labels, dtype=np.int32)


def emails_to_sentences(emails):
    """
    Разбивает дополнительные AI-письма на предложения.
    Всем предложениям присваивается метка 1.
    """

    texts = []

    for email_text in emails:
        sentences = split_into_sentences(email_text)
        texts.extend(sentences)

    labels = np.ones(len(texts), dtype=np.int32)

    return texts, labels


# =========================
# ОБУЧЕНИЕ
# =========================

def main():

    print("Загружаем исходный датасет...")

    dataset = load_dataset(DATASET_NAME)

    print("Доступные splits:", list(dataset.keys()))
    print("Столбцы:", dataset["train"].column_names)

    text_col = "text"
    label_col = "label"

    # Проверяем наличие нужных столбцов
    if text_col not in dataset["train"].column_names:
        raise ValueError(
            f"Не найден столбец {text_col}"
        )

    if label_col not in dataset["train"].column_names:
        raise ValueError(
            f"Не найден столбец {label_col}"
        )

    # =========================
    # ИСХОДНЫЕ ДАННЫЕ
    # =========================

    print("\nПодготавливаем исходную обучающую выборку...")

    X_train, y_train = documents_to_sentences(
        dataset["train"],
        text_col,
        label_col,
        MAX_TRAIN_DOCUMENTS,
    )

    # =========================
    # ДОПОЛНИТЕЛЬНЫЕ ПИСЬМА
    # =========================

    print("\nЗагружаем дополнительные AI-письма...")

    generated_emails = load_generated_emails(
        EMAILS_FOLDER
    )

    X_extra, y_extra = emails_to_sentences(
        generated_emails
    )

    print(
        f"Дополнительных AI-предложений: {len(X_extra):,}"
    )

    # Добавляем дополнительные письма в train
    X_train.extend(X_extra)

    y_train = np.concatenate([
        y_train,
        y_extra
    ])

    # Перемешиваем обучающую выборку
    combined = list(zip(X_train, y_train))

    random.shuffle(combined)

    X_train = [item[0] for item in combined]
    y_train = np.asarray(
        [item[1] for item in combined],
        dtype=np.int32
    )

    # =========================
    # TEST DATA
    # =========================

    print("\nПодготавливаем тестовую выборку...")

    X_test, y_test = documents_to_sentences(
        dataset["test"],
        text_col,
        label_col,
        MAX_TEST_DOCUMENTS,
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "После подготовки данных не осталось предложений."
        )

    print("\nРазмеры выборок:")
    print(f"Train: {len(X_train):,}")
    print(f"Test:  {len(X_test):,}")

    print("\nРаспределение классов в Train:")
    print(
        dict(zip(
            *np.unique(y_train, return_counts=True)
        ))
    )

    print("\nРаспределение классов в Test:")
    print(
        dict(zip(
            *np.unique(y_test, return_counts=True)
        ))
    )

    # =========================
    # TF-IDF + LOGISTIC REGRESSION
    # =========================

    print("\nОбучаем TF-IDF + Logistic Regression...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=NGRAM_RANGE,
                sublinear_tf=True,
                min_df=2,
                max_df=0.98,
                strip_accents=None,
                dtype=np.float32,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                C=4.0,
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_SEED,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    # =========================
    # ОЦЕНКА
    # =========================

    print("\nОцениваем модель на исходном test split...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "precision_ai": float(
            precision_score(
                y_test, y_pred, zero_division=0
            )
        ),
        "recall_ai": float(
            recall_score(
                y_test, y_pred, zero_division=0
            )
        ),
        "f1_ai": float(
            f1_score(
                y_test, y_pred, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_prob)
        ),
    }

    print("\n=== TEST METRICS ===")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n=== Classification report ===")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["human", "ai"],
        zero_division=0,
    ))

    print("=== Confusion matrix ===")

    print(confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ))

    # =========================
    # СОХРАНЕНИЕ
    # =========================

    with open(
        "logreg_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2
        )

    joblib.dump(model, MODEL_PATH)

    print(f"\nМодель сохранена: {MODEL_PATH}")
    print("Метрики сохранены: logreg_metrics.json")


# =========================
# ПРОГНОЗ ДЛЯ НОВОГО ТЕКСТА
# =========================

def predict_text(text, model_path=MODEL_PATH):

    model = joblib.load(model_path)

    sentences = split_into_sentences(text)

    if not sentences:
        return {
            "error": "Текст пустой или не содержит подходящих предложений."
        }

    probabilities = model.predict_proba(sentences)[:, 1]

    sentence_results = []

    for sentence, probability in zip(
        sentences,
        probabilities
    ):

        prediction = (
            "AI" if probability >= THRESHOLD else "Human"
        )

        sentence_results.append({
            "text": sentence,
            "ai_probability": round(
                float(probability), 4
            ),
            "prediction": prediction
        })

    overall_probability = float(
        np.mean(probabilities)
    )

    overall_prediction = (
        "AI"
        if overall_probability >= THRESHOLD
        else "Human"
    )

    return {
        "text": text,
        "ai_probability": round(
            overall_probability, 4
        ),
        "prediction": overall_prediction,
        "sentences": sentence_results
    }


# =========================
# ЗАПУСК
# =========================

if __name__ == "__main__":

    main()

    # Пример проверки нового текста
    text = """
    Здравствуйте! Направляю вам отчет по проекту.
    Все поставленные задачи были успешно выполнены.
    Буду рад обсудить дальнейшие шаги сотрудничества.
    """

    result = predict_text(text)

    print("\nРезультат анализа:")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4
        )
    )
