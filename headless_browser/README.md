# Происхождение HTML-разметки электронных писем

Проект готовит датасет и обучает **один классификатор участия генеративного ИИ
в создании или изменении HTML-разметки**. Основной источник — EML.
Локальные HTML и веб-страницы поддерживаются как дополнительные источники.

Текст, CTA и анкоры ссылок предназначены другому участнику команды:
они извлекаются отдельно и не обучают этот классификатор.
Если ИИ переписал только видимый текст внутри неизменённых тегов,
это не основание ставить HTML-метку 1.

Три файла: main.py — сбор и предсказания; train.py — обучение и оценка;
test_pipeline.py — тесты и валидация. Python 3.11+.
Все команды выполняются из папки скриптов в выбранном Python-окружении.

## Установка

```shell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

Chromium/Playwright нужны только для URL. Чтение локальных EML/HTML/JSON/JSONL,
экспорт текста и валидация используют стандартную библиотеку.
Обучение использует существующие numpy/scipy/scikit-learn; BERT и LLM не нужны.

## Сбор

Два обязательных аргумента: источник и выходной JSON/JSONL.

```shell
python main.py mail.eml dataset.json
python main.py emails dataset.jsonl
python main.py page.html dataset.json
python main.py "https://example.org" dataset.json
python main.py urls.txt dataset.jsonl
```

В папке рекурсивно ищутся EML/HTML/HTM и архивы ZIP/TAR. В urls.txt — один http/https URL на строку,
пустые строки и строки с # пропускаются. URL загружается в headless Chromium,
сохраняется DOM; кнопки и ссылки не нажимаются.
Локальный HTML читается как UTF-8, MIME-часть EML декодируется с её кодировкой.

EML без HTML-части пропускается для этой модели и не получает метку 0.
Повреждённый файл или запись JSONL не останавливает остальные источники:
успешные записи сохраняются, причины пропусков/ошибок выводятся в терминал.
Код 0 означает полный успех, 2 — частичный результат, 1 — отсутствие результата/ошибку запуска.
Если не удалось создать ни одной HTML-записи, прежние выходные файлы не меняются.
Повторный успешный запуск с тем же выходным именем заменяет результат.

## Парсер папки или архива EML

```shell
python main.py emails dataset.json --text-export text_data.jsonl
python main.py emails.zip dataset.json --text-export text_data.jsonl
python main.py emails.tar.gz dataset.json --text-export text_data.jsonl
```

Поддерживаются ZIP, TAR, TAR.GZ/TGZ, TAR.BZ2/TBZ2 и TAR.XZ/TXZ.
Внутри архива читаются только файлы .eml, включая вложенные папки.
Содержимое не распаковывается на диск. Ссылки и небезопасные пути внутри архива
пропускаются. Повреждённое письмо или архив не останавливает другие источники.

В source сохраняется имя архива и путь письма внутри него.
Для каждого HTML-письма создаётся запись с label=null, ai_spans=null,
если подтверждённая разметка не задана явно. Письма без HTML пропускаются.
Текстовый экспорт содержит те же id; формат выхода не зависит от того,
было письмо отдельным файлом или членом архива.

## Формат датасета

.json — массив записей, .jsonl — одна запись на строку.

```json
[
  {
    "id": "email_001",
    "group_id": "template_001",
    "html": "<p>Привет</p>",
    "label": 1,
    "ai_spans": [{"start": 0, "end": 13}],
    "language": "ru",
    "email_type": "marketing",
    "generation_method": "full_generation",
    "annotation_source": "controlled_generation"
  }
]
```

- id уникален; group_id объединяет исходник, вариации и близкие шаблоны.
- html — исходная декодированная строка, без нормализации или пересериализации.
- label=0 — подтверждённое человеческое происхождение разметки, ai_spans=[].
- label=1 — подтверждённое участие ИИ. Если границы известны, ai_spans содержит
  интервалы [start, end) в Unicode-символах html. Если неизвестны — ai_spans=null.
- label=null — происхождение неизвестно, ai_spans=null.
- Известная метка требует непустой annotation_source. Генератор не выдумывается.

Необязательны language, email_type, generation_method, generator_model,
annotation_source, split (train/validation/test), parent_id и source.
parent_id сохраняет связь с исходником, source — путь/URL источника.
Пустые или выходящие за html интервалы, bool/дробные координаты и противоречивые
сочетания меток/интервалов отклоняются. Это границы ручной/контролируемой разметки.

Для исходника с известной историей:

```shell
python main.py mail.eml dataset.json --label 0 --annotation-source confirmed_author
python main.py generated.eml dataset.json --label 1 --annotation-source controlled_generation
```

Вторая команда оставляет ai_spans=null: границы автоматически не придумываются.
Размеченные интервалы можно задать в JSON/JSONL и проверить валидатором.
Без --label новая запись получает label=null и ai_spans=null.

Поддерживается CSV со списком источников и метаданными:

```csv
source,group_id,label,annotation_source,language,email_type
mail.eml,template_001,0,confirmed_author,ru,marketing
variant.eml,template_001,1,controlled_generation,ru,marketing
https://example.org,site_001,,,en,marketing
```

```shell
python main.py sources.csv dataset.jsonl
```

Относительные пути считаются от CSV. Допустимы url вместо source, id, split,
generation_method, generator_model, parent_id и ai_spans как JSON в корректно
экранированной CSV-ячейке. Старый html_label — алиас label.
Колонки cta_label/anchor_label/url_label игнорируются и не становятся HTML-разметкой.

## Экспорт текста для другого участника

```shell
python main.py emails dataset.jsonl --text-export text_data.jsonl
```

id совпадает с HTML-датасетом. HTML-метки текстовым элементам не присваиваются:

```json
{
  "id": "email_001",
  "text": "Полный извлечённый текст",
  "ctas": ["Подробнее"],
  "links": [{"anchor": "Подробнее", "href": "https://example.org"}]
}
```

Обнаружение CTA выполняется простыми правилами RU/EN и не является определением
их авторства. script/style/head/template исключаются из текстового экспорта.
Экспорт может быть JSON или JSONL и не включается в признаки модели.

## Обучение

```shell
python train.py labeled.json -o model.json
python train.py labeled.json --evaluate model.json -o metrics.json
```

Для train и validation нужны оба подтверждённых класса. label=null пропускается
при обучении и оценке. Если split не задан, связанные компоненты разбиваются
примерно на 70/15/15 с seed=42. Связи group_id, parent_id и точные дубликаты html
не могут пересекать выборки. Для близких шаблонов указывайте общий group_id.
Если split задан вручную, задайте его всем размеченным документам;
неразмеченные документы без split наследуют группу либо получают своё разбиение.

Внутреннее представление для модели создаётся отдельно от html:
обычные текстовые узлы и текстовые атрибуты заменяются маркером [TEXT];
конкретные URL и значения идентификаторов нормализуются.
Сохраняются теги, вложенность, имена атрибутов, структурные свойства CSS и счётчики HTML.
Текст скриптов/комментариев и буквальные CSS-строки не становятся текстовыми признаками.

Символьный TF-IDF и логистическая регрессия обучаются только на train.
На validation выбирается порог по F1 при FPR <= 0.20; это подбор порога,
а не отдельная калибровка вероятности. На test считаются ROC-AUC, F1, FPR,
в том числе по языкам/маркетингу при наличии метаданных.
model.json содержит параметры нормализации, словарь/IDF, коэффициенты, порог,
параметры обучения и метрики. Исходные письма в модель не записываются.

## Предсказание

```shell
python main.py mail.eml predicted.json --model model.json
python main.py labeled.json predicted.json --model model.json --text-export text_data.json
```

Подтверждённые label и ai_spans сохраняются. Добавляется только:

```json
{
  "prediction": {
    "html_ai_probability": 0.81,
    "html_ai_label": 1
  }
}
```

Классификатор не локализует AI-разметку: предсказанные интервалы не выдаются,
обучающие ai_spans не копируются в prediction. Качество локализации не измеряется.
Значения в примере иллюстративны.

## Старые данные

```shell
python main.py legacy.jsonl migrated.jsonl --text-export text_data.jsonl
```

Старая схема этого проекта содержит email_id/raw_html/zones.
Переносится только подтверждённая метка HTML-зоны, точно покрывающей весь raw_html.
Метки CTA/анкора/URL и прежние предсказания не используются.
Неразмеченные документы остаются неизвестными; положительная метка без интервалов
мигрирует с ai_spans=null. Частичная HTML-зона не превращается в метку всего документа.
Противоречивые подтверждённые HTML-метки отклоняются.
Связи родителя/группы сохраняются. Старые многозонные модели несовместимы:
HTML-классификатор нужно переобучить.

## Проверка

```shell
python test_pipeline.py --dataset dataset.json
python test_pipeline.py -v
```

Проверяются схема/интервалы, чтение источников, группы и дубликаты, отсутствие
видимого текста в представлении модели, независимый текстовый экспорт,
разделение train/validation/test и сохранение разметки при предсказании.
Синтетические тесты проверяют работу программы; их результаты не доказывают
качество определения AI-происхождения на реальных письмах.
