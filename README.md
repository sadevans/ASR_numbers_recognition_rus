# Распознавание произнесённых чисел (русский ASR)
Проект выполнили:
- Горохова Александра, 2 курс
- Матюшков Андрей, 1 курс

## Установка

```bash
cd ASR_numbers_recognition_rus
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Запускайте скрипты из **корня репозитория**, чтобы корректно разрешался пакет `src`.

## Данные

`download_data.py` скачивает train и dev с Google Drive (`gdown`) и по умолчанию:

- сохраняет архивы в **`data/`**;
- распаковывает в **`data/train`** и **`data/dev`** (подкаталоги создаются автоматически).

```bash
python download_data.py
# если загрузка с Drive падает:
python download_data.py --fuzzy
```

Пути можно переопределить: `--out-dir`, `--extract-dir`, флаг `--no-extract` только скачивает архив.

### Манифест CSV

Ожидаемые колонки: `filename`, `transcription`, `spk_id`, `gender`, `ext`, `samplerate`. Аудиофайлы — относительно корня сплита (`--train-root` / `--dev-root`). Подробности в `SpokenNumbersDataset` (`src/dataset.py`).

После стандартной распаковки пути совпадают с дефолтными: `data/train/train.csv`, `data/dev/dev.csv`. При своей организации каталогов передайте явно нужные пути, например:

```bash
python train.py \
  --train-csv path/to/train.csv \
  --train-root path/to/train_audio_root \
  --dev-csv path/to/dev.csv \
  --dev-root path/to/dev_audio_root
```

## Обучение

- **`python train.py`** — основной скрипт для запуска обуечния
- **`python run_train.py`** — то же `main()`, с предварительным добавлением `src` в `sys.path`

Чекпоинт с лучшим CER на dev: **`checkpoints/best.pt`**. Логи TensorBoard: **`runs/ctc_baseline`** (по умолчанию).

```bash
python train.py --epochs 30 --batch-size 16 --device cuda
tensorboard --logdir runs/ctc_baseline
```

### Аргументы

| Аргумент | Описание |
|----------|-----------|
| `--train-csv`, `--train-root` | train csv и корень папки |
| `--dev-csv`, `--dev-root` | dev csv и корень папки |
| `--epochs` | Число эпох (по умолчанию 30) |
| `--batch-size` | Размер батча |
| `--lr`, `--weight-decay` | Оптимизатор AdamW |
| `--no-augment` | Отключить аугментации волны |
| `--no-spec-augment` | Отключить SpecAugment в модели |
| `--device` | `cuda` или `cpu` |
| `--log-dir`, `--checkpoint-dir` | TensorBoard и чекпоинты |
| `--resume` | Путь к `.pt` для загрузки весов |
| `--text-mode` | Цель CTC: `digits` (строка цифр) или `words` (русские слова, см. `num2words` в `text_normalize.py`) |

Метрики на dev: CER, точное совпадение последовательности, разделение по спикерам in-domain / out-of-domain относительно train (`src/metrics.py`). История эпох дополнительно пишется в `metrics.jsonl` в каталоге логов.

## Структура проекта

| Путь | Назначение |
|------|------------|
| `src/model.py` | `DigitCTCModel`: mel -> CNN -> BiGRU -> CTC |
| `src/dataset.py` | Датасет, 16 kHz, колляция батчей |
| `src/char_vocab.py` | Символы, blank CTC, кодирование и жадный декод |
| `src/augment.py` | Аугментации волны и лог-mel |
| `src/metrics.py` | CER, accuracy, разрезы по спикерам |
| `src/text_normalize.py` | Нормализация числа ↔ строка цели / сабмита |
| `train.py` | Обучение, валидация, `best.pt` |
| `download_data.py` | Загрузка и распаковка train/dev |
| `eda_train_dev.ipynb` | EDA |
| `submit2kaggle_team18.ipynb` | Ноутбук под окружение Kaggle и сабмит |
