Скрипты запускаются на debian подобных ОС, скачивают и устанавливают необходимые модели  самостоятельно (работы проводились на AstraLinux 1.8, ОП 32Гб, CPU 16 ядер).

# RAG-бот на базе Ollama, Qdrant, embeddinggemma, 

Локальная инфраструктура для системы вопросно-ответных ответов по документам.
Скачивается и используются по умолчанию:
- Ollama (модель Llama 3.2 3B) для генерации ответов
- Qdrant (векторная база) для поиска по смыслу ( запускается в докере)
- embeddinggemma (768) для эмбеддингов
- Safety-классификаторы Hugging Face :
  - unitary/toxic-bert                     — токсичность (EN)
  - Izbebe/ru-toxicity-classifier-rosberta — токсичность (RU)
  - smcleod/guardrails-v1                  — prompt injection / jailbreak
  - dslim/bert-base-NER                    — PII (PERSON, ORG, LOC, MISC)

## Требования
- Docker и Docker Compose v2
- Python 3.9+

## описание файлов для исследования работы RAG-бота

```
Task3_4_5/
├── docker-compose.yml          # Конфигурация Docker-сервисов (Qdrant)
├── run_rag2.sh                 # Главный скрипт-оркестратор
├── build_index.py              # Индексатор: файлы → чанки → эмбеддинги → Qdrant
├── test_rag.py                 # Поиск и генерация ответа (RAG-пайплайн)
├── safety_guard.py             # HF-классификаторы безопасности
├── requirements.txt            # список Python-зависимостей
├── knowledge_base/             # Исходные документы для индексации (текстовые файлы)
├── questions.json              # JSON с вопросами
├── qdrant_storage/             # локальное хранилище Qdrant
└── Пример FSI.txt              # Пример данных для Few-shot prompting 
```

## Дотупные параметры запуска run_rag2.sh, добавленные в ходе работы над задачами

```
Основные:
  --reindex                   Удалить коллекцию '$COLLECTION_NAME' и переиндексировать
  --use-ollama                Включить генерацию ответа через Ollama LLM
  --llm MODEL                 Модель Ollama (алиас: --model)
  --skip-docker               Не поднимать Qdrant (считать, что уже запущен)
  -h, --help                  Показать эту справку

Эмбеддинги / чанкинг:
  --embedding-provider NAME   ollama (по умолчанию) или st
  --embedding-model NAME      embeddinggemma | all-MiniLM-L6-v2 | nomic-embed-text | ...
  --embedding-dim N           768 (по умолчанию), 384 для all-MiniLM-L6-v2
  --chunk-size N              Размер чанка в символах (по умолчанию 500)
  --chunk-overlap N           Перекрытие чанков (по умолчанию 50)

Генерация (strict RAG):
  --strict-context            Отвечать ТОЛЬКО по векторной базе
  --min-score FLOAT           Минимальный score чанка (cosine)
  --top-k N, -k N             Сколько чанков передавать LLM (по умолчанию 5)
  --no-answer TEXT            Фраза, если ответа в контексте нет

Вывод:
  --no-chunks, -nc            Не выводить найденные чанки в лог (только сводку)

Безопасный промпт:
  --safe-prompt               Структурированный шаблон [System]/[Examples]/[CONTEXT]/[User]

Chain-of-Thought:
  --cot                       CoT: модель сначала рассуждает пошагово
  --cot-marker TEXT           Маркер между рассуждениями и ответом

Few-shot prompting:
  --few-shot-interactive      Запросить few-shot примеры интерактивно (алиас: --fsi)

Regex-безопасность (без HF):
  --on_regex_injection-scan   Regex-скан контекста и вопроса (алиас: -oris)
  --on-injection MODE         skip | redact | abort

HF safety-классификаторы:
  --safety-in                 ДО генерации: чистка документов + проверка запроса
  --safety-out                ПОСЛЕ генерации: проверка ответа LLM

Индексация выполняется в build_index.py. Все параметры прокидываются туда.

Вопросы:
  --queries-file PATH         JSON-файл с вопросами (алиасы: --queries, -q)
  --interactive, -i           Интерактивный режим

Примеры:
  $0 --reindex --use-ollama -i
  $0 --use-ollama --skip-docker -i
  $0 --use-ollama --safe-prompt --strict-context --min-score 0.55 \\
     --top-k 5 --cot --safety-in --safety-out -oris -nc -i

Автоустановка при отсутствии:
  - python3-venv / python3-pip / python3.X-venv
  - Docker (docker.io) и Docker Compose V2 — если Qdrant недоступен
  - Ollama — если EMBEDDING_PROVIDER=ollama или используется --use-ollama
```