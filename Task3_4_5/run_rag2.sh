#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-qdrant}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ollama}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-embeddinggemma}"
EMBEDDING_DIM="${EMBEDDING_DIM:-768}"
KB_DIR="${KB_DIR:-knowledge_base}"
COLLECTION_NAME="${COLLECTION_NAME:-knowledge_base}"

CHUNK_SIZE="${CHUNK_SIZE:-500}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-50}"
EMBED_BATCH="${EMBED_BATCH:-8}"
UPSERT_BATCH="${UPSERT_BATCH:-64}"

STRICT_CONTEXT=0
MIN_SCORE="0.0"
NO_ANSWER="В предоставленном контексте нет ответа на этот вопрос."
TOP_K="${TOP_K:-5}"
QUERIES_FILE="${QUERIES_FILE:-}"
INTERACTIVE=0
COT=0
COT_MARKER="${COT_MARKER:-===ОТВЕТ===}"
FEW_SHOT_INTERACTIVE=0
SAFE_PROMPT=0
ON_REGEX_INJECTION_SCAN="${ON_REGEX_INJECTION_SCAN:-0}"
ON_INJECTION="${ON_INJECTION:-skip}"
NO_CHUNKS="${NO_CHUNKS:-0}"

SAFETY_IN="${SAFETY_IN:-0}"
SAFETY_OUT="${SAFETY_OUT:-0}"

COMPOSE_V2_URL="${COMPOSE_V2_URL:-https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)}"
COMPOSE_V2_PATH="${COMPOSE_V2_PATH:-/usr/local/lib/docker/cli-plugins/docker-compose}"
OLLAMA_INSTALL_URL="${OLLAMA_INSTALL_URL:-https://ollama.com/install.sh}"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

REINDEX=0
USE_OLLAMA=0
SKIP_DOCKER=0
PASSTHROUGH_ARGS=()

show_help() {
    cat <<EOF
Использование: $0 [опции]

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

Автоустановка при отсутствии:
  - python3-venv / python3-pip / python3.X-venv
  - Docker (docker.io) и Docker Compose V2 — если Qdrant недоступен
  - Ollama — если EMBEDDING_PROVIDER=ollama или используется --use-ollama

Переменные окружения:
  NO_AUTOINSTALL=1            Отключить автоустановку всех пакетов
  OLLAMA_INSTALL_URL=...      URL установочного скрипта Ollama
  PIP_INDEX_URL=...           Зеркало PyPI (по умолчанию Tsinghua)
  EMBEDDING_PROVIDER=...      ollama | st
  EMBEDDING_MODEL=...         Имя модели эмбеддингов
  EMBEDDING_DIM=...           Размерность вектора
  CHUNK_SIZE=...              Размер чанка
  CHUNK_OVERLAP=...           Перекрытие чанков
  TOP_K=...                   Сколько чанков передавать LLM
  SAFETY_IN=1, SAFETY_OUT=1   Включить HF-классификаторы
  NO_CHUNKS=1                 Отключить вывод найденных чанков

Вопросы:
  --queries-file PATH         JSON-файл с вопросами (алиасы: --queries, -q)
  --interactive, -i           Интерактивный режим

Примеры:
  $0 --reindex --use-ollama -i
  $0 --use-ollama --skip-docker -i
  $0 --use-ollama --safe-prompt --strict-context --min-score 0.55 \\
     --top-k 5 --cot --safety-in --safety-out -oris -nc -i
EOF
}

args=("$@")
i=0
while [ "$i" -lt "${#args[@]}" ]; do
    arg="${args[$i]}"
    case "$arg" in
        --reindex)      REINDEX=1 ;;
        --use-ollama)   USE_OLLAMA=1; PASSTHROUGH_ARGS+=("$arg") ;;
        --llm|--model)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            OLLAMA_MODEL="${args[$next_idx]}"; i=$((i + 1)) ;;
        --embedding-provider)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            EMBEDDING_PROVIDER="${args[$next_idx]}"; i=$((i + 1)) ;;
        --embedding-model)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            EMBEDDING_MODEL="${args[$next_idx]}"; i=$((i + 1)) ;;
        --embedding-dim)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            EMBEDDING_DIM="${args[$next_idx]}"; i=$((i + 1)) ;;
        --chunk-size)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            CHUNK_SIZE="${args[$next_idx]}"; i=$((i + 1)) ;;
        --chunk-overlap)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            CHUNK_OVERLAP="${args[$next_idx]}"; i=$((i + 1)) ;;
        --strict-context)   STRICT_CONTEXT=1 ;;
        --top-k|-k)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            TOP_K="${args[$next_idx]}"; i=$((i + 1)) ;;
        --min-score)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            MIN_SCORE="${args[$next_idx]}"; i=$((i + 1)) ;;
        --no-answer)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            NO_ANSWER="${args[$next_idx]}"; i=$((i + 1)) ;;
        --cot|--chain-of-thought)
            COT=1; PASSTHROUGH_ARGS+=("--cot") ;;
        --cot-marker)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            COT_MARKER="${args[$next_idx]}"; i=$((i + 1)) ;;
        --few-shot-interactive|--fsi)
            FEW_SHOT_INTERACTIVE=1
            PASSTHROUGH_ARGS+=("--few-shot-interactive") ;;
        --safe-prompt|--safe)
            SAFE_PROMPT=1
            PASSTHROUGH_ARGS+=("--safe-prompt") ;;
        --on_regex_injection-scan|-oris)
            ON_REGEX_INJECTION_SCAN=1
            PASSTHROUGH_ARGS+=("--on_regex_injection-scan") ;;
        --on-injection)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            ON_INJECTION="${args[$next_idx]}"; i=$((i + 1)) ;;
        --no-chunks|-nc)
            NO_CHUNKS=1
            PASSTHROUGH_ARGS+=("--no-chunks") ;;
        --safety-in)
            SAFETY_IN=1
            PASSTHROUGH_ARGS+=("--safety-in") ;;
        --safety-out)
            SAFETY_OUT=1
            PASSTHROUGH_ARGS+=("--safety-out") ;;
        --queries-file|--queries|-q)
            next_idx=$((i + 1))
            if [ "$next_idx" -ge "${#args[@]}" ]; then
                echo "[ERROR] Опция $arg требует значение."; exit 1
            fi
            QUERIES_FILE="${args[$next_idx]}"; i=$((i + 1)) ;;
        --interactive|-i)
            INTERACTIVE=1; PASSTHROUGH_ARGS+=("$arg") ;;
        --skip-docker)  SKIP_DOCKER=1 ;;
        -h|--help)      show_help; exit 0 ;;
        *)
            case "$arg" in
                ./*.sh|../*.sh|run_rag2.sh|bash|sh|python|python3)
                    echo "[WARN] Игнорирую аргумент '$arg'." ;;
                *)
                    PASSTHROUGH_ARGS+=("$arg") ;;
            esac
            ;;
    esac
    i=$((i + 1))
done

# --- Валидация ---
case "$EMBEDDING_PROVIDER" in
    ollama|st) ;;
    *) echo "[ERROR] --embedding-provider: 'ollama' или 'st'"; exit 1 ;;
esac
if ! [[ "$EMBEDDING_DIM" =~ ^[0-9]+$ ]] || [ "$EMBEDDING_DIM" -le 0 ]; then
    echo "[ERROR] --embedding-dim должен быть положительным целым."; exit 1
fi
if ! [[ "$MIN_SCORE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[ERROR] --min-score должен быть неотрицательным числом."; exit 1
fi
if ! [[ "$TOP_K" =~ ^[0-9]+$ ]] || [ "$TOP_K" -le 0 ]; then
    echo "[ERROR] --top-k должен быть положительным целым."; exit 1
fi
if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || [ "$CHUNK_SIZE" -le 0 ]; then
    echo "[ERROR] --chunk-size должен быть положительным целым."; exit 1
fi
if ! [[ "$CHUNK_OVERLAP" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --chunk-overlap должен быть неотрицательным целым."; exit 1
fi
case "$ON_INJECTION" in
    skip|redact|abort) ;;
    *) echo "[ERROR] --on-injection: skip | redact | abort"; exit 1 ;;
esac

if [ -n "$QUERIES_FILE" ]; then
    if [ ! -f "$QUERIES_FILE" ]; then
        echo "[ERROR] Файл с вопросами не найден: $QUERIES_FILE"; exit 1
    fi
    if ! "$PYTHON" -c "import json; json.load(open('$QUERIES_FILE', encoding='utf-8'))" >/dev/null 2>&1; then
        echo "[ERROR] Файл '$QUERIES_FILE' — невалидный JSON."
        "$PYTHON" -c "import json; json.load(open('$QUERIES_FILE', encoding='utf-8'))" 2>&1 | sed 's/^/        /' || true
        exit 1
    fi
fi

if [ "$SAFETY_IN" -eq 1 ] || [ "$SAFETY_OUT" -eq 1 ]; then
    [ -f "safety_guard.py" ] || {
        echo "[ERROR] Нужен safety_guard.py для --safety-in/--safety-out."; exit 1
    }
fi

if [ ! -f "build_index.py" ]; then
    echo "[ERROR] build_index.py не найден в $(pwd)."
    exit 1
fi

export OLLAMA_MODEL EMBEDDING_PROVIDER EMBEDDING_MODEL EMBEDDING_DIM
export STRICT_CONTEXT MIN_SCORE NO_ANSWER QUERIES_FILE COT COT_MARKER TOP_K
export FEW_SHOT_INTERACTIVE SAFE_PROMPT ON_REGEX_INJECTION_SCAN ON_INJECTION
export SAFETY_IN SAFETY_OUT NO_CHUNKS
export CHUNK_SIZE CHUNK_OVERLAP EMBED_BATCH UPSERT_BATCH

echo "=================================================="
echo "  Установка и запуск инфраструктуры RAG-бота"
echo "=================================================="
echo "[INFO] Провайдер эмбеддингов: $EMBEDDING_PROVIDER"
echo "[INFO] Модель эмбеддингов:   $EMBEDDING_MODEL (dim=$EMBEDDING_DIM)"
echo "[INFO] Чанкинг:              size=$CHUNK_SIZE, overlap=$CHUNK_OVERLAP"
echo "[INFO] LLM для генерации:    $OLLAMA_MODEL"
echo "[INFO] strict-context:       $STRICT_CONTEXT"
echo "[INFO] chain-of-thought:     $COT"
echo "[INFO] safe-prompt:          $SAFE_PROMPT"
echo "[INFO] regex-scan (-oris):   $ON_REGEX_INJECTION_SCAN"
echo "[INFO] safety-in (HF):       $SAFETY_IN"
echo "[INFO] safety-out (HF):      $SAFETY_OUT"
echo "[INFO] show-chunks:          $([ "$NO_CHUNKS" -eq 1 ] && echo "off (-nc)" || echo "on")"
[ "$ON_REGEX_INJECTION_SCAN" -eq 1 ] && echo "[INFO] on-injection:         $ON_INJECTION"
[ "$COT" -eq 1 ] && echo "[INFO] cot-marker:           $COT_MARKER"
[ "$FEW_SHOT_INTERACTIVE" -eq 1 ] && echo "[INFO] few-shot:             интерактивный ввод"
[ "$STRICT_CONTEXT" -eq 1 ] && echo "[INFO] min-score:            $MIN_SCORE"
echo "[INFO] top-k (чанков):       $TOP_K"
[ "$INTERACTIVE" -eq 1 ] && echo "[INFO] Режим вопросов:       интерактивный"
[ -n "$QUERIES_FILE" ] && echo "[INFO] Вопросы из файла:     $QUERIES_FILE"
[ "$REINDEX" -eq 1 ] && echo "[INFO] Режим переиндексации: коллекция '$COLLECTION_NAME' будет пересоздана."
[ "$USE_OLLAMA" -eq 1 ] && echo "[INFO] Режим Ollama LLM: генерация ответа включена."
[ "$SKIP_DOCKER" -eq 1 ] && echo "[INFO] Docker пропускается."

# 1. Проверка curl
command -v curl >/dev/null 2>&1 || { echo "[ERROR] curl не найден."; exit 1; }

# 2. Проверка python3 и модулей venv + ensurepip (с автоустановкой)
command -v "$PYTHON" >/dev/null 2>&1 || { echo "[ERROR] $PYTHON не найден."; exit 1; }

if ! "$PYTHON" -c "import venv, ensurepip" >/dev/null 2>&1; then
    echo "[WARN] Модули venv/ensurepip недоступны для $PYTHON."

    if [ "${NO_AUTOINSTALL:-0}" = "1" ]; then
        echo "[ERROR] Автоустановка отключена (NO_AUTOINSTALL=1)."
        echo "        Установите вручную:"
        echo "          Ubuntu/Debian/Astra: sudo apt install -y python3-venv python3-pip"
        exit 1
    fi

    INSTALLED=0
    PY_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"

    if command -v apt-get >/dev/null 2>&1; then
        echo "[INFO] Пробую установить python3-venv и python3-pip через apt-get..."
        if sudo apt-get update -qq \
           && sudo apt-get install -y python3-venv python3-pip; then
            INSTALLED=1
        elif [ -n "$PY_VER" ] \
             && sudo apt-get install -y "python3.${PY_VER##*.}-venv"; then
            INSTALLED=1
        fi
    fi

    if [ "$INSTALLED" -eq 0 ] && command -v dnf >/dev/null 2>&1; then
        echo "[INFO] Пробую установить через dnf..."
        if sudo dnf install -y python3-pip; then INSTALLED=1; fi
    fi
    if [ "$INSTALLED" -eq 0 ] && command -v yum >/dev/null 2>&1; then
        echo "[INFO] Пробую установить через yum..."
        if sudo yum install -y python3-pip; then INSTALLED=1; fi
    fi
    if [ "$INSTALLED" -eq 0 ] && command -v zypper >/dev/null 2>&1; then
        echo "[INFO] Пробую установить через zypper..."
        if sudo zypper install -y python3-pip; then INSTALLED=1; fi
    fi
    if [ "$INSTALLED" -eq 0 ] && command -v pacman >/dev/null 2>&1; then
        echo "[INFO] Пробую установить через pacman..."
        if sudo pacman -S --noconfirm python-pip; then INSTALLED=1; fi
    fi

    if ! "$PYTHON" -c "import venv, ensurepip" >/dev/null 2>&1; then
        echo "[ERROR] Модули venv/ensurepip всё ещё недоступны."
        if [ -n "$PY_VER" ]; then
            echo "        Попробуйте: sudo apt install -y python${PY_VER}-venv"
        fi
        exit 1
    fi
    echo "[OK] Модули venv/ensurepip установлены."
fi

# 3. venv
if [ ! -x "$VENV_DIR/bin/pip" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[INFO] Создание venv $VENV_DIR..."
    rm -rf "$VENV_DIR"
    if ! "$PYTHON" -m venv "$VENV_DIR"; then
        echo "[ERROR] Не удалось создать venv."
        echo "        Обычно помогает: sudo apt install -y python3.11-venv"
        exit 1
    fi
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[INFO] Зеркало PyPI: $PIP_INDEX_URL"

# 4. Зависимости
DEPS_OK=1
python -c "import qdrant_client, requests, pypdf, numpy, huggingface_hub, langchain_text_splitters" >/dev/null 2>&1 || DEPS_OK=0
if [ "$EMBEDDING_PROVIDER" != "ollama" ]; then
    python -c "import sentence_transformers" >/dev/null 2>&1 || DEPS_OK=0
fi
if [ "$SAFETY_IN" -eq 1 ] || [ "$SAFETY_OUT" -eq 1 ]; then
    python -c "import transformers, torch" >/dev/null 2>&1 || DEPS_OK=0
fi

if [ "$DEPS_OK" -eq 1 ]; then
    echo "[OK] Python-зависимости уже установлены."
else
    echo "[INFO] Обновление pip..."
    pip install --upgrade --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
        --timeout 120 --retries 10 pip

    echo "[INFO] Установка базовых зависимостей..."
    pip install --upgrade --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
        --timeout 300 --retries 10 --prefer-binary \
        qdrant-client requests pypdf numpy huggingface_hub langchain-text-splitters

    if [ "$EMBEDDING_PROVIDER" != "ollama" ]; then
        echo "[INFO] Установка sentence-transformers..."
        pip install --upgrade --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
            --timeout 300 --retries 10 --prefer-binary sentence-transformers
    fi

    if [ "$SAFETY_IN" -eq 1 ] || [ "$SAFETY_OUT" -eq 1 ]; then
        echo "[INFO] Установка HF safety-зависимостей (transformers + torch)..."
        pip install --upgrade --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
            --timeout 600 --retries 10 --prefer-binary transformers torch
    fi
fi

# =================================================================
# 4.5. ПРЕДЗАГРУЗКА HF-МОДЕЛЕЙ
# =================================================================
HF_MODELS_TO_DOWNLOAD=()

if [ "$SAFETY_IN" -eq 1 ] || [ "$SAFETY_OUT" -eq 1 ]; then
    SAFETY_IN_LIST="${SAFETY_IN_MODELS:-unitary/toxic-bert,Izbebe/ru-toxicity-classifier-rosberta,smcleod/guardrails-v1}"
    SAFETY_OUT_LIST="${SAFETY_OUT_MODELS:-unitary/toxic-bert,Izbebe/ru-toxicity-classifier-rosberta,dslim/bert-base-NER}"
    for item in $(echo "$SAFETY_IN_LIST,$SAFETY_OUT_LIST" | tr ',' '\n'); do
        name="${item%%:*}"
        name="$(echo -n "$name" | xargs)"
        [ -z "$name" ] && continue
        case " ${HF_MODELS_TO_DOWNLOAD[*]} " in
            *" $name "*) ;;
            *) HF_MODELS_TO_DOWNLOAD+=("$name") ;;
        esac
    done
fi

if [ "$EMBEDDING_PROVIDER" = "st" ]; then
    case " ${HF_MODELS_TO_DOWNLOAD[*]} " in
        *" $EMBEDDING_MODEL "*) ;;
        *) HF_MODELS_TO_DOWNLOAD+=("$EMBEDDING_MODEL") ;;
    esac
fi

if [ "${#HF_MODELS_TO_DOWNLOAD[@]}" -eq 0 ]; then
    echo "[INFO] HF-модели не требуются — предзагрузка пропущена."
else
    echo "[INFO] Предзагрузка HF-моделей: ${HF_MODELS_TO_DOWNLOAD[*]}"
    HF_MODELS_STR="$(IFS=','; echo "${HF_MODELS_TO_DOWNLOAD[*]}")"
    HF_MODELS_STR="$HF_MODELS_STR" HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    python -u - <<'PY'
import os, sys
models_str = os.environ.get("HF_MODELS_STR", "").strip()
if not models_str:
    print("[SKIP] Список моделей пуст."); sys.exit(0)
models = [m.strip() for m in models_str.split(",") if m.strip()]
try:
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"[ERROR] huggingface_hub не установлен: {e}"); sys.exit(1)
print(f"[INFO] Моделей к обработке: {len(models)}")
cached = downloaded = 0
failed = []
for name in models:
    try:
        snapshot_download(repo_id=name, local_files_only=True)
        print(f"[OK] {name} — уже в кэше."); cached += 1; continue
    except Exception:
        pass
    print(f"[INFO] Скачиваю {name} ...")
    try:
        snapshot_download(repo_id=name)
        print(f"[OK] {name} — скачано."); downloaded += 1
    except Exception as e:
        print(f"[ERROR] Не удалось скачать {name}: {e}")
        failed.append(name)
print(f"[INFO] Итог: в кэше={cached}, скачано={downloaded}, ошибок={len(failed)}")
if failed:
    for n in failed:
        print(f"       - {n}")
    sys.exit(2)
PY
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[ERROR] Предзагрузка HF-моделей упала (код $rc)."; exit 1
    fi
    echo "[OK] Все HF-модели готовы в кэше."
fi
# =================================================================

# =================================================================
# 5. Docker / Qdrant
# =================================================================
QDRANT_UP=0
if curl -s --max-time 2 "$QDRANT_URL/readyz" >/dev/null 2>&1; then
    QDRANT_UP=1
fi

if [ "$SKIP_DOCKER" -eq 1 ]; then
    echo "[INFO] --skip-docker: Docker не трогаем."
    if [ "$QDRANT_UP" -eq 0 ]; then
        echo "[WARN] Qdrant по адресу $QDRANT_URL не отвечает."
    fi
elif [ "$QDRANT_UP" -eq 1 ]; then
    echo "[OK] Qdrant уже запущен и отвечает на $QDRANT_URL — Docker не нужен."
else
    if ! command -v docker >/dev/null 2>&1; then
        echo "[WARN] Docker не найден, а Qdrant не запущен."
        if [ "${NO_AUTOINSTALL:-0}" = "1" ]; then
            echo "[ERROR] Автоустановка отключена (NO_AUTOINSTALL=1)."
            exit 1
        fi
        read -r -p "Установить Docker через apt-get? [y/N] " ANSWER
        case "$ANSWER" in
            [yY]|[yY][eE][sS])
                echo "[INFO] Устанавливаю docker.io..."
                sudo apt-get update -qq
                sudo apt-get install -y docker.io
                sudo apt-get install -y docker-compose-plugin 2>/dev/null || true
                sudo systemctl enable --now docker 2>/dev/null || \
                    sudo service docker start 2>/dev/null || true
                if ! id -nG "$USER" | grep -qw docker; then
                    sudo usermod -aG docker "$USER" 2>/dev/null || true
                    echo "[WARN] Пользователь $USER добавлен в группу 'docker'."
                    echo "       Перелогиньтесь или запустите: newgrp docker"
                fi
                ;;
            *)
                echo "[ERROR] Docker нужен для запуска Qdrant."
                echo "        Или запустите Qdrant вручную и используйте --skip-docker."
                exit 1
                ;;
        esac
    fi

    DOCKER=""
    if docker info >/dev/null 2>&1; then
        DOCKER="docker"
    elif sudo docker info >/dev/null 2>&1; then
        DOCKER="sudo docker"
        echo "[WARN] docker требует sudo (пользователь не в группе docker)."
    else
        echo "[ERROR] Docker-демон не отвечает."
        echo "        Запустите: sudo systemctl start docker"
        exit 1
    fi

    COMPOSE_OK=0
    if $DOCKER compose version >/dev/null 2>&1; then
        DC="$DOCKER compose"; COMPOSE_OK=1
        echo "[OK] Docker Compose V2: $($DOCKER compose version 2>/dev/null | head -1)"
    fi

    if [ "$COMPOSE_OK" -eq 0 ]; then
        echo "[WARN] Docker Compose V2 не найден."
        if [ "${NO_AUTOINSTALL:-0}" != "1" ]; then
            echo "[INFO] Устанавливаю плагин Docker Compose V2..."
            if sudo mkdir -p "$(dirname "$COMPOSE_V2_PATH")" \
               && sudo curl -fSL "$COMPOSE_V2_URL" -o "$COMPOSE_V2_PATH" \
               && sudo chmod +x "$COMPOSE_V2_PATH"; then
                if $DOCKER compose version >/dev/null 2>&1; then
                    DC="$DOCKER compose"; COMPOSE_OK=1
                    echo "[OK] Docker Compose V2: $($DOCKER compose version 2>/dev/null | head -1)"
                fi
            fi
        fi
    fi

    if [ "$COMPOSE_OK" -eq 0 ]; then
        if command -v docker-compose >/dev/null 2>&1 && docker-compose version >/dev/null 2>&1; then
            DC="docker-compose"; COMPOSE_OK=1
            echo "[WARN] Использую Docker Compose V1 (deprecated)."
        fi
    fi

    if [ "$COMPOSE_OK" -eq 0 ]; then
        echo "[ERROR] Не найдено рабочего Docker Compose."
        exit 1
    fi

    [ -f "$COMPOSE_FILE" ] || { echo "[ERROR] Не найден $COMPOSE_FILE"; exit 1; }

    echo "[INFO] Запуск Qdrant: $DC -f $COMPOSE_FILE up -d"
    $DC -f "$COMPOSE_FILE" up -d || true
fi
# =================================================================

# 6. Ожидание Qdrant
qdrant_ready() { curl -s --max-time 2 "$QDRANT_URL/readyz" >/dev/null 2>&1; }
echo -n "[INFO] Ожидание готовности Qdrant ($QDRANT_URL)"
for _ in $(seq 1 60); do
    qdrant_ready && { echo " — готов"; break; }
    echo -n "."; sleep 1
done
qdrant_ready || {
    echo; echo "[ERROR] Qdrant не ответил за 60 секунд."
    if command -v docker >/dev/null 2>&1; then
        docker logs "$QDRANT_CONTAINER" --tail 50 2>/dev/null || \
            sudo docker logs "$QDRANT_CONTAINER" --tail 50 2>/dev/null || true
    fi
    echo "        Возможные пути:"
    echo "          1) docker logs qdrant"
    echo "          2) Запустить Qdrant вручную и указать QDRANT_URL=..."
    exit 1
}

# =================================================================
# 7. Ollama (проверка, автоустановка, запуск)
# -----------------------------------------------------------------
# Ollama нужна, если:
#   - EMBEDDING_PROVIDER=ollama (для эмбеддингов)
#   - или --use-ollama (для генерации ответа)
# =================================================================
NEED_OLLAMA=0
[ "$EMBEDDING_PROVIDER" = "ollama" ] && NEED_OLLAMA=1
[ "$USE_OLLAMA" -eq 1 ] && NEED_OLLAMA=1

if [ "$NEED_OLLAMA" -eq 1 ]; then
    # 7.1 Установка при отсутствии
    if ! command -v ollama >/dev/null 2>&1; then
        echo "[WARN] Ollama не найдена."

        if [ "${NO_AUTOINSTALL:-0}" = "1" ]; then
            echo "[ERROR] Автоустановка отключена (NO_AUTOINSTALL=1)."
            echo "        Установите вручную:"
            echo "          curl -fsSL $OLLAMA_INSTALL_URL | sh"
            exit 1
        fi

        read -r -p "Установить Ollama сейчас? [y/N] " ANSWER
        case "$ANSWER" in
            [yY]|[yY][eE][sS])
                echo "[INFO] Устанавливаю Ollama из $OLLAMA_INSTALL_URL ..."
                if ! curl -fsSL "$OLLAMA_INSTALL_URL" | sh; then
                    echo "[ERROR] Не удалось установить Ollama."
                    echo "        Возможные причины:"
                    echo "          - нет доступа к ollama.com (проверьте прокси/сеть)"
                    echo "          - нет прав на установку (install.sh использует sudo)"
                    echo "        Попробуйте вручную: curl -fsSL $OLLAMA_INSTALL_URL | sh"
                    exit 1
                fi

                # Обновляем PATH на случай, если ollama появилась только что
                hash -r 2>/dev/null || true
                if ! command -v ollama >/dev/null 2>&1; then
                    for p in /usr/local/bin /usr/bin "$HOME/.local/bin"; do
                        if [ -x "$p/ollama" ]; then
                            export PATH="$p:$PATH"
                            break
                        fi
                    done
                fi

                if ! command -v ollama >/dev/null 2>&1; then
                    echo "[ERROR] ollama установлена, но не найдена в PATH."
                    echo "        Перелогиньтесь или добавьте путь в PATH."
                    exit 1
                fi
                echo "[OK] Ollama установлена: $(ollama --version 2>/dev/null || echo 'версия неизвестна')"
                ;;
            *)
                echo "[ERROR] Ollama нужна для:"
                echo "          - эмбеддингов (EMBEDDING_PROVIDER=ollama, по умолчанию)"
                echo "          - генерации ответа (--use-ollama)"
                echo "        Установите вручную: curl -fsSL $OLLAMA_INSTALL_URL | sh"
                echo "        Или переключитесь на sentence-transformers:"
                echo "          EMBEDDING_PROVIDER=st EMBEDDING_MODEL=all-MiniLM-L6-v2 EMBEDDING_DIM=384 $0"
                exit 1
                ;;
        esac
    fi

    # 7.2 Запуск сервиса, если API не отвечает
    ollama_ready() { curl -s --max-time 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

    if ! ollama_ready; then
        echo "[INFO] Ollama не отвечает. Пробую запустить сервис..."
        if command -v systemctl >/dev/null 2>&1; then
            sudo systemctl start ollama 2>/dev/null || true
        fi
        if ! ollama_ready; then
            echo "[INFO] Запускаю ollama serve в фоне..."
            nohup ollama serve >/tmp/ollama_serve.log 2>&1 &
        fi
        echo -n "[INFO] Ожидание готовности Ollama"
        for _ in $(seq 1 60); do
            ollama_ready && { echo " — готов"; break; }
            echo -n "."; sleep 1
        done
        ollama_ready || {
            echo
            echo "[ERROR] Ollama не ответила за 60 секунд."
            echo "        Проверьте: sudo systemctl status ollama"
            echo "        Или: journalctl -e -u ollama | tail -30"
            echo "        Или: tail -30 /tmp/ollama_serve.log"
            exit 1
        }
    fi
    echo "[OK] Ollama доступна: $OLLAMA_URL"

    # 7.3 Эмбеддинг-модель
    if [ "$EMBEDDING_PROVIDER" = "ollama" ]; then
        echo "[INFO] Проверка модели эмбеддингов '$EMBEDDING_MODEL'..."
        if ollama list 2>/dev/null | awk '{print $1}' | grep -qE "^${EMBEDDING_MODEL}(:latest)?$"; then
            echo "[OK] Модель $EMBEDDING_MODEL уже загружена."
        else
            echo "[INFO] Загрузка модели $EMBEDDING_MODEL (может занять минуты)..."
            if ! ollama pull "$EMBEDDING_MODEL"; then
                echo "[ERROR] Не удалось загрузить '$EMBEDDING_MODEL'."
                echo "        Проверьте имя на https://ollama.com/library"
                exit 1
            fi
            echo "[OK] Модель $EMBEDDING_MODEL загружена."
        fi
    fi

    # 7.4 LLM для генерации
    if [ "$USE_OLLAMA" -eq 1 ]; then
        echo "[INFO] Проверка LLM '$OLLAMA_MODEL'..."
        if ollama list 2>/dev/null | awk '{print $1}' | grep -qE "^${OLLAMA_MODEL}(:latest)?$"; then
            echo "[OK] Модель $OLLAMA_MODEL уже загружена."
        else
            echo "[INFO] Загрузка модели $OLLAMA_MODEL (может занять минуты)..."
            if ! ollama pull "$OLLAMA_MODEL"; then
                echo "[ERROR] Не удалось загрузить '$OLLAMA_MODEL'."
                exit 1
            fi
            echo "[OK] Модель $OLLAMA_MODEL загружена."
        fi
    fi
else
    echo "[INFO] Ollama не требуется (EMBEDDING_PROVIDER=$EMBEDDING_PROVIDER, --use-ollama не задан)."
fi
# =================================================================

# 8. Индексация через build_index.py
if [ ! -d "$KB_DIR" ]; then
    echo "[WARN] Папка $KB_DIR не найдена — пропускаю индексацию."
else
    echo "[INFO] Запуск индексатора build_index.py..."
    INDEXER_ARGS=(
        "--qdrant-url" "$QDRANT_URL"
        "--collection" "$COLLECTION_NAME"
        "--kb-dir" "$KB_DIR"
        "--embedding-provider" "$EMBEDDING_PROVIDER"
        "--embedding-model" "$EMBEDDING_MODEL"
        "--embedding-dim" "$EMBEDDING_DIM"
        "--ollama-url" "$OLLAMA_URL"
        "--chunk-size" "$CHUNK_SIZE"
        "--chunk-overlap" "$CHUNK_OVERLAP"
        "--embed-batch" "$EMBED_BATCH"
        "--upsert-batch" "$UPSERT_BATCH"
        "--on-injection" "$ON_INJECTION"
    )
    if [ "$REINDEX" -eq 1 ]; then
        INDEXER_ARGS+=("--reindex")
    fi
    if [ "$SAFETY_IN" -eq 1 ]; then
        INDEXER_ARGS+=("--safety-in")
    fi
    python -u build_index.py "${INDEXER_ARGS[@]}"
fi

# 9. test_rag.py
[ -f "test_rag.py" ] || { echo "[ERROR] test_rag.py не найден."; exit 1; }

echo "[INFO] Запуск test_rag.py..."
export EMBEDDING_PROVIDER EMBEDDING_MODEL EMBEDDING_DIM OLLAMA_URL OLLAMA_MODEL QDRANT_URL COLLECTION_NAME
export STRICT_CONTEXT MIN_SCORE NO_ANSWER QUERIES_FILE COT COT_MARKER TOP_K
export FEW_SHOT_INTERACTIVE SAFE_PROMPT ON_REGEX_INJECTION_SCAN ON_INJECTION
export SAFETY_IN SAFETY_OUT NO_CHUNKS

TEST_ARGS=("${PASSTHROUGH_ARGS[@]}")
TEST_ARGS+=("--embedding-provider" "$EMBEDDING_PROVIDER")
TEST_ARGS+=("--embedding-model" "$EMBEDDING_MODEL")
TEST_ARGS+=("--embedding-dim" "$EMBEDDING_DIM")
if [ "$USE_OLLAMA" -eq 1 ]; then
    TEST_ARGS+=("--model" "$OLLAMA_MODEL")
fi
if [ "$STRICT_CONTEXT" -eq 1 ]; then
    TEST_ARGS+=("--strict-context")
fi
if [ "$MIN_SCORE" != "0.0" ]; then
    TEST_ARGS+=("--min-score" "$MIN_SCORE")
fi
TEST_ARGS+=("--top-k" "$TOP_K")
if [ "$COT" -eq 1 ]; then
    TEST_ARGS+=("--cot" "--cot-marker" "$COT_MARKER")
fi
if [ "$SAFE_PROMPT" -eq 1 ] && [[ ! " ${PASSTHROUGH_ARGS[*]} " =~ " --safe-prompt " ]]; then
    TEST_ARGS+=("--safe-prompt")
fi
if [ "$ON_REGEX_INJECTION_SCAN" -eq 1 ]; then
    if [[ ! " ${PASSTHROUGH_ARGS[*]} " =~ " --on_regex_injection-scan " ]]; then
        TEST_ARGS+=("--on_regex_injection-scan")
    fi
    TEST_ARGS+=("--on-injection" "$ON_INJECTION")
fi
if [ "$SAFETY_IN" -eq 1 ] && [[ ! " ${PASSTHROUGH_ARGS[*]} " =~ " --safety-in " ]]; then
    TEST_ARGS+=("--safety-in")
fi
if [ "$SAFETY_OUT" -eq 1 ] && [[ ! " ${PASSTHROUGH_ARGS[*]} " =~ " --safety-out " ]]; then
    TEST_ARGS+=("--safety-out")
fi
if [ "$NO_CHUNKS" -eq 1 ] && [[ ! " ${PASSTHROUGH_ARGS[*]} " =~ " --no-chunks " ]]; then
    TEST_ARGS+=("--no-chunks")
fi
if [ "$FEW_SHOT_INTERACTIVE" -eq 1 ] && [[ ! " ${PASSTHROUGH_ARGS[*]} " =~ " --few-shot-interactive " ]]; then
    TEST_ARGS+=("--few-shot-interactive")
fi
if [ -n "$QUERIES_FILE" ]; then
    TEST_ARGS+=("--queries-file" "$QUERIES_FILE")
fi

python -u test_rag.py "${TEST_ARGS[@]}"

deactivate 2>/dev/null || true
