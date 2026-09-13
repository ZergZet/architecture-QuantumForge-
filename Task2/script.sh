#!/usr/bin/env bash
set -euo pipefail

# Перейти в директорию, где лежит сам скрипт
cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
SCRIPT="Script.py"
INPUT_JSON="lotr_data.json"

# --- Китайское зеркало PyPI ---
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

echo "=================================================="
echo "  Установка окружения для Script.py"
echo "=================================================="

# 1. Проверка curl (нужен для проверок/загрузок, если понадобится)
if ! command -v curl >/dev/null 2>&1; then
    echo "[WARN] curl не найден. Возможно, потребуется установить:"
    echo "       sudo apt update && sudo apt install -y curl"
fi

# 2. Проверка python3
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[ERROR] $PYTHON не найден. Установите python3:"
    echo "  sudo apt update && sudo apt install -y python3"
    exit 1
fi
echo "[OK] $($PYTHON --version)"

# 3. Проверка модуля venv
if ! "$PYTHON" -c "import venv" >/dev/null 2>&1; then
    echo "[ERROR] Модуль venv недоступен. Установите:"
    echo "  sudo apt update && sudo apt install -y python3-venv python3-pip"
    exit 1
fi

# 4. Создание venv, если его нет или в нём нет pip/python
if [ ! -x "$VENV_DIR/bin/pip" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[INFO] Создание виртуального окружения $VENV_DIR..."
    rm -rf "$VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# 5. Активация venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[INFO] Зеркало PyPI: $PIP_INDEX_URL"

# 6. Установка зависимостей (пропускается, если уже стоят)
if python -c "import requests, bs4" >/dev/null 2>&1; then
    echo "[OK] Python-зависимости уже установлены, пропускаю pip."
else
    echo "[INFO] Обновление pip..."
    pip install --upgrade \
        --index-url "$PIP_INDEX_URL" \
        --trusted-host "$PIP_TRUSTED_HOST" \
        --timeout 120 --retries 10 \
        pip

    echo "[INFO] Установка Python-зависимостей (requests, beautifulsoup4)..."
    pip install --upgrade \
        --index-url "$PIP_INDEX_URL" \
        --trusted-host "$PIP_TRUSTED_HOST" \
        --timeout 300 --retries 10 --prefer-binary \
        requests beautifulsoup4
fi

# 7. Проверка входных данных
if [ ! -f "$INPUT_JSON" ]; then
    echo "[ERROR] Файл $INPUT_JSON не найден в $(pwd)."
    echo "        Положите его рядом со скриптом перед запуском."
    exit 1
fi
echo "[OK] Найден $INPUT_JSON"

# 8. Проверка самого скрипта
if [ ! -f "$SCRIPT" ]; then
    echo "[ERROR] Файл $SCRIPT не найден в $(pwd)"
    exit 1
fi

# 9. Запуск Script.py
echo "[INFO] Запуск $SCRIPT..."
echo "--------------------------------------------------"
python "$SCRIPT" "$@"

deactivate 2>/dev/null || true