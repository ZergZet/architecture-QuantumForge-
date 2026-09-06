import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_mapping(mapping_file):
    """Загружает словарь, очищая ключи и значения от лишних пробелов."""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    cleaned = {}
    for key, value in raw.items():
        clean_key = key.strip()
        clean_value = value.strip()
        if clean_key:
            clean_key = re.sub(r'\s+', ' ', clean_key)
            clean_value = re.sub(r'\s+', ' ', clean_value)
            cleaned[clean_key] = clean_value
    return cleaned

def expand_mapping(mapping):
    """Расширяет словарь, добавляя замены для отдельных слов из составных терминов."""
    expanded = dict(mapping)
    for key, value in mapping.items():
        key_words = re.findall(r"[А-Яа-яA-Za-z']+", key)
        value_words = re.findall(r"[А-Яа-яA-Za-z']+", value)
        if len(key_words) == len(value_words) and len(key_words) > 1:
            for kw, vw in zip(key_words, value_words):
                if kw not in expanded:
                    expanded[kw] = vw
    return expanded

def case_preserving_replace(text, pattern, replacement):
    escaped = re.escape(pattern)
    regex = re.compile(escaped, re.IGNORECASE)
    def replace_match(match):
        matched = match.group(0)
        if matched.isupper():
            return replacement.upper()
        elif matched.istitle():
            return ' '.join(word.capitalize() for word in replacement.split())
        else:
            return replacement.lower()
    return regex.sub(replace_match, text)

def apply_replacements(text, mapping, case_sensitive=False):
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    for key in sorted_keys:
        replacement = mapping[key]
        if case_sensitive:
            text = text.replace(key, replacement)
        else:
            text = case_preserving_replace(text, key, replacement)
    return text

def process_files(input_dir, output_dir, mapping, case_sensitive=False, expand_words=False):
    if expand_words:
        original_count = len(mapping)
        mapping = expand_mapping(mapping)
        print(f"Расширение словаря: {original_count} -> {len(mapping)} замен.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Для отслеживания конфликтов имён
    used_names = defaultdict(int)
    total_files = 0
    processed = 0

    for file_path in input_path.iterdir():
        if not file_path.is_file():
            continue
        total_files += 1

        # 1. Содержимое
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        new_text = apply_replacements(text, mapping, case_sensitive)

        # 2. Имя файла
        stem = file_path.stem
        suffix = file_path.suffix
        stem_with_spaces = stem.replace('_', ' ').strip()
        new_stem_with_spaces = apply_replacements(stem_with_spaces, mapping, case_sensitive)
        new_stem_with_spaces = re.sub(r'\s+', ' ', new_stem_with_spaces).strip()
        new_stem = new_stem_with_spaces.replace(' ', '_')
        new_name = new_stem + suffix

        # Проверяем конфликт имён
        if new_name in used_names:
            used_names[new_name] += 1
            # Добавляем суффикс: _1, _2, ...
            base, ext = os.path.splitext(new_name)
            new_name = f"{base}_{used_names[new_name]}{ext}"
            print(f"⚠️ Конфликт имён: {file_path.name} -> {new_name} (суффикс добавлен)")
        else:
            used_names[new_name] = 1

        out_file = output_path / new_name
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(new_text)
        processed += 1
        print(f"Обработан: {file_path.name} -> {out_file.name}")

    print(f"\nВсего файлов: {total_files}, обработано: {processed}")

def main():
    parser = argparse.ArgumentParser(
        description="Заменяет термины в текстовых файлах и переименовывает их на основе словаря."
    )
    parser.add_argument("--input-dir", default="lotr_articles", help="Папка с исходными документами")
    parser.add_argument("--mapping", default="mapping.json", help="JSON-файл со словарём замен")
    parser.add_argument("--output-dir", default=None, help="Папка для сохранения новых документов")
    parser.add_argument("--case-sensitive", action="store_true", help="Учитывать регистр при замене")
    parser.add_argument("--expand-words", action="store_true", help="Расширять словарь отдельными словами")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir + "_new"

    try:
        mapping = load_mapping(args.mapping)
    except FileNotFoundError:
        print(f"Ошибка: файл словаря '{args.mapping}' не найден.")
        return
    except json.JSONDecodeError:
        print(f"Ошибка: файл '{args.mapping}' содержит невалидный JSON.")
        return

    if not mapping:
        print("Словарь замен пуст. Ничего не делаем.")
        return

    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: папка '{args.input_dir}' не существует.")
        return

    print(f"Входная папка: {args.input_dir}")
    print(f"Выходная папка: {args.output_dir}")
    print(f"Словарь замен: {args.mapping} (записей: {len(mapping)})")
    print(f"Регистр: {'учёт' if args.case_sensitive else 'игнорирование'}")
    print(f"Расширение слова: {'да' if args.expand_words else 'нет'}")
    print("Начинаем обработку...\n")

    process_files(args.input_dir, args.output_dir, mapping,
                  args.case_sensitive, args.expand_words)
    print("Готово!")

if __name__ == "__main__":
    main()