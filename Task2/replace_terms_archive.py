import os
import re
import json
import argparse
from pathlib import Path

def load_mapping(mapping_file):
    with open(mapping_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def case_preserving_replace(text, pattern, replacement):
    escaped = re.escape(pattern)
    regex = re.compile(escaped, re.IGNORECASE)
    def replace_match(match):
        matched = match.group(0)
        if matched.isupper():
            return replacement.upper()
        elif matched.istitle():
            return replacement.capitalize()
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

def process_files(input_dir, output_dir, mapping, case_sensitive=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path in input_path.iterdir():
        if not file_path.is_file():
            continue

        # Читаем содержимое
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        new_text = apply_replacements(text, mapping, case_sensitive)

        # Обработка имени файла: подчёркивания → пробелы для многословных замен
        stem = file_path.stem
        suffix = file_path.suffix
        stem_with_spaces = stem.replace('_', ' ')
        new_stem_with_spaces = apply_replacements(stem_with_spaces, mapping, case_sensitive)
        new_stem = new_stem_with_spaces.replace(' ', '_')
        new_name = new_stem + suffix

        out_file = output_path / new_name
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(new_text)

        print(f"Обработан: {file_path.name} -> {out_file.name}")

def main():
    parser = argparse.ArgumentParser(
        description="Заменяет термины в текстовых файлах и переименовывает их на основе словаря."
    )
    parser.add_argument("--input-dir", default="lotr_articles", help="Папка с исходными документами")
    parser.add_argument("--mapping", default="mapping.json", help="JSON-файл со словарём замен")
    parser.add_argument("--output-dir", default=None, help="Папка для сохранения новых документов")
    parser.add_argument("--case-sensitive", action="store_true", help="Учитывать регистр при замене")
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
    print(f"Учитывать регистр: {args.case_sensitive}")
    print("Начинаем обработку...")

    process_files(args.input_dir, args.output_dir, mapping, args.case_sensitive)
    print("Готово!")

if __name__ == "__main__":
    main()