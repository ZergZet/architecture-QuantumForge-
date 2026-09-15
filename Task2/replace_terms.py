import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ---------- Импорт морфологического анализатора ----------
MORPH_AVAILABLE = False
MORPH_ANALYZER = None

try:
    import pymorphy3 as pymorphy
    MORPH_AVAILABLE = True
except ImportError:
    try:
        import pymorphy2 as pymorphy
        MORPH_AVAILABLE = True
    except ImportError:
        pass

if not MORPH_AVAILABLE:
    print("⚠️ pymorphy2/pymorphy3 не установлен. Морфологическая замена недоступна.")
    print("   Установите: pip install pymorphy3 (рекомендуется для Python 3.12+)")
else:
    try:
        MORPH_ANALYZER = pymorphy.MorphAnalyzer()
    except Exception as e:
        print(f"⚠️ Ошибка инициализации морфологического анализатора: {e}")
        MORPH_AVAILABLE = False
        MORPH_ANALYZER = None

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def remove_accents(text):
    return re.sub(r'\u0301', '', text)

def load_mapping(mapping_file):
    with open(mapping_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    cleaned = {}
    for key, value in raw.items():
        clean_key = key.strip()
        clean_value = value.strip()
        if clean_key:
            clean_key = remove_accents(clean_key)
            clean_key = re.sub(r'\s+', ' ', clean_key)
            clean_value = re.sub(r'\s+', ' ', clean_value)
            cleaned[clean_key] = clean_value
    return cleaned

def expand_mapping(mapping):
    expanded = dict(mapping)
    for key, value in mapping.items():
        key_words = re.findall(r"[А-Яа-яA-Za-z']+", key)
        value_words = re.findall(r"[А-Яа-яA-Za-z']+", value)
        if len(key_words) == len(value_words) and len(key_words) > 1:
            for kw, vw in zip(key_words, value_words):
                kw_clean = remove_accents(kw.lower())
                if kw_clean not in expanded:
                    expanded[kw_clean] = vw
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

def substring_replace(text, pattern, replacement):
    escaped = re.escape(pattern)
    regex = re.compile(escaped, re.IGNORECASE)
    def replace_match(match):
        matched = match.group(0)
        if matched.isupper():
            return replacement.upper()
        elif matched.istitle() or matched[0].isupper():
            return replacement.capitalize()
        else:
            return replacement.lower()
    return regex.sub(replace_match, text)

def build_lemma_mapping(single_word_mapping):
    if not MORPH_AVAILABLE or MORPH_ANALYZER is None:
        return {}
    morph = MORPH_ANALYZER
    lemma_mapping = {}
    for word, replacement in single_word_mapping.items():
        if ' ' in replacement:
            continue
        word_clean = remove_accents(word.lower())
        parsed = morph.parse(word_clean)
        if not parsed:
            continue
        lemma_key = parsed[0].normal_form

        repl_clean = remove_accents(replacement.lower())
        parsed_repl = morph.parse(repl_clean)
        if not parsed_repl:
            continue
        lemma_repl = parsed_repl[0].normal_form

        if lemma_key not in lemma_mapping:
            lemma_mapping[lemma_key] = lemma_repl
    return lemma_mapping

def apply_morphological_replacements(text, single_word_mapping, case_sensitive=False):
    if not MORPH_AVAILABLE or MORPH_ANALYZER is None or not single_word_mapping:
        return text

    morph = MORPH_ANALYZER
    lemma_to_replacement = build_lemma_mapping(single_word_mapping)

    def replace_word(match):
        word = match.group(0)
        word_clean = remove_accents(word)
        parsed = morph.parse(word_clean)
        if not parsed:
            return word
        para = parsed[0]
        lemma = para.normal_form
        if lemma not in lemma_to_replacement:
            return word

        replacement_lemma = lemma_to_replacement[lemma]
        replacement_clean = remove_accents(replacement_lemma)
        new_parsed = morph.parse(replacement_clean)
        if not new_parsed:
            return replacement_lemma.lower()

        tags = para.tag
        best_form = None
        best_score = -1
        for p in new_parsed:
            score = 0
            if tags.POS == p.tag.POS:
                score += 1
            if tags.case == p.tag.case:
                score += 1
            if tags.number == p.tag.number:
                score += 1
            if tags.gender == p.tag.gender:
                score += 1
            if tags.tense == p.tag.tense:
                score += 1
            if score > best_score:
                best_score = score
                best_form = p.word
        if best_form is None:
            best_form = replacement_lemma.lower()

        if word.isupper():
            return best_form.upper()
        elif word.istitle():
            return best_form.capitalize()
        else:
            return best_form.lower()

    pattern = re.compile(r"[А-Яа-яA-Za-z']+")
    return pattern.sub(replace_word, text)

def apply_replacements(text, mapping, case_sensitive=False, use_morphology=False, use_substring=False):
    multi_word_keys = [k for k in mapping.keys() if ' ' in k]
    single_word_keys = [k for k in mapping.keys() if ' ' not in k]

    multi_word_keys.sort(key=len, reverse=True)
    single_word_keys.sort(key=len, reverse=True)

    # 1. Составные термины
    for key in multi_word_keys:
        replacement = mapping[key]
        if case_sensitive:
            text = text.replace(key, replacement)
        else:
            text = case_preserving_replace(text, key, replacement)
        debug_print(f"Составная замена: '{key}' -> '{replacement}'")

    # 2. Однословные
    if use_substring:
        for key in single_word_keys:
            replacement = mapping[key]
            if case_sensitive:
                text = text.replace(key, replacement)
            else:
                old_text = text
                text = substring_replace(text, key, replacement)
                if old_text != text:
                    debug_print(f"Подстрока: '{key}' -> '{replacement}'")
    else:
        if use_morphology and MORPH_AVAILABLE and MORPH_ANALYZER is not None:
            single_word_mapping = {k: mapping[k] for k in single_word_keys}
            text = apply_morphological_replacements(text, single_word_mapping, case_sensitive)
            debug_print("Применена морфологическая замена для однословных ключей")
        else:
            for key in single_word_keys:
                replacement = mapping[key]
                if case_sensitive:
                    text = text.replace(key, replacement)
                else:
                    text = case_preserving_replace(text, key, replacement)
                debug_print(f"Точная замена: '{key}' -> '{replacement}'")

    return text

def process_files(input_dir, output_dir, mapping, case_sensitive=False, expand_words=False,
                  use_morphology=False, use_substring=False):
    if expand_words:
        original_count = len(mapping)
        mapping = expand_mapping(mapping)
        print(f"Расширение словаря: {original_count} -> {len(mapping)} замен.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    used_names = defaultdict(int)
    total_files = 0
    processed = 0

    for file_path in input_path.iterdir():
        if not file_path.is_file():
            continue
        total_files += 1

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text_no_accents = remove_accents(text)
        new_text_no_accents = apply_replacements(
            text_no_accents, mapping, case_sensitive, use_morphology, use_substring
        )
        new_text = new_text_no_accents

        stem = file_path.stem
        suffix = file_path.suffix
        stem_with_spaces = stem.replace('_', ' ').strip()
        stem_with_spaces_no_accents = remove_accents(stem_with_spaces)
        new_stem_with_spaces = apply_replacements(
            stem_with_spaces_no_accents, mapping, case_sensitive, use_morphology, use_substring
        )
        new_stem_with_spaces = re.sub(r'\s+', ' ', new_stem_with_spaces).strip()
        new_stem = new_stem_with_spaces.replace(' ', '_')
        new_name = new_stem + suffix

        if new_name in used_names:
            used_names[new_name] += 1
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
    global DEBUG
    parser = argparse.ArgumentParser(description="Заменяет термины в текстовых файлах и переименовывает их.")
    parser.add_argument("--input-dir", default="lotr_articles")
    parser.add_argument("--mapping", default="mapping.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--case-sensitive", action="store_true")
    parser.add_argument("--expand-words", action="store_true")
    parser.add_argument("--no-morphology", action="store_true",
                        help="Отключить морфологическую замену (по умолчанию включена)")
    parser.add_argument("--substring", action="store_true",
                        help="Включить замену подстрок (по умолчанию выключена, рекомендуется использовать морфологию)")
    parser.add_argument("--debug", action="store_true", help="Выводить отладочную информацию о заменах")
    args = parser.parse_args()

    DEBUG = args.debug

    if args.output_dir is None:
        args.output_dir = args.input_dir + "_new"

    use_morphology = not args.no_morphology and MORPH_AVAILABLE and MORPH_ANALYZER is not None
    use_substring = args.substring  # теперь по умолчанию False

    try:
        mapping = load_mapping(args.mapping)
    except FileNotFoundError:
        print(f"Ошибка: файл словаря '{args.mapping}' не найден.")
        return
    except json.JSONDecodeError:
        print(f"Ошибка: файл '{args.mapping}' содержит невалидный JSON.")
        return

    if not mapping:
        print("Словарь замен пуст.")
        return

    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: папка '{args.input_dir}' не существует.")
        return

    print(f"Входная папка: {args.input_dir}")
    print(f"Выходная папка: {args.output_dir}")
    print(f"Словарь замен: {args.mapping} (записей: {len(mapping)})")
    print(f"Регистр: {'учёт' if args.case_sensitive else 'игнорирование'}")
    print(f"Расширение слова: {'да' if args.expand_words else 'нет'}")
    print(f"Морфология: {'вкл' if use_morphology else 'выкл'}")
    print(f"Замена подстрок: {'вкл' if use_substring else 'выкл'}")
    if use_morphology and not use_substring:
        print("✅ Морфология включена, подстрока отключена – замена словоформ будет корректной.")
    if args.debug:
        print("🐞 Режим отладки включён – будут показаны все замены.")
    print("Начинаем обработку...\n")

    process_files(args.input_dir, args.output_dir, mapping,
                  args.case_sensitive, args.expand_words, use_morphology, use_substring)
    print("Готово!")

if __name__ == "__main__":
    main()
