import json
import random
import argparse
from collections import defaultdict

def load_names_from_json(filepath):
    """
    Загружает JSON и извлекает все названия из списков, а также события из дат.
    Возвращает словарь {категория: [список_названий]}.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    names_by_category = {}
    for category, items in data.items():
        if category == "даты_и_события":
            # Извлекаем только поле "событие"
            events = [item["событие"] for item in items if isinstance(item, dict) and "событие" in item]
            if events:
                names_by_category[category] = events
        elif isinstance(items, list):
            # Фильтруем только строки
            strings = [item for item in items if isinstance(item, str)]
            if strings:
                names_by_category[category] = strings
    return names_by_category

def build_markov_model(names, order=2):
    """
    Строит марковскую модель порядка order (по символам).
    Возвращает словарь: (кортеж_предыдущих_символов) -> список_возможных_следующих_символов.
    """
    model = defaultdict(list)
    for name in names:
        # Добавляем специальные маркеры начала и конца
        padded = '^' * order + name + '$' * order
        for i in range(len(padded) - order):
            key = tuple(padded[i:i+order])
            next_char = padded[i+order]
            model[key].append(next_char)
    return model

def generate_name(model, order=2, min_len=3, max_len=15):
    """
    Генерирует одно случайное имя, используя марковскую модель.
    Начинает с случайного ключа (кортежа), который начинается с '^'.
    Заканчивает, когда встретится '$' или превышена max_len.
    """
    # Выбираем случайный стартовый ключ (должен начинаться с '^')
    start_keys = [key for key in model.keys() if key[0] == '^']
    if not start_keys:
        return ""
    current = random.choice(start_keys)
    name_chars = list(current)  # убираем маркеры начала
    # убираем '^' из начала
    name_chars = [c for c in name_chars if c != '^']

    # Генерируем цепочку
    while len(name_chars) < max_len:
        next_chars = model.get(tuple(current))
        if not next_chars:
            break
        next_char = random.choice(next_chars)
        if next_char == '$':
            break
        name_chars.append(next_char)
        # сдвигаем окно
        current = tuple(current[1:] + (next_char,))

    # Обрезаем до первого '$' (на всякий случай)
    name = ''.join(name_chars).split('$')[0]
    # Убираем возможные оставшиеся '^' в середине (если есть)
    name = name.replace('^', '')
    # Если имя слишком короткое или пустое, генерируем заново (рекурсия)
    if len(name) < min_len:
        return generate_name(model, order, min_len, max_len)
    return name

def main():
    parser = argparse.ArgumentParser(description='Генератор новых названий на основе JSON с сущностями LOTR.')
    parser.add_argument('json_file', help='Путь к JSON-файлу (например, lotr_data.json)')
    parser.add_argument('-n', '--count', type=int, default=5, help='Количество вариантов для каждой категории')
    parser.add_argument('-c', '--category', help='Конкретная категория (например, "персонажи"), если не указана — все')
    parser.add_argument('--order', type=int, default=2, help='Порядок марковской цепи (по умолчанию 2)')
    parser.add_argument('--min_len', type=int, default=3, help='Минимальная длина генерируемого названия')
    parser.add_argument('--max_len', type=int, default=15, help='Максимальная длина генерируемого названия')
    args = parser.parse_args()

    # Загружаем названия по категориям
    names_by_category = load_names_from_json(args.json_file)
    if not names_by_category:
        print("Не найдено названий в JSON.")
        return

    # Если указана конкретная категория, проверяем её наличие
    if args.category:
        if args.category not in names_by_category:
            print(f"Категория '{args.category}' не найдена. Доступны: {', '.join(names_by_category.keys())}")
            return
        categories = {args.category: names_by_category[args.category]}
    else:
        categories = names_by_category

    # Генерация для каждой категории
    for cat, names in categories.items():
        if len(names) < 2:
            print(f"⚠️ Категория '{cat}' содержит слишком мало примеров ({len(names)}), пропускаем.")
            continue

        print(f"\n===== {cat.upper()} =====")
        model = build_markov_model(names, order=args.order)

        generated = set()
        attempts = 0
        # Генерируем уникальные имена, пока не наберём нужное количество
        while len(generated) < args.count and attempts < args.count * 10:
            new_name = generate_name(model, args.order, args.min_len, args.max_len)
            if new_name and new_name not in generated:
                # Проверяем, что имя не совпадает с исходными (опционально)
                if new_name not in names:
                    generated.add(new_name)
            attempts += 1

        for i, name in enumerate(generated, 1):
            print(f"{i}. {name}")

if __name__ == "__main__":
    main()