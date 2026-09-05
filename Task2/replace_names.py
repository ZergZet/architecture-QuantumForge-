import json
import random
from collections import defaultdict

# --- Марковская модель (та же, что в генераторе) ---
def build_markov_model(names, order=2):
    model = defaultdict(list)
    for name in names:
        padded = '^' * order + name + '$' * order
        for i in range(len(padded) - order):
            key = tuple(padded[i:i+order])
            next_char = padded[i+order]
            model[key].append(next_char)
    return model

def generate_name(model, order=2, min_len=3, max_len=15):
    start_keys = [key for key in model.keys() if key[0] == '^']
    if not start_keys:
        return ""
    current = random.choice(start_keys)
    name_chars = [c for c in current if c != '^']
    while len(name_chars) < max_len:
        next_chars = model.get(tuple(current))
        if not next_chars:
            break
        next_char = random.choice(next_chars)
        if next_char == '$':
            break
        name_chars.append(next_char)
        current = tuple(current[1:] + (next_char,))
    name = ''.join(name_chars).split('$')[0].replace('^', '')
    if len(name) < min_len:
        return generate_name(model, order, min_len, max_len)
    return name

# --- Загрузка JSON ---
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def replace_entities(data, order=2, min_len=3, max_len=15):
    new_data = {}
    for category, items in data.items():
        if category == "даты_и_события":
            # Обрабатываем список словарей
            new_events = []
            events = [item["событие"] for item in items if isinstance(item, dict) and "событие" in item]
            if not events:
                new_data[category] = items
                continue
            model = build_markov_model(events, order)
            for event_dict in items:
                if isinstance(event_dict, dict) and "событие" in event_dict:
                    new_event = generate_name(model, order, min_len, max_len)
                    # Сохраняем дату без изменений
                    new_events.append({"дата": event_dict["дата"], "событие": new_event})
                else:
                    new_events.append(event_dict)
            new_data[category] = new_events
        elif isinstance(items, list):
            # Извлекаем все строки
            strings = [item for item in items if isinstance(item, str)]
            if not strings:
                new_data[category] = items
                continue
            model = build_markov_model(strings, order)
            new_list = []
            for item in items:
                if isinstance(item, str):
                    new_name = generate_name(model, order, min_len, max_len)
                    new_list.append(new_name)
                else:
                    new_list.append(item)
            new_data[category] = new_list
        else:
            new_data[category] = items
    return new_data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', help='Путь к исходному JSON')
    parser.add_argument('-o', '--output', default='lotr_data_new.json', help='Файл для сохранения')
    parser.add_argument('--order', type=int, default=2, help='Порядок марковской цепи')
    parser.add_argument('--min_len', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=15)
    args = parser.parse_args()

    data = load_data(args.json_file)
    new_data = replace_entities(data, args.order, args.min_len, args.max_len)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Новый JSON сохранён в {args.output}")

if __name__ == "__main__":
    main()