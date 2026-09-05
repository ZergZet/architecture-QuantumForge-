import json
import random
import re
import argparse
from collections import defaultdict

# ---------- Список служебных слов (русский язык) ----------
STOP_WORDS = set([
    # Предлоги
    "без", "безо", "близ", "в", "во", "вместо", "вне", "внутри", "возле", "вокруг",
    "впереди", "вслед", "для", "до", "за", "из", "из-за", "из-под", "к", "ко",
    "кроме", "между", "на", "над", "надо", "о", "об", "обо", "от", "ото", "перед",
    "передо", "по", "под", "подо", "при", "про", "ради", "с", "со", "сквозь",
    "среди", "у", "через", "чрез",
    # Союзы
    "а", "да", "и", "или", "либо", "ни", "но", "однако", "зато", "же", "если",
    "чтобы", "что", "как", "так", "потому", "поскольку", "оттого", "также", "тоже",
    "не", "ни", "ни...ни", "как...так", "не только...но и", "хотя", "будто", "словно",
    # Частицы
    "бы", "б", "же", "ли", "ль", "не", "ни", "вот", "вон", "ведь", "даже", "уже",
    "именно", "как", "так", "что", "только", "лишь", "почти", "уж", "авось", "небось",
    "прямо", "просто", "точно", "едва", "чуть", "чуть-чуть", "кое", "кой", "таки",
    # Местоимения (личные, возвратные, притяжательные, указательные, вопросительные, относительные, определительные, отрицательные, неопределённые)
    "я", "мы", "ты", "вы", "он", "она", "оно", "они", "себя", "свой", "твой", "ваш",
    "наш", "мой", "его", "её", "их", "этот", "тот", "такой", "таков", "столько",
    "кто", "что", "какой", "который", "чей", "сколько", "весь", "всякий", "каждый",
    "сам", "самый", "иной", "другой", "никто", "ничто", "некого", "нечего", "некто",
    "нечто", "некоторый", "кое-кто", "кое-что", "кое-какой", "чей-то", "что-то",
    "кто-то", "какой-то", "сколько-то", "несколько", "много", "мало",
    # Междометия
    "ах", "ой", "увы", "эх", "ну", "браво", "караул", "эй", "ого", "ба", "ау",
    "гм", "тьфу", "ура", "чу", "шш", "цц", "брысь", "кис-кис", "цып-цып",
    # Другие короткие неизменяемые (наречия, вводные слова, союзные слова)
    "вот", "вон", "там", "тут", "здесь", "туда", "сюда", "оттуда", "отсюда",
    "теперь", "сейчас", "тогда", "потом", "затем", "всегда", "иногда", "где",
    "куда", "откуда", "почему", "зачем", "как", "так", "столь", "настолько",
    "очень", "весьма", "крайне", "чуть", "едва", "вряд", "ли", "разве", "неужели",
    "да", "нет", "конечно", "безусловно", "разумеется", "пожалуй", "наверное",
    "вероятно", "может", "быть", "кажется", "видимо", "значит", "итак", "наконец",
    "впрочем", "однако", "все-таки", "все же", "тем не менее", "следовательно",
    "во-первых", "во-вторых", "наконец", "также", "кроме того", "более того",
    "менее", "более", "самый", "весь", "всякий", "иной", "другой", "тот", "этот"
])

# ---------- Марковская модель ----------
def build_markov_model(words, order=2):
    model = defaultdict(list)
    for word in words:
        padded = '^' * order + word + '$' * order
        for i in range(len(padded) - order):
            key = tuple(padded[i:i+order])
            next_char = padded[i+order]
            model[key].append(next_char)
    return model

def generate_word(model, order=2, target_len=None, max_attempts=100):
    start_keys = [key for key in model.keys() if key[0] == '^']
    if not start_keys:
        return None
    for _ in range(max_attempts):
        current = random.choice(start_keys)
        chars = [c for c in current if c != '^']
        while True:
            next_chars = model.get(tuple(current))
            if not next_chars:
                break
            next_char = random.choice(next_chars)
            if next_char == '$':
                break
            chars.append(next_char)
            current = tuple(current[1:] + (next_char,))
        word = ''.join(chars).replace('^', '')
        if target_len is None:
            if word:
                return word
        else:
            if len(word) == target_len:
                return word
    return None

# ---------- Вспомогательные функции ----------
def is_stop_word(word):
    """Проверяет, является ли слово служебным (регистронезависимо)."""
    return word.lower() in STOP_WORDS

def extract_all_words_from_category(items):
    """Извлекает все слова (токены) из списка названий."""
    words = []
    for item in items:
        if isinstance(item, str):
            found = re.findall(r"[А-Яа-яA-Za-z']+", item)
            words.extend(found)
    return words

def generate_new_name(original, model, order=2, max_attempts=50):
    """
    Генерирует новое название, заменяя все знаменательные слова на новые,
    сохраняя служебные слова и разделители.
    """
    pattern = re.compile(r"[А-Яа-яA-Za-z']+")
    matches = list(pattern.finditer(original))
    if not matches:
        return original

    new_parts = []
    last_end = 0
    for match in matches:
        start, end = match.span()
        original_word = match.group()
        # Добавляем разделитель перед словом
        if start > last_end:
            new_parts.append(original[last_end:start])

        # Проверяем, является ли слово служебным
        if is_stop_word(original_word):
            new_word = original_word  # сохраняем как есть
        else:
            # Генерируем новое слово той же длины
            target_len = len(original_word)
            new_word = None
            for _ in range(max_attempts):
                w = generate_word(model, order, target_len=target_len)
                if w is not None and w != original_word:
                    new_word = w
                    break
            if new_word is None:
                new_word = original_word  # оставляем оригинал, если не удалось сгенерировать
        new_parts.append(new_word)
        last_end = end

    # Добавляем хвост после последнего слова
    if last_end < len(original):
        new_parts.append(original[last_end:])

    return ''.join(new_parts)

# ---------- Основная функция генерации ----------
def generate_mapping(entities_by_category, order=2):
    mapping = {}
    for category, names in entities_by_category.items():
        # Собираем все слова из названий категории (только знаменательные?)
        all_words = extract_all_words_from_category(names)
        if len(all_words) < 2:
            print(f"⚠️ Категория '{category}' содержит слишком мало слов, пропускаем.")
            continue
        model = build_markov_model(all_words, order)

        for original in names:
            new_name = generate_new_name(original, model, order)
            # Проверяем уникальность нового значения
            attempts = 0
            while new_name in mapping.values() and attempts < 10:
                new_name = generate_new_name(original, model, order)
                attempts += 1
            mapping[original] = new_name

    return mapping

# ---------- Загрузка и извлечение сущностей ----------
def load_entities(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entities_by_category = {}
    for category, items in data.items():
        if category == "даты_и_события":
            events = [item["событие"] for item in items if isinstance(item, dict) and "событие" in item]
            if events:
                entities_by_category[category] = events
        elif isinstance(items, list):
            strings = [item for item in items if isinstance(item, str)]
            if strings:
                entities_by_category[category] = strings
    return entities_by_category

# ---------- Главная ----------
def main():
    parser = argparse.ArgumentParser(description='Генерирует словарь замен с сохранением служебных слов.')
    parser.add_argument('json_file', help='Путь к исходному JSON')
    parser.add_argument('-o', '--output', default='mapping.json', help='Выходной JSON')
    parser.add_argument('--order', type=int, default=2, help='Порядок марковской цепи')
    args = parser.parse_args()

    entities = load_entities(args.json_file)
    if not entities:
        print("Не найдено сущностей для генерации.")
        return

    mapping = generate_mapping(entities, args.order)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"✅ Сгенерировано {len(mapping)} пар. Сохранено в {args.output}")

if __name__ == "__main__":
    main()