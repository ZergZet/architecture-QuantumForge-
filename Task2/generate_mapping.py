import json
import random
import re
import argparse
from collections import defaultdict

# ---------- Список служебных слов (без изменений) ----------
STOP_WORDS = set([
    "без", "безо", "близ", "в", "во", "вместо", "вне", "внутри", "возле", "вокруг",
    "впереди", "вслед", "для", "до", "за", "из", "из-за", "из-под", "к", "ко",
    "кроме", "между", "на", "над", "надо", "о", "об", "обо", "от", "ото", "перед",
    "передо", "по", "под", "подо", "при", "про", "ради", "с", "со", "сквозь",
    "среди", "у", "через", "чрез",
    "а", "да", "и", "или", "либо", "ни", "но", "однако", "зато", "же", "если",
    "чтобы", "что", "как", "так", "потому", "поскольку", "оттого", "также", "тоже",
    "не", "ни", "ни...ни", "как...так", "не только...но и", "хотя", "будто", "словно",
    "бы", "б", "же", "ли", "ль", "не", "ни", "вот", "вон", "ведь", "даже", "уже",
    "именно", "как", "так", "что", "только", "лишь", "почти", "уж", "авось", "небось",
    "прямо", "просто", "точно", "едва", "чуть", "чуть-чуть", "кое", "кой", "таки",
    "я", "мы", "ты", "вы", "он", "она", "оно", "они", "себя", "свой", "твой", "ваш",
    "наш", "мой", "его", "её", "их", "этот", "тот", "такой", "таков", "столько",
    "кто", "что", "какой", "который", "чей", "сколько", "весь", "всякий", "каждый",
    "сам", "самый", "иной", "другой", "никто", "ничто", "некого", "нечего", "некто",
    "нечто", "некоторый", "кое-кто", "кое-что", "кое-какой", "чей-то", "что-то",
    "кто-то", "какой-то", "сколько-то", "несколько", "много", "мало",
    "ах", "ой", "увы", "эх", "ну", "браво", "караул", "эй", "ого", "ба", "ау",
    "гм", "тьфу", "ура", "чу", "шш", "цц", "брысь", "кис-кис", "цып-цып",
    "вот", "вон", "там", "тут", "здесь", "туда", "сюда", "оттуда", "отсюда",
    "теперь", "сейчас", "тогда", "потом", "затем", "всегда", "иногда", "где",
    "куда", "откуда", "почему", "зачем", "столь", "настолько",
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

def generate_word(model, order=2, max_attempts=100):
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
        if len(word) >= 2:
            return word
    return None

def fallback_replace(word):
    if len(word) < 2:
        return word + random.choice(['р', 'н', 'л', 'м'])
    chars = list(word)
    if len(chars) > 1:
        i, j = random.sample(range(len(chars)), 2)
        chars[i], chars[j] = chars[j], chars[i]
    vowels = 'аеёиоуыэюя'
    for idx, ch in enumerate(chars):
        if ch.lower() in vowels:
            new_vowel = random.choice(vowels)
            if new_vowel != ch:
                chars[idx] = new_vowel
                break
    suffix = random.choice(['р', 'н', 'л', 'м', 'ст', 'в', 'к'])
    new_word = ''.join(chars) + suffix
    if new_word == word:
        new_word = word + random.choice(['р', 'н', 'л', 'м'])
    return new_word

def is_stop_word(word):
    return word.lower() in STOP_WORDS

def extract_unique_words_from_names(names):
    words = set()
    for name in names:
        if isinstance(name, str):
            found = re.findall(r"[А-Яа-яA-Za-z']+", name)
            for w in found:
                if not is_stop_word(w):
                    words.add(w.lower())
    return words

def apply_word_mapping(text, word_mapping):
    pattern = re.compile(r"[А-Яа-яA-Za-z']+")
    def replace_match(match):
        word = match.group(0)
        lower_word = word.lower()
        if lower_word in word_mapping:
            replacement = word_mapping[lower_word]
            if word.isupper():
                return replacement.upper()
            elif word.istitle():
                return replacement.capitalize()
            else:
                return replacement.lower()
        else:
            return word
    return pattern.sub(replace_match, text)

def generate_unique_mapping_for_category(names, order=2, max_attempts=200):
    unique_words = extract_unique_words_from_names(names)
    if len(unique_words) < 1:
        return {}

    all_words = []
    for name in names:
        if isinstance(name, str):
            all_words.extend(re.findall(r"[А-Яа-яA-Za-z']+", name))
    if len(all_words) < 2:
        return {}

    model = build_markov_model(all_words, order)

    for attempt in range(max_attempts):
        word_mapping = {}
        for w in unique_words:
            new_w = None
            for _ in range(50):
                candidate = generate_word(model, order)
                if (candidate is not None and 
                    candidate != w and 
                    candidate.lower() not in unique_words and   # <--- исправлено
                    candidate not in word_mapping.values()):
                    new_w = candidate
                    break
            if new_w is None:
                new_w = fallback_replace(w)
                attempts_local = 0
                while ((new_w in word_mapping.values() or new_w == w or new_w.lower() in unique_words) 
                       and attempts_local < 20):
                    new_w = fallback_replace(w)
                    attempts_local += 1
            word_mapping[w] = new_w

        new_names = []
        for original in names:
            new_name = apply_word_mapping(original, word_mapping)
            new_names.append(new_name)

        if len(set(new_names)) == len(new_names):
            return {original: new_name for original, new_name in zip(names, new_names)}

    # Если не удалось, добавляем суффиксы (крайний случай)
    print(f"⚠️ Не удалось достичь уникальности за {max_attempts} попыток. Добавляем суффиксы.")
    final_mapping = {}
    used_names = set()
    for original, new_name in zip(names, new_names):
        base = new_name
        counter = 1
        while new_name in used_names:
            new_name = f"{base}_{counter}"
            counter += 1
        used_names.add(new_name)
        final_mapping[original] = new_name
    return final_mapping

def generate_mapping(entities_by_category, order=2):
    final_mapping = {}
    for category, names in entities_by_category.items():
        category_mapping = generate_unique_mapping_for_category(names, order)
        final_mapping.update(category_mapping)
        print(f"✅ Категория '{category}': {len(category_mapping)} замен.")
    return final_mapping

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

def main():
    parser = argparse.ArgumentParser(description='Генерирует словарь замен с уникальными значениями.')
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