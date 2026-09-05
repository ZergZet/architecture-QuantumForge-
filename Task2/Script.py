import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup

# ====== НАСТРОЙКИ ======
INPUT_JSON = "lotr_data.json"
OUTPUT_DIR = "lotr_articles"
WIKI_API = "https://lotr.fandom.com/ru/api.php"
USER_AGENT = "LotrArticleScraper/1.0 (https://example.com; your-email@example.com)"
REQUEST_DELAY = 0.5                  # задержка между запросами (сек)
SEARCH_LIMIT = 3                     # сколько результатов поиска просмотреть
# ========================

def slugify(name):
    """Превращает название в безопасное имя файла."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name.strip('_')

def fetch_article_by_title(title):
    """
    Пытается получить статью по точному заголовку через action=parse.
    Возвращает (текст_статьи, None) или (None, сообщение_об_ошибке).
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "redirects": "1",
        "disablelimitreport": "1",
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(WIKI_API, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return None, f"API error: {data['error'].get('info', 'unknown')}"
        if "parse" not in data:
            return None, "Page not found or no parse data"
        html = data["parse"]["text"]["*"]
        # Извлекаем основной контент
        soup = BeautifulSoup(html, "html.parser")
        content_div = soup.find("div", class_="mw-parser-output")
        if not content_div:
            text = soup.get_text(separator="\n")
        else:
            # Удаляем служебные и шумные блоки
            for cls in ["editlink", "navbox", "mw-editsection", "toc", "infobox", "hatnote"]:
                for tag in content_div.find_all(class_=cls):
                    tag.decompose()
            # Дополнительно убираем таблицы (часто содержат лишнее)
            for table in content_div.find_all("table"):
                table.decompose()
            text = content_div.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return text, None
    except Exception as e:
        return None, f"Request error: {e}"

def search_article(query):
    """
    Ищет статью через list=search, возвращает точный заголовок первой подходящей страницы
    (или None, если ничего не найдено).
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": SEARCH_LIMIT,
        "format": "json",
        "redirects": "1",
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(WIKI_API, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "query" in data and "search" in data["query"]:
            results = data["query"]["search"]
            if results:
                # Берём первый результат (он же самый релевантный)
                return results[0]["title"]
        return None
    except Exception:
        return None

def get_article_content(name):
    """
    Главная функция: сначала пробует прямой запрос, если не получается – ищет через поиск.
    Возвращает (текст_статьи, None) или (None, сообщение_об_ошибке).
    """
    # Пробуем напрямую
    content, err = fetch_article_by_title(name)
    if content is not None:
        return content, None

    # Если не удалось – ищем через поиск
    print(f"   🔍 Прямой запрос не удался, ищем через поиск: '{name}'")
    found_title = search_article(name)
    if found_title is None:
        return None, f"Не найдено ни одной статьи по запросу '{name}'"
    print(f"   ✅ Найдено через поиск: '{found_title}'")

    # Повторно запрашиваем по найденному заголовку
    content, err2 = fetch_article_by_title(found_title)
    if content is not None:
        return content, None
    else:
        return None, f"Не удалось загрузить найденную статью '{found_title}': {err2}"

def main():
    # 1. Загружаем JSON
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Файл {INPUT_JSON} не найден.")
        return
    except json.JSONDecodeError:
        print(f"❌ Ошибка в JSON: {INPUT_JSON}")
        return

    # 2. Собираем все названия сущностей из всех категорий
    all_entities = []
    for category, items in data.items():
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    all_entities.append(item)
                elif isinstance(item, dict) and "событие" in item:
                    all_entities.append(item["событие"])
    unique_entities = list(dict.fromkeys(all_entities))
    print(f"📊 Найдено {len(unique_entities)} уникальных сущностей.")

    # 3. Создаём папку
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. Обрабатываем каждую сущность
    for idx, name in enumerate(unique_entities, 1):
        print(f"\n[{idx}/{len(unique_entities)}] Обработка: {name}")
        slug = slugify(name)
        filepath = os.path.join(OUTPUT_DIR, f"{slug}.txt")

        if os.path.exists(filepath):
            print(f"   ⏭️ Файл уже существует: {filepath}")
            continue

        content, error = get_article_content(name)

        if content is None:
            print(f"   ❌ Не удалось получить статью: {error}")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"[Не удалось загрузить статью]\nОшибка: {error}\n")
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   ✅ Сохранено: {filepath}")

        time.sleep(REQUEST_DELAY)

    print("\n🎉 Готово!")

if __name__ == "__main__":
    main()