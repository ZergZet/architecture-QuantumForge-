#!/usr/bin/env python3
"""
Индексация knowledge_base в Qdrant.

Задачи:
  1. Прочитать файлы из KB_DIR (.txt, .md, .pdf).
  2. Разбить текст на чанки (RecursiveCharacterTextSplitter).
  3. Посчитать эмбеддинги (Ollama HTTP API или sentence-transformers).
  4. Создать/очистить коллекцию в Qdrant.
  5. Залить чанки с payload {filename, title, text}.
  6. Опционально: прогнать чанки через safety_guard (--safety-in).

Офлайн-режим HF включён по умолчанию — модели предзагружены run_rag2.sh.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    _HAS_QDRANT = True
except ImportError as e:
    print(f"[ERROR] qdrant_client не установлен: {e}")
    sys.exit(1)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_SPLITTER = True
except ImportError:
    _HAS_SPLITTER = False

try:
    from safety_guard import SafetyGuard
    _HAS_SAFETY = True
    _SAFETY_IMPORT_ERR = ""
except Exception as e:
    SafetyGuard = None  # type: ignore
    _HAS_SAFETY = False
    _SAFETY_IMPORT_ERR = str(e)


def log(msg: str) -> None:
    print(msg, flush=True)


def read_file(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            log(f"[WARN] pypdf не установлен, пропускаю {p.name}")
            return ""
        try:
            reader = PdfReader(str(p))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            log(f"[WARN] Не удалось прочитать {p.name}: {e}")
            return ""
    return ""


def build_splitter(chunk_size: int, chunk_overlap: int):
    if not _HAS_SPLITTER:
        return None
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator=True,
    )


def chunk_text(text, splitter, chunk_size, chunk_overlap):
    text = (text or "").strip()
    if not text:
        return []
    if splitter is not None:
        return splitter.split_text(text)
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= len(text):
            break
        i += chunk_size - chunk_overlap
    return chunks


class OllamaEmbedder:
    def __init__(self, model, url, timeout=60):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    def encode(self, texts):
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/embed", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        log(f"[DEBUG] POST /api/embed, {len(texts)} текстов, "
            f"ответ за {time.time() - t0:.2f}с")
        return data["embeddings"]


class STEmbedder:
    def __init__(self, model):
        from sentence_transformers import SentenceTransformer
        log(f"[INFO] Загружаю SentenceTransformer({model!r})...")
        self.model = SentenceTransformer(model)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def build_embedder(provider, model, ollama_url):
    if provider == "ollama":
        log(f"[INFO] Эмбеддинги: Ollama HTTP API (model={model!r}, url={ollama_url!r})")
        return OllamaEmbedder(model, ollama_url)
    log(f"[INFO] Эмбеддинги: SentenceTransformer({model!r})")
    return STEmbedder(model)


def qdrant_delete_collection(client, collection):
    try:
        client.delete_collection(collection)
        log(f"[INFO] Коллекция '{collection}' удалена.")
    except Exception as e:
        log(f"[WARN] Не удалось удалить коллекцию '{collection}': {e}")


def qdrant_create_collection(client, collection, dim):
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    log(f"[OK] Коллекция '{collection}' создана (dim={dim}, cosine).")


def main():
    ap = argparse.ArgumentParser(description="Индексация knowledge_base в Qdrant")
    ap.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", default=os.environ.get("COLLECTION_NAME", "knowledge_base"))
    ap.add_argument("--kb-dir", default=os.environ.get("KB_DIR", "knowledge_base"))
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--embedding-provider", default=os.environ.get("EMBEDDING_PROVIDER", "ollama"),
                    choices=["ollama", "st"])
    ap.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", "embeddinggemma"))
    ap.add_argument("--embedding-dim", type=int, default=int(os.environ.get("EMBEDDING_DIM", "768")))
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--chunk-size", type=int, default=int(os.environ.get("CHUNK_SIZE", "500")))
    ap.add_argument("--chunk-overlap", type=int, default=int(os.environ.get("CHUNK_OVERLAP", "50")))
    ap.add_argument("--embed-batch", type=int, default=int(os.environ.get("EMBED_BATCH", "8")))
    ap.add_argument("--upsert-batch", type=int, default=int(os.environ.get("UPSERT_BATCH", "64")))
    ap.add_argument("--safety-in", action="store_true",
                    default=os.environ.get("SAFETY_IN", "0") == "1")
    ap.add_argument("--on-injection", default=os.environ.get("ON_INJECTION", "skip"),
                    choices=["skip", "redact", "abort"])
    args = ap.parse_args()

    t_total_start = time.time()

    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_dir():
        log(f"[ERROR] Папка {kb_dir} не найдена.")
        sys.exit(2)

    safety_guard = None
    if args.safety_in:
        if not _HAS_SAFETY:
            log(f"[SAFETY][ERROR] safety_guard.py недоступен: {_SAFETY_IMPORT_ERR}")
        else:
            try:
                safety_guard = SafetyGuard(use_input_models=True, use_output_models=False)
                log(f"[SAFETY] {safety_guard.describe()}")
            except Exception as e:
                log(f"[SAFETY][ERROR] Не удалось создать SafetyGuard: {e}")

    log(f"[INFO] Подключаюсь к Qdrant {args.qdrant_url}...")
    client = QdrantClient(url=args.qdrant_url, timeout=60)

    if args.reindex:
        log(f"[INFO] Режим --reindex: удаляю коллекцию '{args.collection}'...")
        qdrant_delete_collection(client, args.collection)
        for _ in range(30):
            try:
                client.get_collection(args.collection)
                time.sleep(1)
            except Exception:
                break

    exists = True
    try:
        client.get_collection(args.collection)
        count = client.count(args.collection, exact=True).count
        if count > 0:
            log(f"[OK] Коллекция '{args.collection}' уже содержит {count} точек — "
                f"индексация не нужна.")
            log("     Для переиндексации запустите с флагом --reindex.")
            sys.exit(0)
        log(f"[INFO] Коллекция '{args.collection}' пуста — индексирую.")
    except Exception:
        exists = False

    if not exists:
        qdrant_create_collection(client, args.collection, args.embedding_dim)

    files = [p for p in kb_dir.rglob("*") if p.is_file()]
    if not files:
        log(f"[ERROR] В папке {kb_dir} нет файлов.")
        sys.exit(2)

    log(f"[INFO] Найдено файлов: {len(files)}")
    splitter = build_splitter(args.chunk_size, args.chunk_overlap)
    log(f"[INFO] Чанкинг: "
        f"{'RecursiveCharacterTextSplitter' if splitter else 'fallback (символы)'} "
        f"(size={args.chunk_size}, overlap={args.chunk_overlap})")

    embedder = build_embedder(args.embedding_provider, args.embedding_model, args.ollama_url)

    all_chunks = []
    all_meta = []
    skipped_by_safety = 0

    for f in files:
        text = read_file(f)
        if not text.strip():
            log(f"[WARN] Пустой/неподдерживаемый файл: {f.name}")
            continue
        for chunk in chunk_text(text, splitter, args.chunk_size, args.chunk_overlap):
            if safety_guard is not None:
                v = safety_guard.check_document(chunk)
                if not v.safe:
                    log(f"[SAFETY-IN] Чанк из '{f.name}' отклонён: {v.details}")
                    skipped_by_safety += 1
                    if args.on_injection == "abort":
                        log("[SAFETY-IN] Режим abort — прерываю индексацию.")
                        sys.exit(6)
                    continue
            all_chunks.append(chunk)
            all_meta.append({"filename": f.name, "title": f.stem, "text": chunk})

    if skipped_by_safety:
        log(f"[SAFETY-IN] Пропущено небезопасных чанков: {skipped_by_safety}")

    if not all_chunks:
        log("[ERROR] Не удалось извлечь ни одного чанка.")
        sys.exit(3)

    log(f"[INFO] Сгенерировано {len(all_chunks)} чанков. "
        f"Считаю эмбеддинги батчами по {args.embed_batch}...")

    points = []
    t_embed_start = time.time()
    t_start = time.time()
    for i in range(0, len(all_chunks), args.embed_batch):
        batch_texts = all_chunks[i:i + args.embed_batch]
        try:
            vectors = embedder.encode(batch_texts)
        except Exception as e:
            log(f"[ERROR] Ошибка на батче {i}: {e}")
            sys.exit(5)
        for vec, meta in zip(vectors, all_meta[i:i + args.embed_batch]):
            points.append(PointStruct(id=len(points), vector=list(vec), payload=meta))
        done = min(i + args.embed_batch, len(all_chunks))
        elapsed = time.time() - t_start
        speed = done / elapsed if elapsed > 0 else 0
        log(f"[INFO] Эмбеддинги: {done}/{len(all_chunks)} ({speed:.1f} чанк/с)")

    t_embed_end = time.time()
    embed_elapsed = t_embed_end - t_embed_start
    log(f"[TIME] Подсчёт эмбеддингов занял {embed_elapsed:.1f} сек "
        f"({len(points) / embed_elapsed:.1f} чанк/с в среднем).")

    if points and len(points[0].vector) != args.embedding_dim:
        log(f"[ERROR] Размерность вектора {len(points[0].vector)} "
            f"!= EMBEDDING_DIM={args.embedding_dim}.")
        log(f"        Перезапустите с --embedding-dim {len(points[0].vector)}")
        sys.exit(4)

    log(f"[INFO] Заливаю {len(points)} точек в Qdrant...")
    for i in range(0, len(points), args.upsert_batch):
        client.upsert(collection_name=args.collection,
                      points=points[i:i + args.upsert_batch])
        log(f"[INFO] Загружено {min(i + args.upsert_batch, len(points))}/{len(points)}...")

    elapsed_total = time.time() - t_total_start
    minutes, seconds = divmod(elapsed_total, 60)

    log(f"[OK] Индексация завершена: {len(points)} чанков в '{args.collection}'.")
    log(f"[TIME] Общее время индексации: {minutes:.0f}м {seconds:.1f}с "
        f"(всего {elapsed_total:.1f} сек)")

if __name__ == "__main__":
    main()
