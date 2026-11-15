"""Sincronizacion segura de los PDF del indice PRINCIPAL de Astrea/Streamlit.

No modifica el backend FastAPI, las rutas, el agente ni el indice de pruebas.
Construye en otra carpeta; solo publica un indice si esta completo y verificado.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma

from . import config
from .ingest import load_pdfs, split_documents

_MANIFEST = ".astrea-index-manifest.json"
_SCHEMA = 1
_THREAD_LOCK = threading.RLock()


def _pdf_files() -> list[Path]:
    folder = Path(config.PDF_DIR)
    if not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.casefold(),
    )


def pdf_snapshot() -> tuple[tuple[str, int, int], ...]:
    """Clave ligera para detectar archivos nuevos, borrados o modificados en Streamlit."""
    return tuple(
        (p.name, p.stat().st_size, p.stat().st_mtime_ns)
        for p in _pdf_files()
    )


def manifest_stamp() -> tuple[int, int]:
    """Permite invalidar cache si el estado de Chroma cambia entre reruns."""
    path = Path(config.PERSIST_DIR) / _MANIFEST
    try:
        st = path.stat()
        return st.st_size, st.st_mtime_ns
    except OSError:
        return 0, 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _desired_manifest(files: list[Path]) -> dict:
    return {
        "schema": _SCHEMA,
        "pdf_hashes": {p.name: _sha256(p) for p in files},
        "embedding_model": config.EMBEDDING_MODEL,
        "collection": config.COLLECTION_NAME,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }


def _read_manifest(index_dir: Path) -> dict | None:
    try:
        data = json.loads((index_dir / _MANIFEST).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _index_count(index_dir: Path) -> int:
    """Comprueba la coleccion sin crear silenciosamente una nueva."""
    if not (index_dir / "chroma.sqlite3").is_file():
        return 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(index_dir.resolve()))
        return int(client.get_collection(name=config.COLLECTION_NAME).count())
    except Exception:
        return 0


def existing_index_count() -> int:
    """Solo diagnostico/recuperacion si la reindexacion no puede terminar."""
    return _index_count(Path(config.PERSIST_DIR))


@contextmanager
def _exclusive_lock(index_dir: Path):
    """Evita duplicar reconstrucciones entre sesiones y workers en Linux/Render."""
    lock_id = hashlib.sha256(str(index_dir.resolve()).encode("utf-8")).hexdigest()[:24]
    lock_path = Path(tempfile.gettempdir()) / f"astrea-autoindex-{lock_id}.lock"
    with _THREAD_LOCK, lock_path.open("a+b") as handle:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _restore_backup_if_interrupted(index_dir: Path) -> None:
    """Restaura la ultima copia si el proceso termino entre los dos renames."""
    if index_dir.exists():
        return
    candidates = sorted(
        index_dir.parent.glob(index_dir.name + ".astrea-old-*"),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "chroma.sqlite3").exists():
            candidate.rename(index_dir)
            print("[AUTO-INDEX] Indice anterior recuperado tras interrupcion.", flush=True)
            return


def _write_manifest(index_dir: Path, manifest: dict) -> None:
    # El manifiesto pertenece al indice, NO al data/index_state.json versionado.
    temp = index_dir / (_MANIFEST + ".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(index_dir / _MANIFEST)


def ensure_index(embeddings_provider: Callable, *, force: bool = False) -> bool:
    """Actualiza solo si falta un indice valido o cambiaron PDF/configuracion.

    Devuelve True si publico un indice nuevo, False si el existente esta vigente.
    Ante un fallo, conserva o recupera el indice anterior y propaga el error.
    """
    active = Path(config.PERSIST_DIR)
    index_path = active.resolve()
    pdf_path = Path(config.PDF_DIR).resolve()
    if (
        index_path == Path.cwd().resolve()
        or index_path == pdf_path
        or index_path in pdf_path.parents
        or pdf_path in index_path.parents
    ):
        raise ValueError("PERSIST_DIR debe ser una carpeta independiente de PDF y proyecto")

    with _exclusive_lock(active):
        _restore_backup_if_interrupted(active)
        files = _pdf_files()
        if not files:
            raise FileNotFoundError("No hay PDF para indexar; no se modifico el indice anterior")
        wanted = _desired_manifest(files)
        saved = _read_manifest(active)

        if not force and saved and all(saved.get(k) == v for k, v in wanted.items()):
            count = _index_count(active)
            if count > 0 and count == saved.get("fragments"):
                return False

        active.parent.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        staging = active.with_name(active.name + ".astrea-next-" + suffix)
        backup = active.with_name(active.name + ".astrea-old-" + suffix)

        try:
            # El indice anterior sigue intacto mientras se procesa el corpus.
            pages = load_pdfs(str(Path(config.PDF_DIR)))
            chunks = split_documents(pages)
            if not chunks:
                raise RuntimeError("La lectura no genero fragmentos; indice anterior conservado")
            # Nunca publicar un indice construido mientras los PDF estaban cambiando.
            if _desired_manifest(_pdf_files()) != wanted:
                raise RuntimeError("Los PDF cambiaron durante la indexacion; reintentar")

            embeddings = embeddings_provider()
            store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(staging),
                collection_name=config.COLLECTION_NAME,
                collection_metadata={"hnsw:space": "cosine"},
            )
            count = int(store._collection.count())
            del store
            if count != len(chunks) or not (staging / "chroma.sqlite3").is_file():
                raise RuntimeError(
                    "El nuevo indice esta incompleto; se conserva el indice anterior"
                )
            _write_manifest(staging, {**wanted, "fragments": count})

            moved_old = False
            try:
                if active.exists():
                    active.rename(backup)
                    moved_old = True
                staging.rename(active)
            except BaseException:
                if moved_old and backup.exists() and not active.exists():
                    backup.rename(active)
                raise

            if moved_old:
                try:
                    shutil.rmtree(backup)
                except OSError as exc:
                    # No invalidar una actualizacion exitosa por fallo de limpieza.
                    print(f"[AUTO-INDEX] Copia anterior pendiente de limpieza: {type(exc).__name__}", flush=True)
            print(f"[AUTO-INDEX] Indice principal publicado: {count} fragmentos.", flush=True)
            return True
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
