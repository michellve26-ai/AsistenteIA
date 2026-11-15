"""Pruebas del autoindice aisladas del LLM, Tavily, Chroma real y Render."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import auto_index


class AutoIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf_dir = self.root / "pdfs"
        self.pdf_dir.mkdir()
        self.index = self.root / "indice"
        self.pdf = self.pdf_dir / "manual.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 documento de prueba")
        self.config_patch = patch.multiple(
            auto_index.config,
            PDF_DIR=str(self.pdf_dir),
            PERSIST_DIR=str(self.index),
            EMBEDDING_MODEL="prueba-local",
            COLLECTION_NAME="manuales_soporte",
            CHUNK_SIZE=500,
            CHUNK_OVERLAP=50,
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.calls = []

    def _index_factory(self, **kwargs):
        target = Path(kwargs["persist_directory"])
        self.calls.append(target)
        target.mkdir(parents=True)
        (target / "chroma.sqlite3").write_bytes(b"sqlite-dummy")
        return SimpleNamespace(_collection=SimpleNamespace(count=lambda: 2))

    def _run(self, *, fail=False, force=False):
        def factory(**kwargs):
            if fail:
                raise RuntimeError("embeddings fallaron")
            return self._index_factory(**kwargs)

        with (
            patch.object(auto_index, "load_pdfs", return_value=["pagina"]),
            patch.object(auto_index, "split_documents", return_value=["trozo1", "trozo2"]),
            patch.object(auto_index.Chroma, "from_documents", side_effect=factory),
            patch.object(auto_index, "_index_count", return_value=2),
        ):
            return auto_index.ensure_index(lambda: object(), force=force)

    def test_crea_indice_y_no_repite_sin_cambios(self):
        self.assertTrue(self._run())
        self.assertTrue((self.index / "chroma.sqlite3").exists())
        manifest = json.loads((self.index / ".astrea-index-manifest.json").read_text())
        self.assertEqual(manifest["fragments"], 2)
        self.assertEqual(manifest["collection"], "manuales_soporte")
        self.assertFalse(self._run())
        self.assertEqual(len(self.calls), 1)

    def test_pdf_modificado_genera_otro_indice(self):
        self.assertTrue(self._run())
        self.pdf.write_bytes(b"%PDF-1.4 documento nuevo distinto")
        self.assertTrue(self._run())
        self.assertEqual(len(self.calls), 2)

    def test_pdf_añadido_o_eliminado_dispara_actualizacion(self):
        self.assertTrue(self._run())
        extra = self.pdf_dir / "extra.pdf"
        extra.write_bytes(b"%PDF-1.4 otro")
        self.assertTrue(self._run())
        extra.unlink()
        self.assertTrue(self._run())
        self.assertEqual(len(self.calls), 3)

    def test_error_preserva_base_anterior_y_manifiesto(self):
        self.assertTrue(self._run())
        original = (self.index / ".astrea-index-manifest.json").read_bytes()
        self.pdf.write_bytes(b"PDF actualizado")
        with self.assertRaisesRegex(RuntimeError, "embeddings fallaron"):
            self._run(fail=True)
        self.assertEqual((self.index / ".astrea-index-manifest.json").read_bytes(), original)
        self.assertTrue((self.index / "chroma.sqlite3").exists())
        self.assertEqual(list(self.root.glob("indice.astrea-next-*")), [])
        self.assertEqual(list(self.root.glob("indice.astrea-old-*")), [])

    def test_carpeta_pdf_vacia_no_toca_indice(self):
        self.assertTrue(self._run())
        self.pdf.unlink()
        with self.assertRaises(FileNotFoundError):
            self._run()
        self.assertTrue((self.index / "chroma.sqlite3").exists())

    def test_reconstruccion_manual_solo_bajo_force(self):
        self.assertTrue(self._run())
        self.assertTrue(self._run(force=True))
        self.assertEqual(len(self.calls), 2)

    def test_restaura_backup_si_proceso_interrumpido(self):
        old = self.root / "indice.astrea-old-interrumpido"
        old.mkdir()
        (old / "chroma.sqlite3").write_bytes(b"indice-previo")
        with auto_index._exclusive_lock(self.index):
            auto_index._restore_backup_if_interrupted(self.index)
        self.assertTrue((self.index / "chroma.sqlite3").exists())
        self.assertFalse(old.exists())


if __name__ == "__main__":
    unittest.main()
