"""Regresion de las esperas 429 sin consumir Groq."""
import unittest
from types import SimpleNamespace

from evaluacion.ragas_rate_limit import quota_delay_seconds, retry_wait_seconds


class RagasRateLimitTests(unittest.TestCase):
    def test_espera_en_milisegundos_real(self):
        err = RuntimeError("Rate limit reached. Please try again in 899.999999ms.")
        self.assertAlmostEqual(quota_delay_seconds(err), 0.899999999)

    def test_espera_en_segundos(self):
        err = RuntimeError("Please try again in 10.1175s.")
        self.assertAlmostEqual(quota_delay_seconds(err), 10.1175)

    def test_esperas_compuestas(self):
        self.assertEqual(quota_delay_seconds(RuntimeError("try again in 1h2m3.5s")), 3723.5)
        self.assertEqual(quota_delay_seconds(RuntimeError("try again in 2m3s")), 123.0)
        self.assertEqual(quota_delay_seconds(RuntimeError("try again in 3m")), 180.0)

    def test_cabecera_retry_after(self):
        err = RuntimeError("Groq 429")
        err.response = SimpleNamespace(headers={"retry-after": "2.5"})
        self.assertAlmostEqual(quota_delay_seconds(err), 2.5)

    def test_sin_indicacion(self):
        self.assertIsNone(quota_delay_seconds(RuntimeError("Groq rate limited")))

    def test_backoff_con_margen_y_limite(self):
        err = RuntimeError("Please try again in 800ms")
        self.assertGreaterEqual(retry_wait_seconds(err, 0, 240), 12.0)
        self.assertGreaterEqual(retry_wait_seconds(err, 5, 240), 72.0)
        self.assertIsNone(retry_wait_seconds(
            RuntimeError("Please try again in 500s"), 0, 240
        ))

    def test_fallback_conservador(self):
        self.assertGreaterEqual(
            retry_wait_seconds(RuntimeError("429 quota"), 0, 240), 60
        )


if __name__ == "__main__":
    unittest.main()
