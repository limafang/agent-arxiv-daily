import datetime
import unittest
from types import SimpleNamespace
from unittest import mock

import daily_arxiv


class SummaryTests(unittest.TestCase):
    def test_missing_api_key_keeps_original_abstract(self):
        with mock.patch.object(daily_arxiv.dashscope, "api_key", None):
            with mock.patch.object(daily_arxiv.dashscope.Generation, "call") as call:
                self.assertEqual(
                    daily_arxiv.llm_generate_summary("original"),
                    "original",
                )
                call.assert_not_called()

    def test_dashscope_exception_keeps_original_abstract(self):
        with mock.patch.object(daily_arxiv.dashscope, "api_key", "test-key"):
            with mock.patch.object(
                daily_arxiv.dashscope.Generation,
                "call",
                side_effect=RuntimeError("service unavailable"),
            ):
                self.assertEqual(
                    daily_arxiv.llm_generate_summary("original"),
                    "original",
                )


class PaperCollectionTests(unittest.TestCase):
    def test_code_lookup_request_failure_returns_none(self):
        with mock.patch.object(
            daily_arxiv.requests,
            "get",
            side_effect=daily_arxiv.requests.ConnectionError("offline"),
        ):
            self.assertIsNone(daily_arxiv.get_official_code_url("2608.12345v1"))

    def test_paper_is_kept_when_code_lookup_fails(self):
        result = SimpleNamespace(
            get_short_id=lambda: "2608.12345v1",
            title="A useful paper",
            entry_id="https://arxiv.org/abs/2608.12345v1",
            summary="An abstract.",
            authors=["First Author", "Second Author"],
            primary_category="cs.AI",
            published=datetime.datetime(2026, 8, 24),
            updated=datetime.datetime(2026, 8, 25),
            comment=None,
        )

        client = mock.Mock()
        client.results.return_value = iter([result])
        with mock.patch.object(daily_arxiv.arxiv, "Client", return_value=client):
            with mock.patch.object(
                daily_arxiv, "llm_generate_summary", return_value="Translated abstract."
            ):
                with mock.patch.object(
                    daily_arxiv, "get_official_code_url", return_value=None
                ):
                    data, _ = daily_arxiv.get_daily_papers(
                        "agent", query="LLM agent", max_results=1
                    )

        row = data["agent"]["2608.12345"]
        self.assertIn("|null|", row)
        self.assertIn("Translated abstract.", row)


if __name__ == "__main__":
    unittest.main()
