import unittest

from world.retrieval.corpus import load_corpus
from world.retrieval.keyword_search import search_documents
from world.retrieval.cross_check import cross_check


class TestOkuribiRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus()

    def test_corpus_has_eight_documents(self):
        self.assertEqual(len(self.corpus), 8)

    def test_hidden_forged_flag_is_not_exported(self):
        self.assertTrue(all("forged" not in doc for doc in self.corpus))

    def test_transfer_query_finds_transfer_record(self):
        results = search_documents("status transferred effective date", corpus=self.corpus)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "DOC_02")

    def test_metadata_filter(self):
        results = search_documents(
            "lantern festival", filters={"type": "festival_inventory"}, corpus=self.corpus
        )
        self.assertTrue(all(doc["type"] == "festival_inventory" for doc in results))
        self.assertTrue(any(doc["id"] == "DOC_06" for doc in results))

    def test_cross_check_finds_transfer_conflict(self):
        result = cross_check("DOC_02", "DOC_05", corpus=self.corpus)
        self.assertTrue(result["ok"])
        self.assertEqual(result["check_id"], "CHECK_TRANSFER_VS_REGISTER")

    def test_unknown_document_is_handled(self):
        result = cross_check("DOC_02", "DOC_NOPE", corpus=self.corpus)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "unknown_document_id")


if __name__ == "__main__":
    unittest.main()
