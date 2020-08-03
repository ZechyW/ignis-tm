import unittest
import ignis.corpus


class TestCorpus(unittest.TestCase):
    """
    Test Corpus and CorpusSlice functions
    """

    def setUp(self) -> None:
        """
        Prepare test Corpus objects
        """
        # Fully-specified Documents
        self.corpus_1 = ignis.corpus.Corpus()
        self.corpus_1_doc_1 = self.corpus_1.add_doc(
            tokens=["these", "are", "some", "document", "tokens"],
            metadata=None,
            display_str="<html><body>THESE are some document- +tokens</body></html>",
            plain_text="THESE are some document- +tokens",
        )
        self.corpus_1_doc_2 = self.corpus_1.add_doc(
            ["these", "are", "some", "other", "words"],
            metadata=None,
            display_str="<html><body>these ARE some (other words)</body></html>",
            plain_text="these ARE some (other words)",
        )

        # `display_str` and `plain_text` will be automatically generated
        self.corpus_2 = ignis.corpus.Corpus()
        self.corpus_2_doc_1 = self.corpus_2.add_doc(["document", "tokens"])

    def test_slice_by_token(self):
        """
        Slicing by token should include the relevant documents
        """
        slice_1 = self.corpus_1.slice_full()
        slice_other = slice_1.slice_by_tokens(["other"])
        self.assertEqual(len(slice_other), 1)

        # Non-root search
        non_root = slice_other.slice_by_tokens(["document"])
        self.assertEqual(len(non_root), 0)

        # Root search
        root = slice_other.slice_by_tokens(["document"], include_root=True)
        self.assertEqual(len(root), 1)

    def test_slice_concat(self):
        """
        Should not be able to concat CorpusSlices from different root Corpus objects
        """
        slice_1 = self.corpus_1.slice_full()
        slice_2 = self.corpus_2.slice_full()

        self.assertRaises(RuntimeError, slice_1.concat, slice_2)

    def test_slice_concat_2(self):
        """
        Concatenated CorpusSlices should contain Documents from original CorpusSlices
        """
        slice_all = self.corpus_1.slice_full()
        slice_doc_1 = slice_all.slice_by_ids([self.corpus_1_doc_1])
        slice_doc_2 = slice_all.slice_by_ids([self.corpus_1_doc_2])

        self.assertEqual(len(slice_doc_1), 1)
        self.assertEqual(len(slice_doc_2), 1)

        slice_concat = slice_doc_1.concat(slice_doc_2)
        self.assertEqual(len(slice_concat), 2)

    def test_slice_without_tokens(self):
        """
        Slicing by removing tokens
        """
        # Tokens
        all_slice = self.corpus_1.slice_full()
        doc_1_slice = all_slice.slice_by_ids([self.corpus_1_doc_1])
        doc_2_slice = all_slice.slice_by_ids([self.corpus_1_doc_2])

        # Tokenised representation
        test_slice_1 = all_slice.slice_without_tokens(["document"])
        self.assertEqual(len(test_slice_1), 1)
        self.assertEqual(test_slice_1, doc_2_slice)

        test_slice_2 = all_slice.slice_without_tokens(["+tokens"])
        self.assertEqual(len(test_slice_2), 2)
        self.assertEqual(test_slice_2, all_slice)

        # Human-readable representation
        test_slice_1 = all_slice.slice_without_tokens(["(other"], plain_text=True)
        self.assertEqual(len(test_slice_1), 1)
        self.assertEqual(test_slice_1, doc_1_slice)

        test_slice_2 = all_slice.slice_without_tokens(["tokens"], plain_text=True)
        self.assertEqual(len(test_slice_2), 2)
        self.assertEqual(test_slice_2, all_slice)

        # Phrase matching
        test_slice_1 = all_slice.slice_without_tokens(
            ["(other words)"], plain_text=True
        )
        self.assertEqual(len(test_slice_1), 1)
        self.assertEqual(test_slice_1, doc_1_slice)

        test_slice_2 = all_slice.slice_without_tokens(
            ["document- +tokens"], plain_text=True
        )
        self.assertEqual(len(test_slice_2), 1)
        self.assertEqual(test_slice_2, doc_2_slice)
