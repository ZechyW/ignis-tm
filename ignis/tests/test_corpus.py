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
        self.corpus_1 = ignis.corpus.Corpus()
        self.corpus_1_doc_1 = self.corpus_1.add_doc(["document", "tokens"])
        self.corpus_1_doc_2 = self.corpus_1.add_doc(["other", "words"])

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