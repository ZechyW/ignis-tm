import unittest

import ignis.util


class TestUtil(unittest.TestCase):
    """
    Test utility functions
    """

    def test_improved_phraser(self):
        """
        Phraser smoke tests
        """
        docs = [
            ["look", "the", "quick", "brown", "fox", "likes", "swimming"],
            ["behold", "the", "quick", "brown", "penguin", "likes", "flying"],
            ["John", "likes", "swimming"],
        ]

        phraser = ignis.util.ImprovedPhraser(
            docs, min_count=1, threshold=0.8, scoring="npmi"
        )

        # "the quick brown" should be detected as a natural trigram
        # "likes swimming" should be detected as a natural bigram
        expected_phrases = [
            (("the", "quick", "brown"), 1.0),
            (("quick", "brown"), 1.0),
            (("the", "quick"), 1.0),
            (("likes", "swimming"), 0.8105361810656604),
        ]

        self.assertEqual(phraser.phrases, expected_phrases)

        # "the quick brown" and "likes swimming" should be applied
        phrased_docs = phraser.find_ngrams(docs)
        expected_phrased_docs = [
            ["look", "the quick brown", "fox", "likes swimming"],
            ["behold", "the quick brown", "penguin", "likes", "flying"],
            ["John", "likes swimming"],
        ]

        self.assertEqual(phrased_docs, expected_phrased_docs)

        # With a threshold, only "the quick brown" should be applied
        phrased_docs_2 = phraser.find_ngrams(docs, threshold=0.9)
        expected_phrased_docs_2 = [
            ["look", "the quick brown", "fox", "likes", "swimming"],
            ["behold", "the quick brown", "penguin", "likes", "flying"],
            ["John", "likes", "swimming"],
        ]

        self.assertEqual(phrased_docs_2, expected_phrased_docs_2)
