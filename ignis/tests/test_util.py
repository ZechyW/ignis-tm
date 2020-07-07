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
            (["the", "quick", "brown"], 1.0),
            (["quick", "brown"], 1.0),
            (["the", "quick"], 1.0),
            (["likes", "swimming"], 0.8105361810656604),
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

    def test_improved_phraser_priority(self):
        """
        Phraser priority tests
        """
        docs = [
            ["1", "likes", "swimming"],
            ["2", "likes", "swimming"],
            ["3", "likes", "swimming"],
            ["4", "likes", "swimming"],
            ["1", "swimming", "pool"],
            ["2", "swimming", "pool"],
            ["3", "swimming", "pool"],
            ["4", "swimming", "pool"],
            ["5", "swimming", "pool"],
            ["6", "swimming", "pool"],
            ["7", "swimming", "pool"],
            ["8", "swimming", "pool"],
            ["1", "swimming", "pool"],
            ["2", "swimming", "pool"],
            ["3", "swimming", "pool"],
            ["4", "swimming", "pool"],
            ["5", "swimming", "pool"],
            ["6", "swimming", "pool"],
            ["7", "swimming", "pool"],
            ["8", "swimming", "pool"],
            ["pool", "tubes", "1"],
            ["pool", "tubes", "2"],
            ["pool", "tubes", "3"],
            ["pool", "tubes", "4"],
        ]

        phraser = ignis.util.ImprovedPhraser(
            docs, min_count=1, threshold=0.4, scoring="npmi"
        )

        # Can't naively phrase sequentially from either the front or back of the
        # document, because the middle bigram ("swimming pool") scores higher than
        # the other two even when they are all chained together.
        expected_phrases = [
            (["swimming", "pool"], 0.7032818234055256),
            (["likes", "swimming"], 0.4431726963712192),
            (["pool", "tubes"], 0.4431726963712192),
        ]

        self.assertEqual(phraser.phrases, expected_phrases)

        # "swimming pool" should be detected first, invalidating the other phrases.
        # It should also be found at the beginning and end of documents
        test_docs = [
            ["John", "likes", "swimming", "pool", "tubes"],
            ["swimming", "pool", "tubes"],
            ["this", "swimming", "pool"],
        ]
        phrased_docs = phraser.find_ngrams(test_docs)
        expected_phrase_docs = [
            ["John", "likes", "swimming pool", "tubes"],
            ["swimming pool", "tubes"],
            ["this", "swimming pool"],
        ]

        self.assertEqual(phrased_docs, expected_phrase_docs)
