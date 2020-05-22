# Shared utility functions
import re
import string


def doc_to_tokens(doc, non_alphabetic="keep", stopset=None):
    """

    :param doc:
    :param non_alphabetic:
      "keep" - leave in original tokens
      "split" - split into separate tokens
      "remove" - completely remove
    :param stopset:
    :return:
    """
    if stopset is None:
        stopset = []
    doc = doc.casefold()

    # Handle non alphabetic characters
    if non_alphabetic == "split":
        doc = re.sub(r"([^a-z ]+)", r" \1 ", doc)
    if non_alphabetic == "remove":
        doc = re.sub(r"([^a-z ]+)", "", doc)

    # Naive tokenisation (by whitespace)
    tokens = doc.split()
    # Remove leading/trailing punctuation, and tokens that consist only of punctuation
    tokens = [x.strip(string.punctuation) for x in tokens]
    if stopset is not None:
        tokens = [x for x in tokens if x and x not in stopset]

    return tokens
