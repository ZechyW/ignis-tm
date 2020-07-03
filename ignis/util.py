"""
General utility classes that are not technically a part of Ignis functionality
"""

import importlib
import math
import uuid


class LazyLoader:
    """
    A general class for lazy loading expensive modules.

    Parameters
    ----------
    module_name: str
        Module name to lazy load

    Examples
    --------
    tp = LazyLoader("tomotopy")
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self._module = None

    def __getattr__(self, item):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, item)


class ImprovedPhraser:
    """
    A slightly different implementation of automated n-gram detection derived from
    Gensim's Phraser

    - Applies to highest-scoring and longest n-grams in a document first
    - Chained n-grams that have the same score are combined into a single n-gram
      (e.g., if "quick brown" and "brown fox" have the same score, a new n-gram
      "quick brown fox" is created with that score)

    Assumes that its input is an iterable of strings (it uses substring matching for
    fast n-gram application)

    Parameters
    ----------
    sentences: iterable of iterable of str
        Required argument that will be passed to `gensim.models.Phrases` to generate
        the base n-gram model
    min_count, threshold, max_vocab_size, scoring, common_terms: optional
        Optional arguments that will be passed directly to `gensim.models.Phrases` to
        generate the base n-gram model
    delimiter: str, optional
        Delimiter to join detected phrases with. Will not actually be passed to the
        underlying Gensim model (which expects bytes and not a string anyway).
    """

    def __init__(
        self, sentences, delimiter=" ", **kwargs,
    ):
        self.delimiter = delimiter

        gensim = LazyLoader("gensim")

        gensim_kwarg_names = [
            "min_count",
            "threshold",
            "max_vocab_size",
            "scoring",
            "common_terms",
        ]
        gensim_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in gensim_kwarg_names and value is not None
        }

        model = gensim.models.Phrases(sentences=sentences, **gensim_kwargs)
        model = gensim.models.phrases.Phraser(model)
        model = model.phrasegrams.items()

        # model is a list of tuples(<n-gram>, <score>), where <n-gram> is a
        # tuple (<token 1>, <token 2>, ...).
        # N.B.: Gensim treats all the tokens as bytes, so we will need to .decode()
        # to strings when using them below.

        # The final list of sorted phrases
        self.phrases = []

        # Sort by scores then length, using set() to deduplicate
        scores = set(score for _, score in model)
        scores = sorted(list(scores), reverse=True)
        for score_tier in scores:
            score_phrases = set(
                phrase for phrase, score in model if score == score_tier
            )
            # Decode the byte tokens in the phrase tuple
            score_phrases = set(
                tuple(token.decode() for token in phrase) for phrase in score_phrases
            )

            # Recursively expanded phrase list (including "chained" phrases),
            # again using set() for deduplication
            merged = set(score_phrases)
            done = False
            while not done:
                # Start by assuming we are done; we will reset the flag if anything
                # shows up on this round
                done = True

                # For each original n-gram from the Phraser...
                for phrase in score_phrases:
                    # Look for chains with original phrases or already-merged extras
                    others = (score_phrases | merged) - {phrase}

                    for other in others:
                        # Sanity checks (prevents infinite merging loops)
                        str_phrase = " ".join(phrase)
                        str_other = " ".join(other)
                        if str_phrase in str_other or str_other in str_phrase:
                            continue
                        if (
                            len(phrase) == 2
                            and len(other) == 2
                            and phrase[0] == other[1]
                            and phrase[1] == other[0]
                        ):
                            continue

                        new_phrase = None
                        if phrase[0] == other[-1]:
                            new_phrase = other + phrase[1:]
                        if phrase[-1] == other[0]:
                            new_phrase = phrase + other[1:]

                        if new_phrase and new_phrase not in merged:
                            merged.add(new_phrase)
                            done = False

            # Sort by tokens alphabetically, then by length
            sorted_merged = sorted(list(merged), key=lambda x: " ".join(x))
            sorted_merged = sorted(sorted_merged, key=lambda x: len(x), reverse=True)
            self.phrases += [(phrase, score_tier) for phrase in sorted_merged]

        # And done -- Ready to phrase.

    def find_ngrams(self, docs, threshold=-math.inf):
        """
        Perform n-gram replacement on the given documents using the phrase model
        trained for this instance.

        Parameters
        ----------
        docs: iterable of iterable of str
            Each doc in docs should be an iterable of strings -- Substring matching
            will be used in the replacement process
        threshold: float, optional
            Optionally set a new phrasing threshold; if not set, will apply all the
            available phrases (which are determined by the value of `threshold`
            passed to the base gensim model on init)

        Returns
        -------
        iterable of iterable of str
        """
        # A "safe" delimiter that will be used for the intermediate string joins;
        # should be guaranteed to be different from `self.delimiter` so that we can
        # perform recursive phrasing without inadvertently re-splitting previously
        # joined tokens
        search_delimiter = f"<*>"

        new_docs = []
        for doc in docs:
            # In case we really need a new delimiter: Use part of a UUID (not the
            # whole, to conserve memory)
            while search_delimiter in doc or search_delimiter == self.delimiter:
                search_delimiter = f"<{str(uuid.uuid4())[:3]}>"

            str_doc = search_delimiter.join(doc)

            for phrase, score in self.phrases:
                if score < threshold:
                    continue

                str_phrase = search_delimiter.join(phrase)
                str_replace = self.delimiter.join(phrase)
                str_doc = str_doc.replace(str_phrase, str_replace)

            new_docs.append(str_doc.split(search_delimiter))

        return new_docs
