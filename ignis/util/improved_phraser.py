"""
An improved automated n-gram detection mechanism built on top of the `gensim` Phraser.
"""

import math
import time

from tqdm.auto import tqdm

from ignis.util.lazy_loader import LazyLoader


class ImprovedPhraser:
    """
    A slightly different implementation of automated n-gram detection derived from
    the `gensim` Phraser:

    - Applies to highest-scoring and longest n-grams in a document first.
    - Chained n-grams that have the same score are combined into a single n-gram.
      (E.g., if "quick brown" and "brown fox" have the same score, a new n-gram
      "quick brown fox" is created with that score.)

    Parameters
    ----------
    sentences: iterable of iterable of str
        Passed to `gensim.models.Phrases` to generate the base n-gram model.
    min_count, threshold, max_vocab_size, scoring, common_terms: optional
        Optional arguments that will be passed directly to `gensim.models.Phrases` to
        generate the base n-gram model.
    delimiter: str, optional
        Delimiter to join detected phrases with.

        This will not actually be passed to the underlying `gensim` model, and will
        only be used at the output stage.
    drop_non_alpha: bool, optional
        Whether or not to include phrases that consist entirely of non-alphabetic
        strings. Will drop them by default.
    verbose: bool, optional
        Whether or not to print verbose progress messages.
    """

    def __init__(
        self,
        sentences,
        delimiter=" ",
        drop_non_alpha=True,
        verbose=False,
        **kwargs,
    ):
        start_time = time.perf_counter()

        self.delimiter = delimiter

        gensim = LazyLoader("gensim")

        # Keyword arguments taken by `gensim`
        gensim_kwarg_names = [
            "min_count",
            "threshold",
            "max_vocab_size",
            "scoring",
            "connector_words",
        ]
        gensim_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in gensim_kwarg_names and value is not None
        }

        # Pass a bogus delimiter to gensim -- The default of "_" will throw off the
        # phrase scores if any terms already have underscores in them
        gensim_kwargs["delimiter"] = "<*ignis*>"

        model = gensim.models.Phrases(sentences=sentences, **gensim_kwargs)
        model = model.freeze()
        model = model.phrasegrams.items()

        elapsed = time.perf_counter() - start_time
        start_time = time.perf_counter()
        if verbose:
            print(f"Gensim Phraser initialised. {elapsed:.2f}s", flush=True)

        # model is a list of tuples(<n-gram>, <score>), where <n-gram> is a
        # tuple (<token 1>, <token 2>, ...).
        # N.B.: Gensim treats all the tokens as bytes, so we will need to .decode()
        # to strings when using them below.
        # ---------------------------------------------------
        # P.S.: The above is no longer true with Gensim >= 4.
        # model is a list of tuples (<n-gram>, <score>), where <n-gram> is a string
        # "<token 1><delimiter><token 2>"

        # The final list of sorted phrases
        self.phrases = []

        # Sort by scores then length, using set() to deduplicate
        scores = set(score for _, score in model)
        scores = sorted(list(scores), reverse=True)
        for score_tier in scores:
            score_phrases = set(
                phrase for phrase, score in model if score == score_tier
            )
            score_phrases = set(
                tuple(phrase.split(gensim_kwargs["delimiter"]))
                for phrase in score_phrases
            )

            # Recursively expanded phrase list (including "chained" phrases),
            # again using set() for deduplication
            merged = set()
            done = False
            while not done:
                # Start by assuming we are done; we will reset the flag if anything
                # shows up on this round
                done = True

                # For each original n-gram from the Phraser...
                for phrase in score_phrases:
                    # Check if it is completely non-alphabetic
                    if drop_non_alpha and not any(
                        [token.isalpha() for token in phrase]
                    ):
                        continue

                    # And ensure that the original phrase itself is in the final
                    # result set (assuming it passes the `drop_non_alpha` setting)
                    merged.add(phrase)

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

                        # Non-alphabetic check
                        if drop_non_alpha and not any(
                            [token.isalpha() for token in other]
                        ):
                            continue

                        new_phrase = None
                        if phrase[0] == other[-1]:
                            new_phrase = other + phrase[1:]
                        if phrase[-1] == other[0]:
                            new_phrase = phrase + other[1:]

                        # We also want to put in a hard limit on merged token length,
                        # in case we get phrases like (a, b), (b, c), (c, a), (a, c),
                        # etc. with many combinations.
                        if (
                            new_phrase
                            and new_phrase not in merged
                            and len(new_phrase) <= 5
                        ):
                            merged.add(new_phrase)
                            done = False

            # Sort by tokens alphabetically, then by length
            sorted_merged = sorted(list(merged), key=lambda x: " ".join(x))
            sorted_merged = sorted(sorted_merged, key=lambda x: len(x), reverse=True)
            self.phrases += [(list(phrase), score_tier) for phrase in sorted_merged]

        # Organise the new list of phrases for subsequent application
        self.by_first = {}
        self._map_by_first_token()

        # And done -- Ready to phrase.
        elapsed = time.perf_counter() - start_time
        if verbose:
            print(f"Improved Phraser initialised. {elapsed:.2f}s", flush=True)

    def _map_by_first_token(self):
        """
        Maps this instance's phrases to a Dictionary by first token for more
        efficient phrasing performance.

        (N.B.: Assumes `self.phrases` is already sorted by score then length)
        """
        for phrase, score in self.phrases:
            head = phrase[0]
            if head not in self.by_first:
                self.by_first[head] = []
            self.by_first[head].append((phrase, score))

    def find_ngrams_single(self, doc, threshold=-math.inf):
        """
        Perform n-gram replacement for a single document using the phrase model
        trained for this instance.

        Parameters
        ----------
        doc: iterable of str
            The document to phrase, as an iterable of string tokens.
        threshold: float, optional
            Optionally set a new phrasing threshold; if not set, will apply all the
            available phrases (which are determined by the value of `threshold` passed
            to the base `gensim` model on init)

        Returns
        -------
        iterable of str
            The document after it has undergone phrasing.
        """
        # Find all candidate phrase chains originating from each token,
        # then merge the highest-scoring one.
        new_doc = doc[:]
        index = 0

        while index < len(new_doc):
            token = new_doc[index]

            # Short-circuit here if there are no phrases that start with this token
            # (i.e., there aren't even any candidate phrases)
            if token not in self.by_first:
                index += 1
                continue

            best_candidate = self._find_best_candidate(token, index, new_doc, threshold)

            if best_candidate is None:
                # We checked the candidate phrases, and none of them fit
                index += 1
                continue

            # Preemptively merge any candidates found in the chain starting from
            # this token
            exhausted_chain = False
            while best_candidate["start_index"] > index:
                # Merge
                new_doc[
                    best_candidate["start_index"] : best_candidate["start_index"]
                    + len(best_candidate["phrase"])
                ] = [self.delimiter.join(best_candidate["phrase"])]

                # Get the next highest scoring candidate in the chain
                best_candidate = self._find_best_candidate(
                    token, index, new_doc, threshold
                )

                if best_candidate is None:
                    # There is no candidate for merging anywhere in the chain,
                    # including directly from the current token...
                    exhausted_chain = True
                    break

            if exhausted_chain:
                # ... so continue with the next token in the document
                # (Can't break out of the inner loop and continue the outer one in
                # one command)
                index += 1
                continue

            # Still here? Then there is a best candidate phrase for merging,
            # and it starts on this token
            assert best_candidate["start_index"] == index
            new_doc[index : index + len(best_candidate["phrase"])] = [
                self.delimiter.join(best_candidate["phrase"])
            ]
            index += 1

        return new_doc

    def find_ngrams(self, docs, threshold=-math.inf, verbose=False):
        """
        Perform n-gram replacement on the given documents using the phrase model
        trained for this instance.

        Detects significant bigrams by default, but bigrams with the same phrase
        score are merged, and common terms are ignored (as described in the class
        documentation).

        To detect the next higher order of n-grams (viz., trigrams and above),
        you will need to create a new `ImprovedPhraser` instance, passing the results
        of this function as the new set of base sentences to use.

        Parameters
        ----------
        docs: iterable of iterable of str
            Each doc in docs should be an iterable of string tokens.
        threshold: float, optional
            Optionally set a new phrasing threshold; if not set, will apply all the
            available phrases (which are determined by the value of `threshold`
            passed to the base `gensim` model on init).
        verbose: bool, optional
            Whether or not to show the phrasing progress.

        Returns
        -------
        iterable of iterable of str
            The documents, each an iterable of strings, after they have undergone
            phrasing.
        """
        new_docs = []

        if verbose:
            docs = tqdm(docs)

        for doc in docs:
            new_doc = self.find_ngrams_single(doc, threshold)
            new_docs.append(new_doc)

        return new_docs

    def _find_best_candidate(
        self, start_token, start_index, document, threshold, nesting_level=0
    ):
        """
        Examines all the candidate phrase chains originating from the given token in
        this Phraser model and returns the highest-scoring one.

        E.g., if the document is ["likes", "swimming", "pool", "tubes"] and "swimming
        pool" has a higher phrase score than the other two bigrams, it will be given
        priority even though it is in the middle of the chain (i.e., the phraser
        intentionally skips merging the lower-scoring "likes swimming").

        Parameters
        ----------
        start_token
        start_index
        document
        threshold
        nesting_level: int
            How many times we have recursively checked for best candidates within
            best candidates -- We set a sane hard limit on this.

        Returns
        -------
        dict with keys "start_index", "phrase", and "score"; or
        None if `start_token` is not in the model's phrases
        """
        if start_token not in self.by_first:
            return None

        if nesting_level >= 10:
            # Short-circuit, we don't need to go so deep looking for candidates,
            # especially since our max candidate length is also hard-limited
            return None

        # Maps next `start_token` -> `start_index` entries for recursion
        next_starts = {}

        best_candidate = None
        phrases = self.by_first[start_token]

        for phrase, score in phrases:
            if score < threshold:
                continue

            doc_tokens = document[start_index : start_index + len(phrase)]

            if tuple(doc_tokens) == tuple(phrase):
                # Found a candidate chain
                this_candidate = dict(
                    start_index=start_index, phrase=phrase, score=score
                )
            else:
                # This phrase does not match the document
                continue

            # Set new `best_candidate` if appropriate
            if best_candidate is None:
                best_candidate = this_candidate
            else:
                if this_candidate["score"] > best_candidate["score"] or (
                    this_candidate["score"] == best_candidate["score"]
                    and len(this_candidate["phrase"]) > len(best_candidate["phrase"])
                ):
                    best_candidate = this_candidate

            # If we are still in this loop, this phrase is valid, and could form a
            # chain with the next few tokens in the document; queue them for
            # recursive checking
            for index, next_start in enumerate(phrase[1:]):
                if (
                    document[start_index + 1 + index] == next_start
                    and next_start not in next_starts
                ):
                    next_starts[next_start] = start_index + 1 + index

        # Cash out the chains in `next_starts`
        for next_start, next_index in next_starts.items():
            next_candidate = self._find_best_candidate(
                next_start,
                next_index,
                tuple(document),
                threshold,
                nesting_level=nesting_level + 1,
            )
            if next_candidate is None:
                continue

            if next_candidate["score"] > best_candidate["score"] or (
                next_candidate["score"] == best_candidate["score"]
                and len(next_candidate["phrase"]) > len(best_candidate["phrase"])
            ):
                best_candidate = next_candidate

        return best_candidate
