#!/usr/bin/env python
# coding: utf-8

# Ignis: Text Pre-processing
# ==========================

# In[1]:


import glob
import pathlib
import re
import threading
import time

import gensim
import ipywidgets as widgets
import tqdm

# In[2]:


# Data ingestion
# --------------------
#
# We will track the contents and filename of each document, then tokenise them all and feed them into an `ignis.Corpus` that will be saved.
#
# We should, by all accounts, actually be preparing a separate text cleaning function and running the raw text through it immediately, but this way we can see the effects of each step of the data cleaning.

# In[3]:


raw_files = glob.glob("./data/bbc/*/*.txt")


# In[4]:


raw_docs = []
for file in tqdm.tqdm(raw_files):
    filename = pathlib.Path(file).as_posix()
    metadata = {"filename": filename}

    with open(file) as f:
        text = f.read()

    raw_docs.append([metadata, text])


# In[5]:


def show_raw_doc(doc_id=0):
    print(raw_docs[doc_id][0])
    print()
    print(raw_docs[doc_id][1])


widgets.interact(show_raw_doc, doc_id=(0, len(raw_docs) - 1))


# Text pre-processing and tokenisation
# ------

# ### Naive tokenisation (by whitespace)
# - Case folding
# - Strip leading/trailing non-informative punctuation from tokens
# - Remove single apostrophes
# - Remove single brackets within words
#   - For dealing with cases like "the recipient(s)" -- Which will get tokenised to "the recipient(s" otherwise

# In[6]:


strip_punctuation = "'\"()[]<>?!,.:;/|_"
bracket_pairs = [
    ["(", ")"],
    ["[", "]"],
]


def naive_tokenise(doc):
    """
    Naively tokenises a document.
    
    Returns
    -------
    str
        The document as a string of space-separated tokens
    """
    new_tokens = []

    tokens = doc.split()
    for token in tokens:
        token = token.casefold()
        token = token.strip(strip_punctuation)
        token = token.replace("'", "")

        for bracket_pair in bracket_pairs:
            if bracket_pair[0] in token and bracket_pair[1] not in token:
                token = token.replace(bracket_pair[0], "")
            if bracket_pair[1] in token and bracket_pair[0] not in token:
                token = token.replace(bracket_pair[1], "")

        if token != "":
            new_tokens.append(token)

    return new_tokens


# In[7]:


naive_tokenise('This is a t(e)st of the system\'s "tokenisation" operation(s).')


# In[8]:


naive_docs = []
for raw_doc in raw_docs:
    naive_docs.append([raw_doc[0], naive_tokenise(raw_doc[1])])


# In[9]:


def show_naive_doc(doc_id=0):
    print(naive_docs[doc_id][0])
    print()
    print(" ".join(naive_docs[doc_id][1]))


widgets.interact(show_naive_doc, doc_id=(0, len(naive_docs) - 1))


# ### Automated n-gram detection
#
# Chunk into significant bigrams/trigrams based on collocation frequency
#
# (N.B.: Gensim implies that the input to the Phraser should be a list of single sentences, but we will feed it a list of documents instead.)
#
# - Min count: How many documents the n-grams need to appear in
#
# - Scoring: "default" or "npmi"
#
# - Threshold: Intuitively, higher threshold means fewer phrases.
#   - With the default scorer, this is greater than or equal to 0; with the NPMI scorer, this is in the range -1 to 1.
#
# - Common terms: These terms will be ignored if they come between normal words.
#   - E.g., if `common_terms` includes the word "of", then when the phraser sees "Wheel of Fortune" it actually evaluates _"Wheel Fortune"_ as an n-gram, putting "of" back in only at the output level.

# In[10]:


min_count = 5
scoring = "npmi"
# We want a relatively high threshold so that we don't start littering spurious n-grams all over our corpus, diluting our results.
# E.g., we want "Lord_of_the_Rings", but not "slightly_better_than_analysts"
threshold = 0.7
common_terms = ["a", "an", "the", "of", "on", "in", "at"]


# This could take a while, so set up a threaded function with a basic progress indicator in the main thread

# In[11]:


def find_trigrams(docs, results):
    # Build, finalise, and apply the bigram model
    bigram_model = gensim.models.Phrases(
        docs,
        min_count=min_count,
        threshold=threshold,
        scoring=scoring,
        common_terms=common_terms,
    )
    bigram_model = gensim.models.phrases.Phraser(bigram_model)

    bigram_docs = bigram_model[docs]

    # Repeat to get trigrams
    trigram_model = gensim.models.Phrases(
        bigram_docs,
        min_count=min_count,
        threshold=threshold,
        scoring=scoring,
        common_terms=common_terms,
    )
    trigram_model = gensim.models.phrases.Phraser(trigram_model)

    trigram_docs = trigram_model[docs]

    results[0] = trigram_docs


# In[12]:


print("Finding trigrams", flush=True, end="")
start_time = time.perf_counter()

# Just send the textual content through the Phraser, not the document metadata
for_phrasing = [naive_doc[1] for naive_doc in naive_docs]

# Will contain the documents after trigram processing
results = [None]
t = threading.Thread(target=find_trigrams, args=(for_phrasing, results))
t.start()

progress_countdown = 1.0

while t.isAlive():
    time.sleep(0.1)
    progress_countdown -= 0.1
    if progress_countdown <= 0:
        print(" .", flush=True, end="")
        progress_countdown = 1

elapsed = time.perf_counter() - start_time
print(f" Done. ({elapsed:.3f}s)")

after_phrasing = results[0]

# Put metadata back in
phrased_docs = []
for index, tokens in enumerate(after_phrasing):
    phrased_docs.append([naive_docs[index][0], tokens])


# In[13]:


def show_phrased_doc(doc_id=0):
    print(phrased_docs[doc_id][0])
    print()
    print(" ".join(phrased_docs[doc_id][1]))


widgets.interact(show_phrased_doc, doc_id=(0, len(phrased_docs) - 1))


# ### Post-phrasing cleaning
#
# - Remove stop words (optional)
# - Remove purely numeric/non-alphabetic/single-character tokens
#   - Under the assumption that significant tokens, like the "19" in "Covid 19" or the "11" in "Chapter 11 (bankruptcy)" would have been picked up by the phraser

# In[14]:


# Not needed if using term weighting, but we could use stopsets from NLTK or other sources
stopset = []


def second_tokenise(tokens):
    new_tokens = []
    for token in tokens:
        if token in stopset or re.match("^[^a-z]+$", token) or len(token) <= 1:
            continue
        new_tokens.append(token)

    return new_tokens


# In[15]:


final_docs = []
for phrased_doc in phrased_docs:
    final_docs.append([phrased_doc[0], second_tokenise(phrased_doc[1])])


# In[16]:


def show_final_doc(doc_id=0):
    print(final_docs[doc_id][0])
    print()
    print(" ".join(final_docs[doc_id][1]))


widgets.interact(show_final_doc, doc_id=(0, len(final_docs) - 1))


# Save to Ignis Corpus
# ----

# In[17]:


import ignis


# In[18]:


corpus = ignis.Corpus()

for metadata, tokens in final_docs:
    corpus.add_doc(metadata, tokens)
corpus.save("bbc-full.corpus")


# In[19]:


# And make sure it loads without errors as well.
corpus = ignis.load_corpus("bbc-full.corpus")
