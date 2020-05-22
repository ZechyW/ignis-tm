#!/usr/bin/env python
# coding: utf-8

# Basic Text Pre-processing and Topic Modelling
# ======

# In[1]:


import glob
import math
import re
import threading
import time

import gensim
import nltk
import pyLDAvis
import tomotopy as tp
import tqdm


# Data ingestion
# -----

# In[2]:


data_files = glob.glob("./data/bbc/*/*.txt")


# In[3]:


raw_docs = []
for file in tqdm.tqdm(data_files):
    with open(file) as f:
        doc = f.read()
        raw_docs.append(doc)

print(raw_docs[0])


# Text pre-processing and tokenisation
# ------

# - Naive tokenisation (by whitespace)
# - Strip leading/trailing non-informative punctuation from tokens

# In[4]:


remove_punctuation = "'\"()?!,."


def naive_tokenise(doc):
    tokens = doc.split()
    tokens = [x.strip(remove_punctuation) for x in tokens]
    return tokens


docs = [naive_tokenise(doc) for doc in raw_docs]

print(" ".join(docs[0]))


# Chunk into significant bigrams/trigrams based on collocation frequency
# - Min count: Must appear in at least 0.1% of the documents
# - Threshold: Intuitively, higher threshold means fewer phrases
# - Common terms: These terms will be ignored if they come between normal words. E.g., if `common_terms` includes the word "of", then when the phraser sees "Wheel of Fortune" it actually evaluates _"Wheel Fortune"_ as an n-gram, putting "of" back in only at the output level.

# In[5]:


min_count = math.ceil(len(docs) / 1000)
threshold = 25
common_terms = ["a", "an", "the", "of", "on", "in", "at"]


# This could take a while, so set up a threaded function with a basic progress indicator in the main thread

# In[6]:


def find_ngrams(docs, results):
    bigram = gensim.models.Phrases(
        docs, min_count=min_count, threshold=threshold, common_terms=common_terms
    )
    trigram = gensim.models.Phrases(
        bigram[docs],
        min_count=min_count,
        threshold=threshold,
        common_terms=common_terms,
    )

    # Finalise the bigram/trigram generators for efficiency
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    results[0] = bigram_mod
    results[1] = trigram_mod


# In[7]:


print("Generating n-grams", flush=True, end="")

results = [None, None]
t = threading.Thread(target=find_ngrams, args=(docs, results))
t.start()

progress_countdown = 1.0

while t.isAlive():
    time.sleep(0.1)
    progress_countdown -= 0.1
    if progress_countdown <= 0:
        print(" .", flush=True, end="")
        progress_countdown = 1

print(" Done.")

bigram_mod = results[0]
trigram_mod = results[1]

docs = [trigram_mod[bigram_mod[doc]] for doc in docs]

print(" ".join(docs[0]))


# Second-pass tokenisation
# - Case folding
# - Remove single apostrophes
# - Remove stop words, remove purely numeric/non-alphabetic tokens

# In[8]:


stopset = set(nltk.corpus.stopwords.words("english"))


def second_tokenise(tokens):
    new_tokens = []
    for token in tokens:
        token = token.casefold()
        token = token.replace("'", "")
        if token in stopset or re.match("^[^a-z]+$", token):
            continue
        new_tokens.append(token)

    return new_tokens


docs = [second_tokenise(doc) for doc in docs]

print(" ".join(docs[0]))


# Model training (LDA)
# ----

# Add processed docs to the LDA model and train it

# In[9]:


model_seed = 11399
num_topics = 20
train_and_save = True
model_file = "model.bin"
# Training iterations
train_batch = 50
train_total = 500


# In[10]:


if train_and_save:
    model = tp.LDAModel(seed=model_seed, k=num_topics)

    for doc in tqdm.tqdm(docs):
        model.add_doc(doc)

    model.train(0, workers=0)
    print(
        f"Num docs: {len(model.docs)}, Vocab size: {model.num_vocabs}, "
        f"Num words: {model.num_words}"
    )
    print(f"Removed top words: {model.removed_top_words}")

    print("Training model...", flush=True)

    try:
        for i in range(0, train_total, train_batch):
            start_time = time.perf_counter()
            model.train(train_batch, workers=0)
            elapsed = time.perf_counter() - start_time
            print(
                f"Iteration: {i + train_batch}\tLog-likelihood: {model.ll_per_word}\t"
                f"Time: {elapsed:.3f}s",
                flush=True,
            )
    except KeyboardInterrupt:
        print("Stopping train sequence.")
    model.save(model_file)
    print(f"Saved to '{model_file}'.")
else:
    model = tp.LDAModel.load(model_file)
    print(f"Loaded from '{model_file}'.")


# Topic labelling

# In[11]:


print("Extracting suggested topic labels...", flush=True)
# extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
extractor = tp.label.PMIExtractor(min_cf=5, min_df=3, max_len=5, max_cand=20000)
candidates = extractor.extract(model)
# labeler = tp.label.FoRelevance(model, candidates, min_df=5, smoothing=1e-2,
# mu=0.25)
labeler = tp.label.FoRelevance(model, candidates, min_df=3, smoothing=1e-2, mu=0.25)
print("Done.")


# Print results
# ------

# In[12]:


def print_topic(topic_id):
    # Labels
    labels = ", ".join(
        label for label, score in labeler.get_topic_labels(topic_id, top_n=10)
    )
    print(f"Suggested labels: {labels}")

    # Print this topic
    words_probs = model.get_topic_words(topic_id, top_n=10)
    words = [x[0] for x in words_probs]

    words = ", ".join(words)
    print(words)


# In[13]:


for k in range(model.k):
    print(f"[Topic {k+1}]")
    print_topic(k)
    print()


# Visualise
# --------
# - Present data in the format expected by pyLDAvis

# In[14]:


model_data = {
    "topic_term_dists": [model.get_topic_word_dist(k) for k in range(model.k)],
    "doc_topic_dists": [model.docs[n].get_topic_dist() for n in range(len(model.docs))],
    "doc_lengths": [len(model.docs[n].words) for n in range(len(model.docs))],
    "vocab": model.vocabs,
    "term_frequency": model.vocab_freq,
}


# Again, this could take a while

# In[15]:


def prepare_vis(model_data, results):
    vis_data = pyLDAvis.prepare(**model_data)
    results[0] = vis_data


# In[16]:


print("Preparing LDA visualisation", flush=True, end="")

results = [None]
t = threading.Thread(target=prepare_vis, args=(model_data, results))
t.start()

progress_countdown = 1.0

while t.isAlive():
    time.sleep(0.1)
    progress_countdown -= 0.1
    if progress_countdown <= 0:
        print(" .", flush=True, end="")
        progress_countdown = 1

print(" Done.")

vis_data = results[0]


# In[17]:


pyLDAvis.display(vis_data)


# Iterate
# --------
# - See what the main topics might be, slice initial corpus and re-run LDA to get sub-topics
