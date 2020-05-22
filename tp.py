# HDP Test
import glob
import pathlib
import string
import time

import gensim
import nltk as nltk
import tomotopy as tp
import tqdm

from utils import doc_to_tokens

# Config
model_seed = 11399
data_path = pathlib.Path("D:/Datasets/bbc")
train_and_save = True
model_file = "model.bin"
label_topics = True

# Tomotopy model
model = tp.LDAModel(seed=model_seed, k=20, rm_top=10)

# Text pre-processing
stopset = set(nltk.corpus.stopwords.words("english") + list(string.punctuation))
# Pronouns, titles
stopset.update(
    ["i", "i'm", "i'd", "i've", "i'll"]
    + ["you", "you're", "you'd", "you'll"]
    + ["she", "she's", "she'd", "she'll"]
    + ["he", "he's", "he'd", "he'll"]
    + ["we", "we're", "we'd", "we'll"]
    + ["they", "they're", "they'd", "they'll"]
    + ["mr", "mrs", "ms", "dr"]
)
# Modals
stopset.update(["would", "will", "could", "can", "should", "shall"])

# Ingest text files from directory (e.g., BBC dataset)
print(f"Reading dataset from {data_path}...")
docs = []
for file in tqdm.tqdm(glob.glob(f"{data_path}/*/*.txt")):
    with open(file) as f:
        doc = f.read()
        tokens = doc_to_tokens(doc, stopset=stopset, non_alphabetic="split")
        docs.append(tokens)

print("Done.")

# Use Gensim's phraser to chunk bigrams/trigrams together
# Trigrams (or maximally 4-grams) built iteratively from significant collocations with
# bigrams
print("Generating common bigrams/trigrams...")

bigram = gensim.models.Phrases(docs, min_count=5, threshold=10)
trigram = gensim.models.Phrases(bigram[docs], threshold=10)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

for doc in tqdm.tqdm(docs):
    model.add_doc(trigram_mod[bigram_mod[doc]])

print("Done.")

# Model training
if train_and_save:
    model.train(0, workers=0)
    print(
        f"Num docs: {len(model.docs)}, Vocab size: {model.num_vocabs}, "
        f"Num words: {model.num_words}"
    )
    print(f"Removed top words: {model.removed_top_words}")

    print("Training model...")

    train_batch = 50
    train_total = 500
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
    print(f"Saving to {model_file}.")
    model.save(model_file)
else:
    model = tp.HLDAModel.load(model_file)

# Automated topic labelling
if label_topics:
    print("Extracting suggested topic labels...")
    # extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    extractor = tp.label.PMIExtractor(min_cf=5, min_df=3, max_len=5, max_cand=20000)
    candidates = extractor.extract(model)
    # labeler = tp.label.FoRelevance(model, candidates, min_df=5, smoothing=1e-2,
    # mu=0.25)
    labeler = tp.label.FoRelevance(model, candidates, min_df=3, smoothing=1e-2, mu=0.25)
    print("Done.")

print("-" * 10)


# Results by topic
def print_topic(topic_id):
    # Labels
    if label_topics:
        labels = ", ".join(
            label for label, score in labeler.get_topic_labels(topic_id, top_n=10)
        )
        print(f"Suggested labels: {labels}")

    # Print this topic
    words_probs = model.get_topic_words(topic_id, top_n=10)
    words = [x[0] for x in words_probs]

    words = ", ".join(words)
    print(words)


for k in range(model.k):
    print(f"[Topic {k}]")
    print_topic(k)
    print()
