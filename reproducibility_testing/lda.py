import tomotopy as tp
import os
import numpy
import random

# Simple LDA over `docs.txt`
docs = []
for line in open("docs.txt", "r", encoding="utf8"):
    doc = line.split()
    docs.append(doc)

for np_seed in range(1, 3):
    numpy.random.seed(np_seed)
    for r_seed in range(1, 3):
        random.seed(r_seed)

        model = tp.LDAModel(k=5, seed=123456789)
        for doc in docs:
            model.add_doc(doc)

        for i in range(0, 1000, 100):
            model.train(100, workers=1, parallel=tp.ParallelScheme.NONE)
            if i == 900:
                print(
                    f"Iteration: {i + 100} LL: {model.ll_per_word:.5f} "
                    f"(PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}) "
                    f"(numpy.random.seed: {np_seed}) "
                    f"(random.seed: {r_seed})"
                )
