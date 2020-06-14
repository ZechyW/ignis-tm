import tomotopy as tp
import os

# Simple LDA over `docs.txt`
with open("docs.txt", "r", encoding="utf8") as fp:
    model = tp.LDAModel(k=5, seed=123456789)
    for line in fp:
        model.add_doc(line.split())

for i in range(0, 1000, 100):
    model.train(100, workers=1, parallel=tp.ParallelScheme.NONE)
    if i == 900:
        print(
            f"Iteration: {i + 100} LL: {model.ll_per_word:.5f} (HS: "
            f"{os.environ.get('PYTHONHASHSEED')})"
        )
