import ignis
import numpy

# numpy.random.seed(1234)

corpus = ignis.load_corpus("bbc-full.corpus")
model_options = {
    "k": 6,
    "term_weighting": "idf",
    "until_max_ll": False,
    "verbose": True,
    "seed": 7157,
}
vis_options = {"verbose": True}
results = ignis.train_model(
    corpus,
    model_type="tp_lda",
    model_options=model_options,
    vis_type="pyldavis",
    vis_options=vis_options,
)
