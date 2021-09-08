"""Sherlog example: reviewer confidence.

TODO - finish source, connect with data.
"""

import torchtext
from torchtext.datasets import YelpReviewFull
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from itertools import islice, chain

from torch import tensor, cat, cumsum, randint, nn
import numpy as np

# constants for the reviewer model
NUMBER_OF_DOCUMENTS = 3000
NUMBER_OF_TOPICS = 8
NUMBER_OF_JUDGES = 20
JUDGES_PER_PAPER = 3

# load the data
dataset = list(islice(YelpReviewFull(root="/tmp/yelp", split=("train")), NUMBER_OF_DOCUMENTS))

tokenizer = get_tokenizer("basic_english")
vocabulary = build_vocab_from_iterator(tokenizer(review) for _, review in dataset)

documents = [tensor(vocabulary(tokenizer(review))) for _, review in dataset]
flat_documents = cat(documents)
# what a mess
offsets = cumsum(tensor([len(review) for review in chain([[]], documents[:-1])]), 0)

topics = randint(0, NUMBER_OF_TOPICS, (NUMBER_OF_DOCUMENTS,))
judges = tensor([
    np.random.choice(NUMBER_OF_JUDGES, JUDGES_PER_PAPER, replace=False) for _ in range(NUMBER_OF_DOCUMENTS)
]).long()

# model
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(len(vocabulary), 64, sparse=True)
        self.fc = nn.Linear(64, 5)
    
    def forward(self, text, offsets):
        return self.fc(self.embedding(text, offsets))

SOURCE = \
"""
threshold(Judge; normal[4, 0.5]) <- judge(Judge).
expertise(Judge, Topic; gamma[1, 1]) <- judge(Judge), topic(Topic).

quality_prob(Paper; quality_nn[Paper]) <- paper(Paper).
quality(Paper; categorical[P]) <- quality_prob(Paper, P).
"""