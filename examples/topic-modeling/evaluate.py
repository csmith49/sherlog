import click

from sherlog.interface import initialize, print, minotaur
from sherlog.inference import Optimizer, Embedding, minibatch
from sherlog.inference.bayes import Point, Delta
from sherlog.program import loads

from typing import List, Mapping, Iterable
from itertools import chain
from random import random
from torch import ones, softmax

SCHEMA = {
    "topics" : ["nature", "technology"],
    "words" : ["cat", "dog", "computer", "mouse", "shed", "server"],
    "documents" : {
        "interfaces" : ["computer", "mouse", "server"]
    }
}

class Schema:
    def __init__(self, topics : List[str], words : List[str], documents : Mapping[str, List[str]]):
        self._topics = topics
        self._words = words
        self._documents = documents

        # TODO - Build per-document topic distribution parameters and per-topic word distribution parameters.
        #        They should be stored as tensors, with requires_grad=True. Furthermore, we should be able to
        #        do the following:
        #        1. Generate the parameters as "parameter" objects and pass them to the optimizer.
        #        2. Generate an observation of each parameter to be added to the evidence constructor.

    # DEFINING THE EDB

    def topics(self) -> Iterable[str]:
        for topic in self._topics:
            yield f"topic({topic})"

    def documents(self) -> Iterable[str]:
        for document in self._documents.keys():
            yield f"document({document})"

    def edb(self) -> Iterable[str]:
        yield from self.topics()
        yield from self.documents()

    @property
    def number_of_topics(self) -> int:
        return len(self._topics)
    
    @property
    def number_of_words(self) -> int:
        return len(self._words)

    # POINTS FOR BAYESIAN ESTIMATES

    def topics_points(self, *documents) -> Iterable[Point]:
        if documents:
            for document in documents:
                yield Point("topics", (document,))
        else:
            for document in self._documents.keys():
                yield Point("topics", (document,))

    def words_points(self) -> Iterable[Point]:
        for topic in self._topics:
            yield Point("words", (topic,))

    def points(self, *documents) -> Iterable[Point]:
        yield from self.topics_points(*documents)
        yield from self.words_points()

    # DEFINING THE IDB

    def words(self, document = None) -> Iterable[str]:
        for doc, tokens in self._documents.items():
            if document is None or doc == document:
                for index, token in enumerate(tokens):
                    yield f"word({doc}, {index}, {token})"
    
    def idb(self, *documents : str) -> Iterable[str]:
        if documents:
            for document in documents:
                yield from self.words(document)
        else:
            yield from self.words()

    # PARAMETERS

    def alpha(self) -> Iterable[str]:
        yield f"!parameter alpha : positive[{self.number_of_topics}]"

    def beta(self) -> Iterable[str]:
        yield f"!parameter beta : positive[{self.number_of_words}]"

    def parameters(self) -> Iterable[str]:
        yield from self.alpha()
        yield from self.beta()

    # RULES

    def plates(self) -> Iterable[str]:
        yield "topics(D; dirichlet[alpha]) <- document(D)"
        yield "words(T; dirichlet[beta]) <- topic(T)"
    
    def sample_topic(self) -> Iterable[str]:
        domain = ", ".join(self._topics)
        yield f"topic(D, I; {{{domain}}} <~ categorical[T]) <- topics(D, T)"
    
    def sample_word(self) -> Iterable[str]:
        domain = ", ".join(self._words)
        yield f"word(D, I; {{{domain}}} <~ categorical[W]) <- topic(D, I, T), words(T, W)"

    def rules(self) -> Iterable[str]:
        yield from self.plates()
        yield from self.sample_topic()
        yield from self.sample_word()

    #  SOURCE

    def source(self) -> str:
        result = ""
        lines = chain(self.parameters(), self.rules(), self.edb())
        for line in lines:
            result += f"{line}.\n"
        return result

    # EVIDENCE

    def evidence(self, documents = None, subsample : float = 1.0) -> str:
        observations = [atom for atom in self.idb(documents) if random() <= subsample]
        return ", ".join(observations)

@click.command()
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Optimizer learning rate.")
@click.option("-s", "--samples", default=1, type=int, help="Number of per-explanation executions.")
@click.option("-e", "--epochs", default=100, type=int, help="Number of optimization epochs.")
@click.option("-i", "--instrumentation", type=str, help="Instrumentation log destination.")
def cli(**kwargs):

    # initialize
    print("Initializing...")
    initialize(port=8007, instrumentation=kwargs["instrumentation"])

    minotaur.enter("topic-modeling")

    # construct the schema
    schema = Schema(
        topics=["tech", "nature"],
        words=["cat", "dog", "mouse", "shed", "server", "monitor", "tree", "graph"],
        documents={
            "interfaces" : ["monitor", "mouse", "server"],
            "algorithms" : ["server", "shed", "tree", "graph"],
            "animals" : ["cat", "dog", "mouse", "shed"]
        }
    )

    print(schema.source())
    print(schema.evidence())

    # then load the program source
    print("Loading the program and documents...")
    program, _ = loads(schema.source())
    
    data = [doc for doc in schema._documents.keys()]

    embedder = Embedding(
        evidence=lambda doc: schema.idb(doc),
        points=lambda doc: schema.points(doc)
    )
    
    optimizer = Optimizer(
        program=program,
        learning_rate=kwargs["learning_rate"],
        samples=kwargs["samples"],
        delta={
            "topics" : Delta(softmax(ones(schema.number_of_topics), dim=0)),
            "words" : Delta(softmax(ones(schema.number_of_words), dim=0))
        },
        points=schema.points()
    )

    # and optimize!
    for batch in minibatch(data, batch_size=len(data), epochs=kwargs["epochs"]):
        optimizer.maximize(*embedder.embed_all(batch.data))
        loss = optimizer.optimize()

        print(f"Epoch {batch.epoch}")
        print(f"Loss: {loss.item():.3f}")

        for point in schema.words_points():
            print(f"{point} - {optimizer.lookup_points((point,))}")
    minotaur.exit()

if __name__ == "__main__":
    cli()