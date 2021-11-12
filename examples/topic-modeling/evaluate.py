import click
from sherlog.inference.embedding import StringEmbedding

from sherlog.interface import initialize, print, minotaur
from sherlog.inference import Optimizer, StringEmbedding
from sherlog.program import loads

from typing import List, Mapping, Iterable
from itertools import chain
from random import random

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

    # DEFINING THE IDB

    def words(self) -> Iterable[str]:
        for document, tokens in self._documents.items():
            for index, token in enumerate(tokens):
                yield f"word({document}, {index}, {token})"
    
    def idb(self) -> Iterable[str]:
        yield from self.words()

    # PARAMETERS

    def alpha(self) -> Iterable[str]:
        yield f"!parameter alpha : positive[{len(self._topics)}]"

    def beta(self) -> Iterable[str]:
        yield f"!parameter beta : positive[{len(self._words)}]"

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

    def evidence(self, subsample : float = 1.0) -> str:
        observations = [atom for atom in self.idb() if random() <= subsample]
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
    embedder = StringEmbedding()
    optimizer = Optimizer(
        program=program,
        learning_rate=kwargs["learning_rate"],
        samples=kwargs["samples"]
    )

    # and optimize!
    for epoch in range(kwargs["epochs"]):
        optimizer.maximize(embedder.embed(schema.evidence()))
        loss = optimizer.optimize()

        print(epoch, loss)

    minotaur.exit()

if __name__ == "__main__":
    cli()