import torchvision
import torchvision.transforms as transforms
from ...logs import get
from ..evaluation.datasource import DataSource
from random import randint

logger = get("tooling.data.mnist")

def _load(location, train=True):
    logger.info("Importing MNIST data...")
    return torchvision.datasets.MNIST(
        root=f"{location}/MNIST",
        train=train,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,)
            )
        ])
    )

class Image:
    def __init__(self, data, label, index, train):
        self.data = data
        self.label = label
        self.index = index
        self.train = train

    def atom(self):
        if self.train:
            return f"train({self.index})"
        else:
            return f"test({self.index})"

    def symbol(self):
        if self.train:
            return f"image_train_{self.index}"
        else:
            return f"image_test_{self.index}"

class MNISTDataSource(DataSource):
    def __init__(self, location="/tmp"):
        self._distributions = {
            "train" : _load(location, train=True),
            "test" : _load(location, train=False)
        }

    def _sample(self, train=True):
        if train:
            distribution = self._distributions["train"]
        else:
            distribution = self._distributions["test"]

        index = randint(0, len(distribution))
        data, label = distribution[index]

        return Image(data, label, index, train)

    def training_data(self, *args, quantity=1, **kwargs):
        for _ in range(quantity):
            yield self._sample(train=True)

    def testing_data(self, *args, quantity=1, **kwargs):
        for _ in range(quantity):
            yield self._sample(train=True)
