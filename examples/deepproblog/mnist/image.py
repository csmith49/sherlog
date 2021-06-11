import torchvision
import torchvision.transforms as transforms
from dataclasses import dataclass
from random import randint
from sherlog.logs import get_external

logger = get_external("npi.mnist.example")

# load the datasets
logger.info("Loading MNIST data...")
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
DATA_DIR = "/tmp"
TRAIN = torchvision.datasets.MNIST(
    root=f"{DATA_DIR}/MNIST",
    train=True,
    download=True,
    transform=TRANSFORM
)
TEST = torchvision.datasets.MNIST(
    root=f"{DATA_DIR}/MNIST",
    train=False,
    download=True,
    transform=TRANSFORM
)
logger.info("MNIST data loaded from {DATA_DIR}/MNIST.")


# images are just references indexing above
@dataclass
class Image:
    dataset : str
    index : int

    def get(self):
        if self.dataset == "train": return TRAIN[self.index]
        elif self.dataset == "test": return TEST[self.index]
        else: raise KeyError()

    @property
    def vector(self):
        return self.get()[0]

    @property    
    def label(self):
        return self.get()[1]

    @property
    def atom(self):
        return f"{self.dataset}({self.index})"

    @property
    def symbol(self):
        return f"image_{self.dataset}_{self.index}"

# examples pair images and results
@dataclass
class Example:
    left : Image
    right : Image
    total : int

    def output_permutations(self):
        yield from range(0, 19)

# random access into train and test
def random_image(dataset):
    if dataset == "train":
        index = randint(0, len(TRAIN) - 1)
    elif dataset == "test":
        index = randint(0, len(TEST) - 1)
    else: raise KeyError()

    return Image(dataset, index)

def sample(quantity : int, dataset : str):
    for _ in range(quantity):
        left = random_image(dataset)
        right = random_image(dataset)
        total = left.label + right.label

        yield Example(left, right, total)