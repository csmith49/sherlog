import sherlog
import torch
import torchvision

register = sherlog.Register()

raw_data = torchvision.datasets.MNIST(root="/tmp", download=True, transform=torchvision.transforms.ToTensor())

def one-hot(integer, indices=10):
    return torch.nn.functional.one_hot(torch.tensor(integer), num_classes=indices)

def convert_img(img):
    array, cls = img
    # convert the array to a 1-d tensor
    vector = array.view(array.shape[0], -1)[0]
    # construct the 1-hot encoding
    return (vector, one_hot(cls, indices=10))

dataset = [convert_img(img) for img in raw_data]

img_size = 784
hidden_dim = 100
output_size = 10

classify_digit = torch.nn.Sequential(
    torch.nn.Linear(img_size, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden, dim, output_size)
)
registrar.tag(classify_digit, name="classify_digit")

