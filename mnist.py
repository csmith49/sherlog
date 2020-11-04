import sherlog
import torch
import torchvision

register = sherlog.Register()

raw_data = torchvision.datasets.MNIST(root="/tmp", download=True, transform=torchvision.transforms.ToTensor())

def convert_img(img):
    array, cls = img
    # convert the array to a 1-d tensor
    vector = array.view(array.shape[0], -1)[0]
    # construct the 1-hot encoding
    classification = torch.nn.functional.one_hot(torch.tensor(cls), num_classes=10)
    return (vector, classification)

dataset = [convert_img(img) for img in raw_data]
register.tag(dataset, name="dataset")

img_size = 784
latent_dimension = 10
mean_hidden_dimension = 100
sdev_hidden_dimension = 100
decode_hidden_dimension = 100

encode_mean = torch.nn.Sequential(
    torch.nn.Linear(img_size, mean_hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(mean_hidden_dimension, latent_dimension)
)
register.tag(encode_mean, name="encode_mean")

encode_sdev = torch.nn.Sequential(
    torch.nn.Linear(img_size, sdev_hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(sdev_hidden_dimension, latent_dimension)
)
register.tag(encode_sdev, name="encode_sdev")

decode = torch.nn.Sequential(
    torch.nn.Linear(latent_dimension, decode_hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(decode_hidden_dimension, img_size)
)
register.tag(decode, name="decode")

def reconstruction_loss(x, y):
    loss = torch.dist(x, y, p=2)
    return loss
register.tag(reconstruction_loss, name="reconstruction_loss")
