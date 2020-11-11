import torch

class Instance:
    def __init__(self, story, context):
        self.story = story
        self.context = context

    def sample_loss(self):
        return self.story.loss(self.context)

    def sample_density(self):
        return self.story.density(self.context)

    def density(self, num_samples=1):
        ds = []
        for i in range(num_samples):
            ds.append(self.sample_density())
        ds = torch.stack(ds)
        return torch.mean(ds)

    def loss(self, num_samples=1):
        losses = []
        for i in range(num_samples):
            losses.append(self.sample_loss())
        losses = torch.stack(losses)
        return torch.mean(losses)

    def likelihood(self, num_samples=100):
        return self.story.likelihood(self.context, num_samples=num_samples)