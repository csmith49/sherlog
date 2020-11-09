import torch

class Instance:
    def __init__(self, story, context):
        self.story = story
        self.context = context

    def sample_loss(self):
        return self.story.loss(self.context)

    def loss(self, num_samples=1):
        losses = []
        for i in range(num_samples):
            losses.append(self.sample_loss())
        losses = torch.stack(losses)
        return torch.mean(losses)

    def likelihood(self, num_samples=100):
        return self.story.likelihood(self.context, num_samples=num_samples)