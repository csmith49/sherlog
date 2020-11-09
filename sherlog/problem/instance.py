class Instance:
    def __init__(self, story, context):
        self.story = story
        self.context = context

    def loss(self):
        return self.story.loss(self.context)

    def likelihood(self, num_samples=100):
        return self.story.likelihood(self.context, num_samples=num_samples)