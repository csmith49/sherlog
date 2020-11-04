class Instance:
    def __init__(self, story, context):
        self.story = story
        self.context = context

    def loss(self):
        context = self.story.run(self.context.clone())
        return self.story.loss(context)