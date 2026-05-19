from .Direct import Direct


class Categorical(Direct):

    def estimate(self, task, fields, token_per_second=None):
        super().estimate(task, fields, token_per_second=token_per_second)