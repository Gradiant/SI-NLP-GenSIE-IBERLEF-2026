from .Direct import Direct


class Categorical(Direct):

    def estimate(self, task, fields):
        super().estimate(task, fields)