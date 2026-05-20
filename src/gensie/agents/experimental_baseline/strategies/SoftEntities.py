from .FixedEntities import FixedEntities

class SoftEntities(FixedEntities):

    def estimate(self, task, fields, token_per_second=None):
        super().estimate(task, fields, token_per_second=token_per_second)