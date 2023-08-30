class SquareLR(object):
    def __init__(self, optimizer, factor: float = ..., init_lr: int = 0):

        # Parameters
        self.optimizer = optimizer
        self.factor = factor
        self.init_lr = init_lr
        self.lr = optimizer.param_groups[0]['lr']

    def step(self):
        if self.init_lr == 0:
            self.lr = self.optimizer.param_groups[0]['lr']

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr ** self.factor
        self.lr = self.lr ** self.factor

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]