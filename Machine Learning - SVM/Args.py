class Args:
    def __init__(self) -> None:
        # self.dataset = 'CIFAR-10'
        # self.img_shape = (3, 32, 32)
        # self.input_shape = (-1, 3*32*32)
        self.dataset = 'MNIST'
        self.img_shape = (28, 28)
        self.input_shape = (-1, 28*28)

        self.kernel = 'Linear'
        self.C = 1