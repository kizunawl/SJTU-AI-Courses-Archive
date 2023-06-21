class parameters():
    def __init__(self) -> None:
        self.model_save_dir = './model'
        self.fig_save_dir = './figures'
        self.data_type = 'grid'
        self.batchsize = 2
        self.dataset_dir = './dataset'
        self.epochs = 20
        self.input_size = 32
        self.output_size = 10
        self.hidden_layer = 64
        self.pad = 2
        self.kernel_size = 3
        self.slice = 2
        self.side_len = 16