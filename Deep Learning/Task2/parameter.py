class parameters():
    def __init__(self) -> None:
        self.model_save_dir = './model'
        self.fig_save_dir = './figures'
        self.data_type = 'Augment_200'
        self.batchsize = 10
        self.dataset_dir = './dataset'
        self.epochs = 80
        self.input_size = 28
        self.output_size = 10
        self.hidden_layer = 64