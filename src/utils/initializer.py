



class INIT:
    def __init__(self, components, params, paths):
        self.parameters = params
        self.components = components
        self.paths = paths
        print()
        
    def run(self):
        self.init_components()
        self.init_parameters()
        self.init_paths()
        
    def init_components(self):
        self.model     = self.components['model']
        self.opt       = self.components['opt']
        self.loss_fn   = self.components['loss_fn']
        self.dataset = self.components['dataset']


    def init_parameters(self):
        self.epochs     = self.parameters['epochs']
        self.dtst_name  = self.parameters['dtst_name']
        self.epoch_thr  = self.parameters['epoch_thresh']
        self.score_thr  = self.parameters['score_thresh']
        self.device     = self.parameters['device']
        self.batch_size = self.parameters['batch_size']
        self.inf_model  = self.parameters['inf_model_name']

    def init_paths(self):
        self.trained_models = self.paths['trained_models']
        self.metrics = self.paths['metrics']
        self.figures = self.paths['figures']