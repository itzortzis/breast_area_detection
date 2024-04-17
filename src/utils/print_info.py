




class Printer:
    def __init__(self, comps):
        self.comps = comps
        self.print_logo()
        self.print_train_details()
        
        
    def print_logo(self):
        print("""\
                    *    *    * *
         **  **     *        *
        *  **  *   ***   *  ***
        *      *    *    *   *
        *      *    **   **  *
                    """)


    def print_train_details(self):
        self.print_logo()
        print('You are about to train the model on ' + self.comps.dtst_name)
        print('with the following details:')
        print('\t Training epochs: ', self.comps.epochs)
        print('\t Epoch threshold: ', self.comps.epoch_thr)
        print('\t Score threshold: ', self.comps.score_thr)
        print('\t Trained models path: ', self.comps.trained_models)
        print('\t Metrics path: ', self.comps.metrics)
        print('\t Device: ', self.comps.device)
        print()
        # option = input("Do you wish to continue? [Y/n]: ")
        return True or (option == 'Y' or option == 'y')
        