import os
import torch
import glob
import json

class Saver(object):
    def __init__(self, hyper_params):
        self.directory = 'run'
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = max([int(x.split('_')[-1]) for x in self.runs]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.hyper_params = hyper_params
        # Save Hyper Params
        json.dump(hyper_params, open(os.path.join(self.experiment_dir, 'hyper_params.txt'), 'wt'))

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            filename = os.path.join(self.experiment_dir,'best_model.pth.tar')
            torch.save(state, filename)

    def save_experiment_config(self, hyper_params):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')

        for key, val in hyper_params.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()
