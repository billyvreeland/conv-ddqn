
from collections import deque
import csv
import datetime
from os import getcwd, mkdir, path
import pickle
from tempfile import mkdtemp


class GymLogger:
    def __init__(self, meta_data_dict, value_names=None):
        self.episode_history = deque()
        self.value_names = value_names
        self.setup_output()
        self.write_meta_data(meta_data_dict)

    def setup_output(self):
        cwd = getcwd()
        top_dir = path.join(cwd, 'recorded_results')
        if not path.exists(top_dir):
            mkdir(top_dir)
        now = datetime.datetime.now()
        output_dir = path.join(top_dir, now.strftime('%Y-%m-%d'))
        if not path.exists(output_dir):
            mkdir(output_dir)
        prefix = str(now).replace(':', '-').replace(' ', '-')
        self.output_folder = mkdtemp(dir=output_dir, prefix=prefix)

    def write_meta_data(self, meta_data):
        """
        Save meta data associated with test
        """
        sorted_keys = sorted(meta_data.keys())
        sorted_vals = [meta_data[k] for k in sorted_keys]
        with open(path.join(self.output_folder, 'meta_data.csv'), 'w+') as csvfile:
            meta_data_writer = csv.writer(csvfile)
            meta_data_writer.writerow(sorted_keys)
            meta_data_writer.writerow(sorted_vals)

    def update(self, episode_info):
        self.episode_history.append(episode_info)

    def print_status(self):
        pass

    def save_model_weights(self, weights, episode_num):
        weights_file_name = 'episode_' + str(episode_num) + '_weights.pickle'
        with open(path.join(self.output_folder, weights_file_name), 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_history(self):
        with open(path.join(self.output_folder, 'episode_history.csv'), 'w+') as csvfile:
            history_writer = csv.writer(csvfile)
            if self.value_names is not None:
                history_writer.writerow(self.value_names)
            for e in self.episode_history:
                history_writer.writerow(e)

        print('Results saved to ' + self.output_folder)
