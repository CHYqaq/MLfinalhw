import os.path as osp
import os
import copy
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

class CongestionDataset(Dataset):
    def __init__(self, data_infos, pipeline=None, test_mode=True, **kwargs):
        super().__init__()
        self.data_infos = data_infos
        self.test_mode = test_mode
        if pipeline:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results['feature'] = np.load(results['feature_path'])
        results['label'] = np.load(results['label_path'])

        results = self.pipeline(results) if self.pipeline else results

        feature =  results['feature'].float()
        label = results['label'].float()

        return feature, label, results['label']

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

def load_annotations(dataroot):
    data_infos = []
    feature_root = osp.join(dataroot, 'feature')
    label_root = osp.join(dataroot, 'label')

    feature_files = sorted(os.listdir(feature_root))
    label_files = sorted(os.listdir(label_root))

    for feature_file, label_file in zip(feature_files, label_files):
        feature_path = osp.join(feature_root, feature_file)
        label_path = osp.join(label_root, label_file)
        data_infos.append(dict(feature_path=feature_path, label_path=label_path))

    return data_infos
