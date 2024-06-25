import os.path as osp
import os
import copy
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

class CongestionDataset(Dataset):
    def __init__(self, dataroot, pipeline=None, test_mode=True, **kwargs):
        super().__init__()
        self.dataroot = dataroot
        self.test_mode = test_mode
        if pipeline:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        feature_root = osp.join(self.dataroot, 'feature')
        label_root = osp.join(self.dataroot, 'label')

        feature_files = sorted(os.listdir(feature_root))
        label_files = sorted(os.listdir(label_root))

        for feature_file, label_file in zip(feature_files, label_files):
            feature_path = osp.join(feature_root, feature_file)
            label_path = osp.join(label_root, label_file)
            data_infos.append(dict(feature_path=feature_path, label_path=label_path))

        return data_infos

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results['feature'] = np.load(results['feature_path'])
        
        if not self.test_mode:
            results['label'] = np.load(results['label_path'])

        if self.pipeline:
            processed_results = self.pipeline(results)
            if not self.test_mode:
                processed_results['label_path'] = results['label_path']  # 确保 label_path 被保留
        else:
            processed_results = results

        feature = processed_results['feature'].float()

        if not self.test_mode:
            label = processed_results['label'].float()
            return feature, label, processed_results['label_path']
        else:
            return feature, processed_results['label_path']

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)
