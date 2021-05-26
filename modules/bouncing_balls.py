import os

import numpy as np

from torchvision.datasets.vision import VisionDataset


class BouncingBalls(VisionDataset):
    """The class implements a PyTorch-compatible version of the bouncing balls
    dataset. The dataset is able to provide the additional frame numerosity
    feature. When the numerosity information is available, the num flag must be
    enabled.
    """

    def __init__(
        self,
        root,
        split, # 'training', 'validation' or 'testing'
        num=False # return frame numerosity
    ):
        super(BouncingBalls, self).__init__(root)
        
        # load the dataset
        files = os.listdir(os.path.join(root, split))
        data = []
        for file in files:
            data.append(np.load(os.path.join(root, split, file)))

        if not num:
            res = int(np.sqrt(data[0].shape[1]))
        else:
            res = int(np.sqrt(data[0].shape[1] - 1))
        shape = (data[0].shape[0], res, res)
        if not num:
            data = [seq.reshape(shape) for seq in data]
        else:
            data = [(seq[0,0], seq[:,1:].reshape(shape)) for seq in data]

        self.num = num
        self.res = res
        self.data = data

    def __getitem__(self, index):
        seq = self.data[index]
        return seq if not self.num\
            else int(seq[0]), seq[1] # numerosity feature, sequence

    def __len__(self):
        return len(self.data)