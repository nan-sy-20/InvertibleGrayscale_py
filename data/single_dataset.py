from PIL import Image

from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_transform

class SingleDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__()
        self.A_paths = sorted(make_dataset(opt.dataroot. opt.max_dataset_size))
        input_nc = opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path)
        A = self.transform(A_img)
        return {'A': A, 'A_path': A_path}

    def __len__(self):
        if self.opt.max_dataset_size == -1:
            return len(self.A_paths)
        else:
            return self.opt.max_dataset_size
