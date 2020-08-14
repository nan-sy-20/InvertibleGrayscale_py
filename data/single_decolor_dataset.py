from PIL import Image
from skimage import color

from data.base_dataset import BaseDataset, get_transform

class SingleDecolorDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        with open(opt.dataroot+'.csv', newline='') as f:
            lines = f.readlines()
        self.A_paths = []
        for line in lines:
            line = line.rstrip('\r\n')
            self.A_paths.append(line)

        self.transform_rgb = get_transform(opt, grayscale=False)
        #self.transform_lum = get_transform(opt, grayscale=True)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path)
        A = self.transform_rgb(A_img)

        # lab = color.rgb2lab(A_img)
        # lum = lab[:,:,0]
        # lum_ = Image.fromarray(lum/100.0)
        # B = self.transform_lum(lum_)
        # return {'A': A, 'A_paths': A_path, 'B':B}
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        if self.opt.max_dataset_size == -1:
            return len(self.A_paths)
        else:
            return self.opt.max_dataset_size
