import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(directory, max_dataset_size=float('inf')):
    imgs = []
    assert os.path.isdir(directory), '\u2755 %s is not a valid directory' % directory
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                imgs.append(path)
    return imgs[:min(max_dataset_size, len(imgs))]
