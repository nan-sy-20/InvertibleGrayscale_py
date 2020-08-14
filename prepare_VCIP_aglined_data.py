'''
VCIP challenge dataset
images are well cropped to 256x256
'''

import os
from utils.util import clear_folder, find_nth
from PIL import Image
from shutil import copyfile

gen_combined = False

if gen_combined:
    # training data
    data_root = '/home/synan/dataset/VCIP'
    a_dir = os.path.join(data_root, 'NIR')
    b_dir = os.path.join(data_root, 'RGB-Registered')
    ab_dir = os.path.join(data_root, 'aligned', 'all_og')

    clear_folder(ab_dir)
    for file in os.listdir(a_dir):
        name, ext = os.path.splitext(file)
        filename = name[:find_nth(name, '_', 2)]
        img_a = Image.open(os.path.join(a_dir, filename+'_nir'+ext))
        img_b = Image.open(os.path.join(b_dir, filename+'_rgb_reg'+ext))
        w_a, h_a = img_a.size
        w_b, h_b = img_b.size
        if h_a == h_b:
            dst = Image.new('RGB', (w_a + w_b, h_a))
            dst.paste(img_a, (0, 0))
            dst.paste(img_b, (w_b, 0))
            dst.save(os.path.join(ab_dir, filename+ext))
        else:
            print('img height error')
    os.system('ls %s | wc -l' % a_dir)
    os.system('ls %s | wc -l' % ab_dir)

    '''split'''
    eval_list = ['country_0083.png', 'field_0093.png', 'forest_0099.png', 'mountain_0093.png']
    tr_folder = os.path.join(data_root, 'aligned', 'train')
    ev_folder = os.path.join(data_root, 'aligned', 'eval')

    clear_folder(tr_folder)
    clear_folder(ev_folder)
    for file in os.listdir(ab_dir):
        if file not in eval_list:
            copyfile(os.path.join(ab_dir, file), os.path.join(tr_folder, file))
        else:
            copyfile(os.path.join(ab_dir, file), os.path.join(ev_folder, file))
    os.system('ls %s | wc -l' % tr_folder)
    os.system('ls %s | wc -l' % ev_folder)

    # testing data
    test_dir = os.path.join(data_root, 'Validation')
    new_dir = os.path.join(data_root, 'aligned', 'test')
    clear_folder(new_dir)
    for file in os.listdir(test_dir):
        name, ext = os.path.splitext(file)
        real_name = name[:find_nth(name, '_', 2)]
        if 'nir' in file:
            img_a = Image.open(os.path.join(test_dir, real_name+'_nir'+ext))
            img_b = Image.open(os.path.join(test_dir, real_name+'_rgb_reg'+ext))
            w_a, h_a = img_a.size
            w_b, h_b = img_b.size
            if h_a == h_b:
                dst = Image.new('RGB', (w_a + w_b, h_a))
                dst.paste(img_a, (0, 0))
                dst.paste(img_b, (w_b, 0))
                dst.save(os.path.join(new_dir, real_name + ext))
            else:
                print('img height error')
    os.system('ls %s | wc -l' % test_dir)
    os.system('ls %s | wc -l' % new_dir)


else:
    'build symlink'
    data_root = '/home/synan/dataset/VCIP/aligned'
    tr_og = os.path.join(data_root, 'train')
    ev_og = os.path.join(data_root, 'eval')
    te_og = os.path.join(data_root, 'test')

    new_root = os.path.abspath('datasets/VCIP')
    tr_new = os.path.join(new_root, 'train')
    ev_new = os.path.join(new_root, 'eval')
    te_new = os.path.join(new_root, 'test')

    clear_folder(tr_new)
    clear_folder(ev_new)
    clear_folder(te_new)

    def build_symlink(old_dir, new_dir):
        for file in os.listdir(old_dir):
            old_file = os.path.join(old_dir, file)
            new_file = os.path.join(new_dir, file)
            os.symlink(old_file, new_file)
        os.system('ls %s | wc -l' % new_dir)

    build_symlink(tr_og, tr_new)
    build_symlink(te_og, te_new)
    build_symlink(ev_og, ev_new)









