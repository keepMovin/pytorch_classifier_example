import os
import random
import shutil

def split_data(dirpath):
    dirs = os.listdir(dirpath)
    destdirs = 'data/test_shapes/'

    if not os.path.exists(destdirs):
        os.makedirs(destdirs)
    for dir in dirs:
        tempdir = os.path.join(dirpath, dir)
        destdir = os.path.join(destdirs, dir)
        if not os.path.exists(destdir):
            os.mkdir(destdir)
        ps = os.listdir(tempdir)
        random.shuffle(ps)
        le = int(len(ps)*0.8)

        for p in ps[le:]:
            shutil.move(os.path.join(tempdir, p), os.path.join(destdir, p))

split_data('data/shapes')
