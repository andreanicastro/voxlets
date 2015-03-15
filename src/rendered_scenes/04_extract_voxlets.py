'''
Loads in the combined sboxes and clusters them to form a dictionary
smaller
'''
import numpy as np
import cPickle as pickle
import sys
import os
import yaml
from time import time
import scipy.io

sys.path.append(os.path.expanduser('~/projects/shape_sharing/src/'))
from common import paths
from common import voxel_data
from common import images
from common import parameters
from common import features
from common import carving
from common import voxlets
from common import scene

if not os.path.exists(paths.RenderedData.voxlets_dict_data_path):
    os.makedirs(paths.RenderedData.voxlets_dict_data_path)


def flatten_sbox(sbox):
    return sbox.V.flatten()


def process_sequence(sequence):

    print "Processing " + sequence['name']
    sc = scene.Scene()
    sc.load_sequence(sequence, frame_nos=0, segment_with_gt=True, save_grids=False)
    sc.santity_render(save_folder='/tmp/')

    idxs = sc.im.random_sample_from_mask(
        parameters.VoxletTraining.pca_number_points_from_each_image)

    "Now try to make this nice and like parrallel or something...?"
    t1 = time()
    gt_shoeboxes = [sc.extract_single_voxlet(
        idx, extract_from='gt_tsdf', post_transform=flatten_sbox) for idx in idxs]
    view_shoeboxes = [sc.extract_single_voxlet(
        idx, extract_from='visible_tsdf', post_transform=flatten_sbox) for idx in idxs]
    print "Took %f s" % (time() - t1)

    np_gt_sboxes = np.array(gt_shoeboxes)
    np_view_sboxes = np.array(view_shoeboxes)

    print "Shoeboxes are shape " + str(np_gt_sboxes.shape)
    print "Features are shape " + str(np_view_sboxes.shape)

    D = dict(shoeboxes=np_gt_sboxes, features=np_view_sboxes)
    savepath = paths.RenderedData.voxlets_dict_data_path + \
        sequence['name'] + '.mat'
    print savepath
    scipy.io.savemat(savepath, D, do_compression=True)


# need to import these *after* the pool helper has been defined
if parameters.multicore:
    import multiprocessing
    import functools
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map


if __name__ == '__main__':

    tic = time()
    mapper(process_sequence, paths.RenderedData.train_sequence())
    print "In total took %f s" % (time() - tic)
