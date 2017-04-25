import numpy as np
import real_data_paths as paths
import system_setup
import cPickle as pickle
import os, sys
import yaml
from time import time

sys.path.append('..')
from common import images
from common import voxel_data
from common import carving

parameters_path = './training_params.yaml'
parameters = yaml.load(open(parameters_path))

def get_scene_pose(scene):
    with open(scene + '/scene_pose.yaml') as f:
        return yaml.load(f)

folder_convert = {'data': 'fold_0', 'data1': 'fold_1', 'data2': 'fold_2'}
inv_folder_convert = {y:x for x, y in folder_convert.iteritems()}
inv_folder_convert['fold_01'] = 'data1'
newsavepath = '/home/michael/Dropbox/Public/voxlets/new_carved/'

# doing a loop here to loop over all possible files...
def process_sequence(sequence):
    
    print "Considering", sequence['folder']
    
    # hack to convert between old and new folder names
    old_folder = inv_folder_convert[sequence['folder'].split('/')[-2]]
    newpath = ('/media/michael/Seagate/phd_projects/volume_completion_data'
                '/data/oisin_house/' + old_folder + '/')
    sequence['folder'] = newpath
    
    scene = sequence['folder'] + sequence['scene']
    new_folder = folder_convert[sequence['folder'].split('/')[-2]]

    this_dir = newsavepath + new_folder + '/' + sequence['scene']
    
    def forcemkdir(pth):
        if not os.path.exists(pth):
            os.makedirs(pth)
    
    forcemkdir(newsavepath + new_folder)
    forcemkdir(this_dir)
        
    if os.path.exists(this_dir + '/ground_truth_tsdf.dat'):
        print "Skipping - ", this_dir
        return

    # # ignore if the output file exists...
    # if os.path.exists(scene + '/ground_truth_tsdf.pkl'):
    #     return

    print "Processing ", scene
    vid = images.RGBDVideo()
    vid.load_from_yaml(scene + '/poses.yaml')
    print "WARNING" * 20
    vid.frames = vid.frames

    # load the scene parameters...
    scene_pose = get_scene_pose(scene)
    vgrid_size = np.array(scene_pose['size'])
    voxel_size = parameters['voxel_size']
    vgrid_shape = vgrid_size / voxel_size

    # initialise voxel grid (could add helper function to make it explicit...?)
    vox = voxel_data.WorldVoxels()
    vox.V = np.zeros(vgrid_shape, np.uint8)
    vox.set_voxel_size(voxel_size)
    vox.set_origin(np.array([0, 0, 0]))

    print "Performing voxel carving", scene
    carver = carving.Fusion()
    carver.set_video(vid)
    carver.set_voxel_grid(vox)
    vox, visible = carver.fuse(mu=parameters['mu'], filtering=False, measure_in_frustrum=True)
    in_frustrum = carver.in_frustrum

    print "Saving...", scene
    print vox.V.dtype
    print visible.V.dtype
    print in_frustrum.V.dtype

    print this_dir + '/ground_truth_tsdf.dat'
    with open(this_dir + '/ground_truth_tsdf.dat', 'w') as f:
        vox.V.astype(np.float16).tofile(f)
        # pickle.dump(vox, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(this_dir + '/in_frustrum.dat', 'w') as f:
        (in_frustrum.V > 0).astype(np.uint8).tofile(f)

    with open(this_dir + '/weights.dat', 'w') as f:
        carver.accum.weights.astype(np.float16).tofile(f)

    # with open(this_dir + '/gr.dat', 'w') as f:
    #     (in_frustrum.V > 0).astype(np.uint8).tofile(f)

    # with open(scene + '/ground_truth_tsdf.pkl', 'w') as f:
    #     pickle.dump(vox, f, protocol=pickle.HIGHEST_PROTOCOL)
    # #
    # # with open(scene + '/visible.pkl', 'w') as f:
    # #     pickle.dump(visible, f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open(scene + '/in_frustrum.pkl', 'w') as f:
    #     pickle.dump(in_frustrum, f, protocol=pickle.HIGHEST_PROTOCOL)



# need to import these *after* the pool helper has been defined
if system_setup.multicore:
    import multiprocessing
    import functools
    pool = multiprocessing.Pool(system_setup.cores)
    mapper = pool.map
else:
    mapper = map

tic = time()
for scn in paths.all_train_data + paths.test_data:
    process_sequence(scn)

#process_sequence(paths.all_train_data[1])
#
# for scene in [{'scene': 'saved2_00186', 'folder': '/media/michael/Seagate/phd_projects/volume_completion_data/data/oisin_house/data2/'}]:
#     process_sequence(scene)

print "In total took %f s" % (time() - tic)
