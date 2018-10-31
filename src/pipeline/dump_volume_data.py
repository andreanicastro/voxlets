
import pickle
import os
import sys
sys.path.append("..")
from common import voxlets

prediction_folder = '/home/an4915/dev/voxlets/data/predictions/cvpr2016_for_release'

if __name__ == '__main__':
    dirs = os.listdir(prediction_folder)

    dirs = [os.path.join(prediction_folder, d) for d in dirs
            if os.path.isdir(os.path.join(prediction_folder, d))]

    for dir in dirs:
        pfile = os.path.join(dir, 'pickles', 'short_and_tall_samples.pkl')

        prediction = pickle.load(open(pfile))

        pickle.dump(prediction.V,
                    open(os.path.join(
                         dir, 'pickles','short_and_tall_samples_volume.pkl'),
                         'wb'))
