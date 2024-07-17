def _this_file_path():
    import pathlib
    return pathlib.Path(__file__).parent.absolute()


def get_894_11_class_mapping():

    from scipy.io import loadmat
    import os

    m = loadmat(os.path.join(_this_file_path(), 'ClassMapping.mat'))
    classes11 = dict(zip([x[0] for x in m['elevenClass'][0]], range(11)))
    classes40 = dict(zip([x[0] for x in m['nyu40class'][0]], range(40)))
    classes36 = dict(zip([x[0] for x in m['p5d36class'][0]], range(36)))
    m894_40 = [classes40[x[0]] for x in m['mapNYU894To40'][0]]
    m40_36 = [classes36[x[0]] for x in m['mapNYU40to36'][0]]
    m36_11 = [classes11[x[0]] for x in m['map36to11'][0]]

    m894_11 = [m36_11[m40_36[m894_40[x]]] for x in range(894)]

    #print(classes11.keys())
    #map894_11 = [0] + [x + 1 for x in nyu.get_894_11_class_mapping()]

    return [0] + [x + 1 for x in m894_11]


def extract_2d_labels(nyu_path):

    import h5py
    import numpy as np
    import cv2
    import numpy as np

    f = h5py.File('/d02/data/NYU_V2/nyu_depth_v2_labeled.mat', 'r')
    print(f.keys())
    labels = np.array(f['labels'])

    cv2.imwrite("filename.png", np.zeros((10, 10)))

    # dict_keys(['ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furn', 'objs'])

    class_colors = np.flip(np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, .7, 0],
        [0.66, 0.66, 1],
        [0.8, 0.8, 1],
        [1, 1, 0.0392201],
        [1, 0.490452, 0.0624932],
        [0.657877, 0.0505005, 1],
        [0.0363214, 0.0959549, 0.6],
        [0.316852, 0.548847, 0.186899],
        [8, .5, .1],
        [1, 0.241096, 0.718126],
    ]), 1)


    label_image = np.zeros((480, 640, 3), dtype=np.uint8)
    map894_11 = get_894_11_class_mapping()

    def color_mapper(x):
        class_colors[map894_11[x]] * 255
    v_color_mapper = np.vectorize(color_mapper)

    print(labels.shape)

    for label_data in labels[0:0]:

        import time
        start_time = time.time()

        for x in range(640):
            for y in range(480):
                label_image[y, x] = class_colors[map894_11[label_data[x,y]]] * 255

        print("--- %s seconds ---" % (time.time() - start_time))

        cv2.imwrite("filename.png", label_image)

