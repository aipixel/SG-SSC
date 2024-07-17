import ctypes
import numpy as np
import os
from skimage import io
import time

from enum import Enum
class InputType(Enum):
   DEPTH_ONLY = 1
   DEPTH_COLOR = 2
   DEPTH_EDGES = 3

def get_segmentation_class_map():
    return np.array([0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11,
                                   11, 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11], dtype=np.int32)
def get_class_names():
    return ["ceil.", "floor", "wall ", "wind.", "chair", "bed  ", "sofa ", "table", "tvs  ", "furn.", "objs."]


#nvcc --ptxas-options=-v --compiler-options '-fPIC' -o lib_preproc.so --shared lib_preproc.cu

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'src/lib_preproc.so'))

_lib.setup.argtypes = (ctypes.c_int,
              ctypes.c_int,
              ctypes.c_void_p,
              ctypes.c_int,
              ctypes.c_int,
              ctypes.c_float,
              ctypes.c_float)

def lib_preproc_setup(device=0, num_threads=1024, K=None, frame_shape=(640, 480), v_unit=0.02,
                      v_margin=0.24, floor_high=4.0, debug=0):

    global _lib

    frame_width = frame_shape[0]
    frame_height = frame_shape[1]

    if K is None:
        K = np.array([518.8579, 0.0, frame_width / 2.0, 0.0, 518.8579, frame_height / 2.0, 0.0, 0.0, 1.0],dtype=np.float32)

    _lib.setup(ctypes.c_int(device),
                  ctypes.c_int(num_threads),
                  K.ctypes.data_as(ctypes.c_void_p),
                  ctypes.c_int(frame_width),
                  ctypes.c_int(frame_height),
                  ctypes.c_float(v_unit),
                  ctypes.c_float(v_margin),
                  ctypes.c_float(floor_high),
                  ctypes.c_int(debug)
               )




_lib.Process.argtypes = (ctypes.c_char_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p
                              )
def process(file_prefix, prior_data, vox_shape, down_scale = 4):
    global _lib

    #start_time = time.time()

    segmentation_class_map = get_segmentation_class_map()

    vox_origin = np.ones(3,dtype=np.float32)
    cam_pose = np.ones(16,dtype=np.float32)
    vox_size = np.array([vox_shape[0], vox_shape[1], vox_shape[2]], dtype=np.int32)

    vox_shape_down = (vox_shape[0]//4, vox_shape[1]//4, vox_shape[2]//4)
    vox_prior_shape = (vox_shape[0]//4, vox_shape[1]//4, vox_shape[2]//4, 12)

    vox_tsdf = np.zeros(vox_shape, dtype=np.float32)
    vox_grid = np.zeros(vox_shape, dtype=np.uint8)
    vox_prior = np.zeros(vox_prior_shape, dtype=np.float32)

    segmentation_label = np.zeros(vox_shape_down, dtype=np.uint8)
    depth_map = np.zeros((480,640), dtype=np.int32)
    vox_prior_full = np.zeros((240,144,240,12), dtype=np.float32)
    segmentation_label_full = np.zeros((240,144,240), dtype=np.uint8)
    vox_weights = np.zeros(vox_shape_down, dtype=np.float32)

    #print("reproc3d-prepare: %6.4f seconds" % (time.time() - start_time))

    # NYUCAD
    #basename = os.path.basename(file_prefix)
    #npz_file = os.path.join('/d02/data/nyucad/NYUCAD_npz/NYUCADtest_npz',basename+'_voxels.npz')
    #npz = np.load(npz_file)
    #depth_image = npz['depth'].reshape(480,640)

    depth_image = io.imread('{}.png'.format(file_prefix))  # , as_gray=True)
    #print("reproc3d-read depth: %6.4f seconds" % (time.time() - start_time))

    # rgb_image = io.imread('{}_color.jpg'.format(file_prefix))

    #npz_file = '{}_pred2D.npz'.format(file_prefix)

    #loaded_npz = np.load(npz_file)
    #prior_data = loaded_npz['prior2d']
    #print("reproc3d-read prior: %6.4f seconds" % (time.time() - start_time))

    _lib.Process(ctypes.c_char_p(bytes(file_prefix+'.bin','utf-8')),
                      cam_pose.ctypes.data_as(ctypes.c_void_p),
                      vox_size.ctypes.data_as(ctypes.c_void_p),
                      vox_origin.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(down_scale),
                      segmentation_class_map.ctypes.data_as(ctypes.c_void_p),
                      depth_image.ctypes.data_as(ctypes.c_void_p),
                      prior_data.ctypes.data_as(ctypes.c_void_p),
                      vox_grid.ctypes.data_as(ctypes.c_void_p),
                      vox_tsdf.ctypes.data_as(ctypes.c_void_p),
                      vox_prior.ctypes.data_as(ctypes.c_void_p),
                      vox_weights.ctypes.data_as(ctypes.c_void_p),
                      segmentation_label.ctypes.data_as(ctypes.c_void_p),
                      depth_map.ctypes.data_as(ctypes.c_void_p),
                      vox_prior_full.ctypes.data_as(ctypes.c_void_p),
                      segmentation_label_full.ctypes.data_as(ctypes.c_void_p)
                      )
    #print("reproc3d-cuda call: %6.4f seconds" % (time.time() - start_time))

    return cam_pose, vox_origin, vox_grid, vox_tsdf, vox_prior, segmentation_label, vox_weights, depth_map, vox_prior_full, segmentation_label_full



