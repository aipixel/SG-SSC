import ctypes
import numpy as np
import os
from skimage import io
from cuda.preproc3d import get_segmentation_class_map

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src/lib_preproc.so'))

_lib.GetNormals.argtypes = (ctypes.c_char_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p,
                            ctypes.c_void_p
                            )

def get_normals(file_prefix, voxel_shape):
    global _lib

    segmentation_class_map = get_segmentation_class_map()

    vox_origin = np.ones(3, dtype=np.float32)
    cam_pose = np.ones(16, dtype=np.float32)
    vox_size = np.array([voxel_shape[0], voxel_shape[1], voxel_shape[2]], dtype=np.int32)

    depth_image = io.imread('{}.png'.format(file_prefix))  # , as_gray=True)

    #print(depth_image.shape)
    #print(np.unique(depth_image))


    normals = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
    labels_2d = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.uint8)
    xyz = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)

    #print(file_prefix+'.bin')

    _lib.GetNormals(ctypes.c_char_p(bytes(file_prefix+'.bin', 'utf-8')),
                    cam_pose.ctypes.data_as(ctypes.c_void_p),
                    vox_size.ctypes.data_as(ctypes.c_void_p),
                    vox_origin.ctypes.data_as(ctypes.c_void_p),
                    segmentation_class_map.ctypes.data_as(ctypes.c_void_p),
                    depth_image.ctypes.data_as(ctypes.c_void_p),
                    normals.ctypes.data_as(ctypes.c_void_p),
                    xyz.ctypes.data_as(ctypes.c_void_p),
                    labels_2d.ctypes.data_as(ctypes.c_void_p)
                    )
    return normals, labels_2d
