import argparse
from utils import path_config

# default settings
GPU = 0
WORKERS = 2
DATASET = "NYU"
PREPROC_PATH = ""
BASE_PATH_TRAIN = ""
BASE_PATH_TEST = ""

# Model settings
WEIGHTS = "none"
INPUT_TYPE = "rgb+normals"
NUM_CLASSES = 11
vox_shape = (240, 144, 240)
vox_shape_down = (60, 36, 60)

def parse_arguments():
    global GPU, WORKERS, DATASET, BASE_PATH_TRAIN, BASE_PATH_TEST, \
           PREPROC_PATH, WEIGHTS, INPUT_TYPE, NUM_CLASSES
           
    print("\nSGNet Preprocessing Script\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Target dataset", type=str, choices=['NYU', 'NYUCAD', 'SemanticKITTI'])
    parser.add_argument("--weights", help="Pretraind 2D weights to preprocess.", default=WEIGHTS,type=str)
    parser.add_argument("--workers", help="Concurrent threads. Default " + str(WORKERS),
                        type=int, default=WORKERS, required=False)
    parser.add_argument("--gpu", help="GPU device. Default " + str(GPU),
                        type=int, default=GPU, required=False)
    parser.add_argument("--input_type",  help="Network input type. Default " + INPUT_TYPE,
                        type=str, default=INPUT_TYPE, required=False,
                        choices=['rgb+normals', 'rgb+depth']
                        )
    args = parser.parse_args()
    DATASET = args.dataset
    WORKERS = args.workers
    GPU = args.gpu
    WEIGHTS = args.weights
    INPUT_TYPE = args.input_type
    
    path_dict = path_config.read_config()
    if DATASET == "NYU":
        BASE_PATH_TRAIN = path_dict["NYU_BASE_TRAIN"]
        BASE_PATH_TEST = path_dict["NYU_BASE_TEST"]
        if INPUT_TYPE == "rgb+depth":
            PREPROC_PATH = path_dict["NYU_RGB_PRIOR_PREPROC"]
        else:
            PREPROC_PATH = path_dict["NYU_RGB_NORMALS_PRIOR_PREPROC"]
    elif DATASET == "NYUCAD":
        BASE_PATH_TRAIN = path_dict["NYUCAD_BASE_TRAIN"]
        BASE_PATH_TEST = path_dict["NYUCAD_BASE_TEST"]
        if INPUT_TYPE == "rgb+depth":
            PREPROC_PATH = path_dict["NYUCAD_RGB_PRIOR_PREPROC"]
        else:
            PREPROC_PATH = path_dict["NYUCAD_RGB_NORMALS_PRIOR_PREPROC"]
    elif DATASET == "SemanticKITTI":
        BASE_PATH_TRAIN = path_dict["KITTI_BASE_TRAIN"]
        BASE_PATH_TEST = path_dict["KITTI_BASE_TEST"]
        if INPUT_TYPE == "rgb+depth":
            PREPROC_PATH = path_dict["SemanticKITTI_RGB_PRIOR_PREPROC"]
        else:
            PREPROC_PATH = path_dict["SemanticKITTI_RGB_NORMALS_PRIOR_PREPROC"]
    else:
        print("Dataset", DATASET, "not supported yet!")
        exit(-1)


def preproc():

    from cuda.preproc3d import lib_preproc_setup, process
    from utils.data import get_file_prefixes_from_path, DL2Dev
    import numpy as np

    from tqdm import tqdm
    import os
    from torch.utils.data import DataLoader
    from utils.data import MultimodalDataset
    from torchvision.transforms import Compose
    from utils.transforms2D import ToTensor, Normalize
    from models.SGFNet import BiModalRDFNetLW
    import torch
    import torch.nn.functional as F
    from utils.cuda import get_device
    from tqdm import tqdm
    from skimage import io
    from utils.image import decode_outputs


    print("Selected device:", "cuda:" + str(GPU))
    dev = get_device("cuda:" + str(GPU))
    torch.cuda.empty_cache()
    
    print("Checking already done...", end=" ", flush=True)
    already_done = [os.path.basename(x) for x in get_file_prefixes_from_path(PREPROC_PATH, criteria="*.npz")]
    print(len(already_done))
    base_path = {
            'train': BASE_PATH_TRAIN,
            'valid': BASE_PATH_TEST
    }
    print("Checking files to process...", end=" ", flush=True)
    prefixes = {
         'train': [x for x in get_file_prefixes_from_path(base_path['train'])
                   if os.path.basename(x) not in already_done
                   ],
         'valid': [x for x in get_file_prefixes_from_path(base_path['valid'])
                   if os.path.basename(x) not in already_done
                   ]
    }
    print("train({}) - valid({})".format(len(prefixes['train']), len(prefixes['valid'])))
    
    transforms = Compose([ToTensor(),
                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                          ])
    if INPUT_TYPE == 'rgb+normals':
        train_ds = MultimodalDataset(prefixes['train'], transf=transforms, read_normals=True, read_xyz=False)
        valid_ds = MultimodalDataset(prefixes['valid'], transf=transforms, read_normals=True, read_xyz=False)
        model = BiModalRDFNetLW(num_classes=NUM_CLASSES)
    else:
        print("INPUT_TYPE", INPUT_TYPE, "not supported yet!")
        exit(-1)
    dataloaders = {
        'train': DL2Dev(DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=WORKERS), dev),
        'valid': DL2Dev(DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=WORKERS), dev)
    }
    
    model.load_state_dict(torch.load(os.path.join("weights", WEIGHTS),map_location=dev))
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(dev)
    model = model.eval()
    v_unit = 0.02
    floor_high = 4.0 if DATASET == "NYU" else 0.0
    lib_preproc_setup(device=GPU, num_threads=512, K=None, frame_shape=(640, 480), v_unit=v_unit,
                      v_margin=0.24, floor_high=floor_high, debug=0)
    for dataset in ['valid', 'train']:
        with tqdm(total=len(prefixes[dataset]), desc="") as pbar, \
             open("bad_bin_files.txt", "a") as f_bad, \
             open("zero_bin_files.txt","a") as f_zero:
            for prefix, (inputs, labels) in zip(prefixes[dataset], dataloaders[dataset]):
                basename = os.path.basename(prefix)
                pbar.set_description(prefix)
                pbar.update()
                preproc_dir = os.path.join(PREPROC_PATH, dataset)
                os.makedirs(preproc_dir,exist_ok=True)
                preproc_file = os.path.join(preproc_dir, basename+".npz")
                
                pred = model(inputs[0], inputs[1])
                outputs = pred[-1]
                outputs = torch.nn.Softmax(dim=1)(outputs)
                
                out_h, out_w = 480,640
                in_h, in_w = 468,624
                top, left = (out_h - in_h) // 2, (out_w - in_w) // 2
                outputs = outputs[:, :, top:top + in_h, left:left + in_w].contiguous()
                labels = labels[:, top:top + in_h, left:left + in_w].contiguous()
                input_rgb = inputs[0][:, :, top:top + in_h, left:left + in_w].contiguous()
                print("pred_rgb path: ",prefix + "_color.jpg")
                pred_rgb = io.imread(prefix + "_color.jpg")
                out_h, out_w = pred_rgb.shape[:-1]
                pred_rgb[top:top + in_h, left:left + in_w:] = decode_outputs(input_rgb, outputs, labels)[0]
                pred_rgb_file = prefix + "_pred2D_" + INPUT_TYPE + ".png"
                io.imsave(pred_rgb_file, pred_rgb)
                
                pred_data = outputs.transpose(1, 3)  # NCHW => NWHC
                pred_data = pred_data.transpose(1, 2).cpu().detach().numpy()[0]  # NWHC => NHWC
                pred_out = np.zeros((480, 640, NUM_CLASSES), np.float32)
                pred_out[top:top + in_h, left:left + in_w:] = pred_data
                #pred_out is 2d label onehot(480, 640, 11)
                #vox_shape = (240, 144, 240)
                cam_pose, vox_origin, vox_grid, vox_tsdf, vox_prior, segmentation_label, vox_weights, depth_map, vox_prior_full, segmentation_label_full = process(prefix, pred_out, vox_shape)
                if np.sum(vox_grid)==0:
                    f_zero.write(basename+'\n')
                    f_zero.flush()
                    continue
                if np.sum(vox_prior)==0:
                    print("Error in preprocess!!!", basename)
                    f_bad.write(basename + '\n')
                    f_bad.flush()
                    continue
                np.savez_compressed(preproc_file, vox_tsdf=vox_tsdf, vox_prior=vox_prior, gt=segmentation_label, vox_weights=vox_weights,
                                    position=depth_map, vox_prior_full=vox_prior_full, cam_pose=cam_pose, vox_origin=vox_origin)
# Main Function
def main():
    parse_arguments()
    preproc()
    
if __name__ == '__main__':
  main()
