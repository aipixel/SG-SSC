import argparse
from utils import path_config

# default settings
GPU = 0

# Dataloader settings
BATCH_SIZE = 4
WORKERS = 2
DATASET = "NYU"
PREPROC_PATH = ""

# Model settings
WEIGHTS = "none"
BATCH_NORM = True
INPUT_TYPE = "rgb+normals"


def parse_arguments():
    global GPU, BATCH_SIZE, WORKERS, DATASET, PREPROC_PATH,\
           WEIGHTS, BATCH_NORM, INPUT_TYPE

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Target dataset", type=str, choices=['NYU', 'NYUCAD', 'SemanticKITTI'])
    parser.add_argument("--batch_size", help="Training batch size. Default: "+str(BATCH_SIZE),
                        type=int, default=BATCH_SIZE, required=False)
    parser.add_argument("--weights", help="Pretraind weights. ", type=str)
    parser.add_argument("--workers", help="Concurrent threads. Default " + str(WORKERS),
                        type=int, default=WORKERS, required=False)
    parser.add_argument("--gpu", help="GPU device. Default " + str(GPU),
                        type=int, default=GPU, required=False)
    parser.add_argument("--input_type",  help="Network input type. Default " + INPUT_TYPE,
                        type=str, default=INPUT_TYPE, required=False,
                        choices=['rgb+normals', 'rgb+depth', 'depth']
                        )
    parser.add_argument("--bn",  help="Apply batch normalization? Default yes",
                        type=str, default="yes", required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    args = parser.parse_args()

    DATASET = args.dataset
    BATCH_SIZE = args.batch_size
    WORKERS = args.workers
    GPU = args.gpu
    WEIGHTS = args.weights
    INPUT_TYPE = args.input_type
    BATCH_NORM = args.bn in ['yes', 'Yes', 'y', 'Y']

    path_dict = path_config.read_config()
    if DATASET == "NYU":
        if INPUT_TYPE == "rgb+depth":
            PREPROC_PATH = path_dict["NYU_RGB_PRIOR_PREPROC"]
        else:
            PREPROC_PATH = path_dict["NYU_RGB_NORMALS_PRIOR_PREPROC"]
    elif DATASET == "NYUCAD":
        if INPUT_TYPE == "rgb+depth":
            PREPROC_PATH = path_dict["NYUCAD_RGB_PRIOR_PREPROC"]
        else:
            PREPROC_PATH = path_dict["NYUCAD_RGB_NORMALS_PRIOR_PREPROC"]
    elif DATASET == "SemanticKITTI":
        if INPUT_TYPE == "rgb+depth":
            PREPROC_PATH = path_dict["SemanticKITTI_RGB_PRIOR_PREPROC"]
        else:
            PREPROC_PATH = path_dict["SemanticKITTI_RGB_NORMALS_PRIOR_PREPROC"]
    else:
        print("Dataset", DATASET, "not supported yet!")
        exit(-1)

def eval():
    from tqdm import tqdm
    import os

    from utils.data import SSCMultimodalDataset, sample2dev
    from utils.data import get_file_prefixes_from_path
    from torch.utils.data import DataLoader
    import torch
    from utils.cuda import get_device
    import numpy as np
    from utils.metrics import MIoU, CompletionIoU
    from models.SGNet import get_model

    nyu_classes = ["ceil", "floor", "wall", "window", "chair", "bed", "sofa", "table", "tvs", "furniture", "objects",
                   "empty"]

    print("Selected device:", "cuda:" + str(GPU))
    dev = get_device("cuda:" + str(GPU))
    torch.cuda.empty_cache()

    valid_prefixes = get_file_prefixes_from_path(os.path.join(PREPROC_PATH, "valid"), criteria="*.npz")
    print('PREPROC_PATH:',PREPROC_PATH)
    valid_ds = SSCMultimodalDataset(valid_prefixes)
    dataloader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    miou = MIoU(num_classes=12, ignore_class=0)
    ciou = CompletionIoU()

    model = get_model(input_type=INPUT_TYPE, batch_norm=BATCH_NORM, inst_norm=False)
    print("loading", WEIGHTS)
    model.load_state_dict(torch.load(os.path.join("weights", WEIGHTS),map_location=dev))

    model.to(dev)
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="") as pbar:
            for sample, prefix in zip(dataloader, valid_prefixes):
                sample = sample2dev(sample, dev)

                vox_tsdf = sample['vox_tsdf']
                gt = sample['gt']
                vox_prior = sample['vox_prior']
                vox_weights = sample['vox_weights']

                basename = os.path.basename(prefix)
                pred_sc, pred_ssc, = model(vox_tsdf, vox_prior)

                miou.update(pred_ssc, gt, vox_weights)
                ciou.update(pred_sc, gt, vox_weights)
                
                pbar.set_description('Test miou:{:5.1f}'.format(miou.compute() * 100))
                pbar.update()

    comp_iou, precision, recall = ciou.compute()

    print("prec rec. IoU  MIou")
    print("{:4.1f} {:4.1f} {:4.1f} {:4.1f}".format(100 * precision, 100 * recall, 100 * comp_iou,
                                                   miou.compute() * 100))

    per_class_iou = miou.per_class_iou()
    for i in range(len(per_class_iou)):
        text = '{:12.12}: {:5.1f}'.format(nyu_classes[i], 100 * per_class_iou[i])
        print(text, end="        ")
        if i % 4 == 3:
            print()

    print("\nLatex Line:")
    print("{:4.1f} & {:4.1f} & {:4.1f} &".format(100 * precision, 100 * recall, 100 * comp_iou), end=" ")
    for i in range(len(per_class_iou)):
        text = '{:4.1f} &'.format(100 * per_class_iou[i])
        print(text, end=" ")

    print("{:4.1f} \\\\".format(miou.compute() * 100))


# Main Function
def main():
    parse_arguments()
    eval()


if __name__ == '__main__':
  main()

