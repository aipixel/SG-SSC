import argparse
from utils import path_config

# default settings
GPU = 0

# Optimizer settings
BASE_LR = 4e-4
LR_MULTIPLIER = 5.0
DECAY = 1e-5
EPOCHS = 200

# Dataloader settings
BATCH_SIZE = 4
VAL_BATCH_MULT = 1
WORKERS = 2
DATASET = "NYU"
PREPROC_PATH = ""

# Model settings
WEIGHTS = "none"
BATCH_NORM = True
INPUT_TYPE = "rgb+normals"
CLASS_BAL = True
ONE_CYCLE = True
DATA_AUG = True


def parse_arguments():
    global GPU, BASE_LR, LR_MULTIPLIER, DECAY, EPOCHS,\
           BATCH_SIZE, VAL_BATCH_MULT, WORKERS, DATASET, PREPROC_PATH, \
           WEIGHTS, BATCH_NORM, INPUT_TYPE, CLASS_BAL, ONE_CYCLE, DATA_AUG

    print("\nSGNet Training Script\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Target dataset", type=str, choices=['NYU', 'NYUCAD', 'SemanticKITTI'])
    parser.add_argument("--batch_size", help="Training batch size. Default: "+str(BATCH_SIZE),
                        type=int, default=BATCH_SIZE, required=False)
    parser.add_argument("--val_batch_multiplier",  help="Val batch size. Default: "+str(VAL_BATCH_MULT),
                        type=int, default=VAL_BATCH_MULT, required=False)
    parser.add_argument("--base_lr", help="Base LR for One cycle learning. Default " + str(BASE_LR),
                        type=float, default=BASE_LR, required=False)
    parser.add_argument("--lr_multiplier", help="Max LR multiplier. Default " + str(LR_MULTIPLIER),
                        type=float, default=LR_MULTIPLIER, required=False)
    parser.add_argument("--decay", help="Weight decay. Default " + str(DECAY),
                        type=float, default=DECAY, required=False)
    parser.add_argument("--workers", help="Concurrent threads. Default " + str(WORKERS),
                        type=int, default=WORKERS, required=False)
    parser.add_argument("--gpu", help="GPU device. Default " + str(GPU),
                        type=int, default=GPU, required=False)
    parser.add_argument("--weights", help="Pretraind weights. Default " + WEIGHTS,
                        type=str, default=WEIGHTS, required=False)
    parser.add_argument("--epochs", help="How many epochs? Default " + str(EPOCHS),
                        type=int, default=EPOCHS, required=False)
    parser.add_argument("--input_type",  help="Network input type. Default " + INPUT_TYPE,
                        type=str, default=INPUT_TYPE, required=False,
                        choices=['rgb+normals', 'rgb+depth', 'depth']
                        )
    parser.add_argument("--bn",  help="Apply batch bormalization?. Default yes",
                        type=str, default="yes", required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    parser.add_argument("--class_bal",  help="Apply class balancing?. Default yes",
                        type=str, default="yes", required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    parser.add_argument("--one_cycle",  help="Apply OCL?. Default yes",
                        type=str, default="yes", required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    parser.add_argument("--data_aug",  help="Data augmentation?Is it an oracle test?. Default no",
                        type=str, default="yes", required=False,
                        choices=['yes', 'Yes', 'y', 'Y', 'no', 'No', 'n', 'N']
                        )
    args = parser.parse_args()

    DATASET = args.dataset
    BATCH_SIZE = args.batch_size
    VAL_BATCH_MULT = args.val_batch_multiplier
    BASE_LR = args.base_lr
    LR_MULTIPLIER = args.lr_multiplier
    DECAY = args.decay
    WORKERS = args.workers
    GPU = args.gpu
    WEIGHTS = args.weights
    EPOCHS = args.epochs
    INPUT_TYPE = args.input_type
    BATCH_NORM = args.bn in ['yes', 'Yes', 'y', 'Y']
    CLASS_BAL = args.class_bal in ['yes', 'Yes', 'y', 'Y']
    ONE_CYCLE = args.one_cycle in ['yes', 'Yes', 'y', 'Y']
    DATA_AUG = args.data_aug in ['yes', 'Yes', 'y', 'Y']

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
        
def train():

    import os
    from torch.utils.data import DataLoader
    from utils.data import SSCMultimodalDataset, get_file_prefixes_from_path
    from models.SGNet import get_model
    import torch
    from torch import optim
    from utils.train_utils import train_3d
    from utils.losses import SSCCrossEntropy, WeightedSSCCrossEntropy
    from utils.cuda import get_device

    suffix = "{}-{}".format(DATASET, INPUT_TYPE)
    if not BATCH_NORM:
         suffix = suffix + "-nobn"
    if not CLASS_BAL:
         suffix = suffix + "-nocb"
    if DATA_AUG:
        print("3D Data augmentation activated!!!")
        suffix = suffix + "_da"

    print("Selected device:", "cuda:" + str(GPU))
    dev = get_device("cuda:" + str(GPU))
    torch.cuda.empty_cache()

    train_prefixes = get_file_prefixes_from_path(os.path.join(PREPROC_PATH, "train"), criteria="*.npz")
    valid_prefixes = get_file_prefixes_from_path(os.path.join(PREPROC_PATH, "valid"), criteria="*.npz")
    print("Train: ", len(train_prefixes), "Valid: ", len(valid_prefixes))

    train_ds = SSCMultimodalDataset(train_prefixes, data_augmentation=DATA_AUG)
    valid_ds = SSCMultimodalDataset(valid_prefixes)

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS),
        'valid': DataLoader(valid_ds, batch_size=BATCH_SIZE*VAL_BATCH_MULT, shuffle=False, num_workers=WORKERS)
    }

    print("Input type:", INPUT_TYPE)
    model = get_model(input_type=INPUT_TYPE, batch_norm=BATCH_NORM, inst_norm=False)

    if WEIGHTS != "none":
        print("Loading", WEIGHTS, "...")
        model.load_state_dict(torch.load(WEIGHTS),map_location=dev)

    model.to(dev)

    #opt = optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=DECAY)
    opt = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=DECAY, betas=(0.9, 0.999))
    if ONE_CYCLE:
        sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=BASE_LR * LR_MULTIPLIER,
                                        steps_per_epoch=len(dataloaders['train']), epochs=EPOCHS)
    else:
        sch = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5, last_epoch=-1)

    class_weights1 = [1, 1] if CLASS_BAL else [1, 1]
    class_weights2 = [0.01, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2] if CLASS_BAL else [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    criterion1 = WeightedSSCCrossEntropy(weight=torch.Tensor(class_weights1).to(dev), num_classes=2)
    criterion2 = WeightedSSCCrossEntropy(weight=torch.Tensor(class_weights2).to(dev), num_classes=12)

    # Train and evaluate
    model = train_3d(model, dev, dataloaders, criterion1, criterion2, opt,
                        scheduler=sch, num_epochs=EPOCHS, patience=120, suffix=suffix)


# Main Function
def main():
    parse_arguments()
    train()


if __name__ == '__main__':
  main()

