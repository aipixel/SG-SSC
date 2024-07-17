import argparse
from utils import path_config

# default settings
GPU = 0
INPUT_TYPE = "rgb+normals"

# Optimizer settings
BASE_LR = 2e-4
LR_MULTIPLIER = 25.0
DECAY = 1e-5
EPOCHS = 200
NUM_CLASSES = 11
ONE_CYCLE = True

# Dataloader settings
BATCH_SIZE = 4
VAL_BATCH_MULT = 2
WORKERS = 2
DATASET = "NYU"
BASE_PATH_TRAIN = ""
BASE_PATH_TEST = ""

def parse_arguments():
    global GPU, INPUT_TYPE, BASE_LR, LR_MULTIPLIER, DECAY, EPOCHS, BATCH_SIZE, \
           VAL_BATCH_MULT, WORKERS, DATASET, BASE_PATH_TRAIN, BASE_PATH_TEST

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
    parser.add_argument("--epochs", help="How many epochs? Default " + str(EPOCHS),
                        type=int, default=EPOCHS, required=False)
    parser.add_argument("--input_type",  help="Network input type. Default " + INPUT_TYPE,
                        type=str, default=INPUT_TYPE, required=False,
                        choices=['rgb+normals', 'rgb+depth', 'depth']
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
    EPOCHS = args.epochs
    INPUT_TYPE = args.input_type
    
    path_dict = path_config.read_config()
    if DATASET == "NYU":
        BASE_PATH_TRAIN = path_dict["NYU_BASE_TRAIN"]
        BASE_PATH_TEST = path_dict["NYU_BASE_TEST"]
    elif DATASET == "NYUCAD":
        BASE_PATH_TRAIN = path_dict["NYUCAD_BASE_TRAIN"]
        BASE_PATH_TEST = path_dict["NYUCAD_BASE_TEST"]
    elif DATASET == "SemanticKITTI":
        BASE_PATH_TRAIN = path_dict["KITTI_BASE_TRAIN"]
        BASE_PATH_TEST = path_dict["KITTI_BASE_TEST"]
    else:
        print("Dataset", DATASET, "not supported yet!")
        exit(-1)
    
def train():

    import os
    import torch
    from torch.utils.data import DataLoader
    from utils.transforms2D import RandomResize, RandomCrop, CenterCrop, HorizontalFlip, ToTensor, Normalize, Pad
    from utils.data import get_file_prefixes_from_path, MultimodalDataset, DL2Dev
    from torchvision.transforms import Compose
    from models.SGFNet import BiModalRDFNetLW
    from torch import optim
    from utils.train_utils import train_2d
    from utils.losses import CustomCrossEntropy, MscCrossEntropyLoss
    from utils.cuda import get_device

    suffix = "{}-{}".format(DATASET, INPUT_TYPE)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    print("Selected device:", "cuda:" + str(GPU))
    dev = get_device("cuda:" + str(GPU))
    torch.cuda.empty_cache()
    
    train_transforms = Compose([
                                CenterCrop((468,624)),
                                RandomResize((0.7,1.8)),
                                Pad((480,640), [0, 0, 0], 0),
                                RandomCrop((480,640)),
                                HorizontalFlip(prob=0.5),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
     
    valid_transforms = Compose([ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    base_path = {
            'train': BASE_PATH_TRAIN,
            'valid': BASE_PATH_TEST
    }
    train_prefixes = get_file_prefixes_from_path(base_path['train'])
    valid_prefixes = get_file_prefixes_from_path(base_path['valid'])
    print("train_prefixes: ",len(train_prefixes))
    print("valid_prefixes: ",len(valid_prefixes))
    
    train_ds = MultimodalDataset(train_prefixes, transf=train_transforms, read_normals=True, read_xyz=False )
    valid_ds = MultimodalDataset(valid_prefixes, transf=valid_transforms, read_normals=True, read_xyz=False )
    
    dataloaders = {
         'train': DL2Dev(DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS), dev),
         'valid': DL2Dev(DataLoader(valid_ds, batch_size=BATCH_SIZE*VAL_BATCH_MULT, shuffle=False, num_workers=WORKERS), dev)
    }
    
    model = BiModalRDFNetLW(num_classes=NUM_CLASSES)
    model.to(dev)
    
    opt = optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=DECAY)
    if ONE_CYCLE:
        sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=BASE_LR*LR_MULTIPLIER,
                                        steps_per_epoch=len(dataloaders['train']), epochs=EPOCHS)
    else:
        sch = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5, last_epoch=-1)
        
    criterion = MscCrossEntropyLoss(weight=None, num_classes = NUM_CLASSES).to(dev)
    
    # Train and evaluate
    model = train_2d(model, dev, dataloaders, criterion, opt, scheduler=sch, num_epochs=EPOCHS, patience=80, suffix=suffix)


# Main Function
def main():
    parse_arguments()
    train()


if __name__ == '__main__':
  main()
