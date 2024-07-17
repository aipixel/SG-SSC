import torch
import time
from tqdm import tqdm
from utils.metrics import Accuracy, MIoU, CompletionIoU
from torch.utils.tensorboard import SummaryWriter
from utils.misc import get_run
from utils.data import sample2dev
import torch.nn.functional as F
from torch import optim

nyu_classes = ["ceil", "floor", "wall", "window", "chair", "bed", "sofa", "table", "tvs", "furniture", "objects",
               "empty"]

def train_2d(model, device, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, patience=50, suffix=None):
    run = get_run()
    time.sleep(.1)

    if suffix is None:
        model_name = "R{}_{}".format(run, type(model).__name__)
    else:
        model_name = "R{}_{}_{}".format(run, type(model).__name__, suffix)
    print("Training model", model_name)
    
    #tb_writer = SummaryWriter(log_dir="log/{}".format(model_name))
    #tb_writer.add_text("Optimizer", str(optimizer))
    #tb_writer.add_text("Criterion", str(criterion))

    since = time.time()
    best_miou = 0.0
    waiting = 0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            tqdm_desc = "{}: Epoch {}/{} Loss: {:.4f} Acc: {:.4f} MIoU: {:.4f} Lr: {:.8f}"
            num_samples = 0
            m_acc = Accuracy()
            m_miou = MIoU()

            with tqdm(total=len(dataloaders[phase]), desc="") as pbar:
                for inputs, labels in dataloaders[phase]:
                    num_samples += labels.size(0)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        if len(inputs) == 1:
                            outputs = model(inputs[0])
                        elif len(inputs) == 2:
                            outputs = model(inputs[0], inputs[1])
                        elif len(inputs) == 3:
                            outputs = model(inputs[0], inputs[1], inputs[2])

                        preds = outputs[-1]
                        if phase == 'valid':
                            out_h, out_w = 480,640
                            in_h, in_w = 468,624
                            top, left = (out_h - in_h) // 2, (out_w - in_w) // 2
                            preds = preds[:, :, top:top + in_h, left:left + in_w].contiguous()
                            labels = labels[:, top:top + in_h, left:left + in_w].contiguous()
                            
                        loss = criterion(outputs, labels)
                        m_acc.update(preds, labels)
                        m_miou.update(preds, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            if type(scheduler) is optim.lr_scheduler.OneCycleLR:
                                scheduler.step()

                    # statistics
                    l = loss.item()
                    running_loss += l * labels.size(0)
                    pbar.set_description(tqdm_desc.format(phase, epoch + 1, num_epochs,
                                                          running_loss / num_samples,
                                                          m_acc.compute(),
                                                          m_miou.compute(),
                                                          optimizer.param_groups[0]['lr']
                                                          ))
                    pbar.update()
            epoch_loss = running_loss / num_samples
            epoch_acc = m_acc.compute()
            epoch_miou = m_miou.compute()
            epoch_per_class_iou = m_miou.per_class_iou()

            #tb_writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
            #tb_writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch)
            #tb_writer.add_scalar('mIoU/{}'.format(phase), epoch_miou, epoch)

            if phase == 'train':
                if type(scheduler) is not optim.lr_scheduler.OneCycleLR:
                    scheduler.step()

            # deep copy the model
            if phase == 'valid' and epoch_miou > best_miou:
                waiting = 0
                print("mIoU improved from {:.5f} to  {:.5f}".format(best_miou,epoch_miou))
                best_epoch = epoch
                best_miou = epoch_miou
                best_per_class_iou = epoch_per_class_iou
                torch.save(model.state_dict(), "weights/{}_EPOCH_{}".format(model_name, epoch))
            elif phase == 'valid':
                print("mIoU {:.5f} was not an improvement from {:.5f}".format(epoch_miou,best_miou))
                waiting += 1
            if phase == 'valid':
                tb_text = ""
                for i in range(11):
                    text = '{:12.12}: {:5.1f}'.format(nyu_classes[i], 100 * epoch_per_class_iou[i])
                    tb_text += text
                    print(text, end="        ")
                    if i % 4 == 3:
                        print()
                #tb_writer.add_text("Per Class IoU", tb_text, global_step=epoch)
                print()
                time.sleep(.5)
        torch.cuda.empty_cache()
        if waiting > patience:
            print("out of patience!!!")
            break
            
    torch.save(model.state_dict(), "weights/{}_EPOCH_{}".format(model_name, epoch))
    time_elapsed = time.time() - since
    print(model_name)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val MIoU%: {:6.1f}  Epoch: {}'.format(100 * best_miou, best_epoch))

    tb_text = ""
    for i in range(11):
        text = '{:12.12}: {:5.1f}'.format(nyu_classes[i], 100 * best_per_class_iou[i])
        tb_text += text
        print(text, end="        ")
        if i % 4 == 3:
            print()
    #tb_writer.add_text("Best Per Class IoU", tb_text)
    print()
    return model
    
    
def train_3d(model, dev, dataloaders, criterion1, criterion2, optimizer,  scheduler=None, num_epochs=25, patience=50, suffix=None):
    run = get_run()
    time.sleep(.1)

    if suffix is None:
        model_name = "R{}_{}".format(run, type(model).__name__)
    else:
        model_name = "R{}_{}_{}".format(run, type(model).__name__, suffix)
    print("Training model", model_name)

    #tb_writer = SummaryWriter(log_dir="log/{}".format(model_name))
    #tb_writer.add_text("Optimizer", str(optimizer))
    #tb_writer.add_text("Criterion", str(criterion))

    since = time.time()
    best_miou = 0.0
    waiting = 0
    
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            # Iterate over data.
            tqdm_desc = "{}: Epoch {}/{} Loss: {:.4f} Loss1: {:.4f} Loss2: {:.4f} MIoU: {:.4f} Lr: {:.8f}"
            num_samples = 0
            m_miou = MIoU(num_classes=12, ignore_class=0)
            ciou = CompletionIoU()

            with tqdm(total=len(dataloaders[phase]), desc="") as pbar:
                for batch_num, sample in enumerate(dataloaders[phase]):
                    sample = sample2dev(sample, dev)

                    vox_tsdf = sample['vox_tsdf']
                    vox_prior = sample['vox_prior']
                    gt = sample['gt']
                    gt_sc = sample['gt'].clone()
                    gt_sc[gt_sc>0] = 1.0
                    #print('gt_sc.shape:',gt_sc.shape,gt_sc.max(),gt_sc.min())
                    #print('gt.shape:',gt.shape,gt.max(),gt.min())
                    vox_weights = sample['vox_weights']

                    if torch.max(gt)>11:
                        print("maior")
                        continue
                    if torch.min(gt)<0:
                        print("menor")
                        continue

                    num_samples += gt.size(0)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs1, outputs2 = model(vox_tsdf, vox_prior)
                        loss1 = criterion1(outputs1, gt_sc, vox_weights)
                        loss2 = criterion2(outputs2, gt, vox_weights)
                        loss = loss1 + loss2
                        
                        m_miou.update(outputs2, gt, vox_weights)
                        ciou.update(outputs1, gt, vox_weights)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            if type(scheduler) is optim.lr_scheduler.OneCycleLR:
                                scheduler.step()
                    # statistics
                    l = loss.item()
                    running_loss += l * gt.size(0)
                    pbar.set_description(tqdm_desc.format(phase, epoch + 1, num_epochs,
                                                          running_loss / num_samples,
                                                          loss1.item(),
                                                          loss2.item(),
                                                          m_miou.compute(),
                                                          optimizer.param_groups[0]['lr']
                                                          ))
                    pbar.update()
            epoch_loss = running_loss / num_samples
            epoch_miou = m_miou.compute()
            epoch_per_class_iou = m_miou.per_class_iou()
            comp_iou, precision, recall = ciou.compute()
            #tb_writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
            #tb_writer.add_scalar('mIoU/{}'.format(phase), epoch_miou, epoch)

            if phase == 'train':
                if type(scheduler) is not optim.lr_scheduler.OneCycleLR:
                    scheduler.step()
                    
            # deep copy the model
            if phase == 'valid' and epoch_miou > best_miou:
                waiting = 0
                print("mIoU improved from {:.5f} to  {:.5f}".format(best_miou,epoch_miou))
                best_epoch = epoch
                best_miou = epoch_miou
                best_per_class_iou = epoch_per_class_iou
                torch.save(model.state_dict(), "weights/{}_EPOCH_{}".format(model_name, epoch))
            elif phase == 'valid':
                print("mIoU {:.5f} was not an improvement from {:.5f}".format(epoch_miou,best_miou))
                waiting += 1
            if phase == 'valid':
                print("prec rec. IoU  MIou")
                print("{:4.1f} {:4.1f} {:4.1f} {:4.1f}".format(100 * precision, 100 * recall, 100 * comp_iou, epoch_miou * 100))
                tb_text = ""
                for i in range(11):
                    text = '{:12.12}: {:5.1f}'.format(nyu_classes[i], 100 * epoch_per_class_iou[i])
                    tb_text += text
                    print(text, end="        ")
                    if i % 4 == 3:
                        print()
                #tb_writer.add_text("Per Class IoU", tb_text, global_step=epoch)
                print()
                time.sleep(.5)
        #torch.cuda.empty_cache()
        if waiting > patience:
            print("out of patience!!!")
            break

    torch.save(model.state_dict(), "weights/{}_EPOCH_{}".format(model_name, epoch))
    time_elapsed = time.time() - since
    print(model_name)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val MIoU%: {:6.1f}  Epoch: {}'.format(100 * best_miou, best_epoch))

    tb_text = ""
    for i in range(11):
        text = '{:12.12}: {:5.1f}'.format(nyu_classes[i], 100 * best_per_class_iou[i])
        tb_text += text
        print(text, end="        ")
        if i % 4 == 3:
            print()
    #tb_writer.add_text("Best Per Class IoU", tb_text)
    print()
    return model
