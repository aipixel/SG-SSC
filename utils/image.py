import numpy as np
from torch import Tensor


def get_class_colors40():
    class_colors = np.array([
        [0.66, 0.66, 1],  # 1 wall
        [0, .7, 0],  # 2 floor
        [.8, .5, .1],  # 3 cabinet
        [1, 0.490452, 0.0624932],  # 4 bed
        [1, 1, 0.0392201],  # 5 chair
        [0.657877, 0.0505005, 1],  # 6 sofa
        [0.0363214, 0.0959549, 0.6],  # 7 table
        [1, 0.241096, 0.718126],  # 8 door
        [0.8, 0.8, 1],  # 9 window
        [.7, .45, .9],  # 10 bookshelf
        [.95, 0.3, 0.65],  # 11 picture
        [.65, .4, .85],  # 12 counter
        [0.7, 0.7, .9],  # 13 blinds
        [0.03, 0.09, 0.55],  # 14 desk
        [.6, .35, .8],  # 15 shelves
        [0.65, 0.65, .85],  # 16 curtain
        [.55, .3, .75],  # 17 dresser
        [.9, 0.45, 0.06],  # 18 pillow
        [.9, 0.25, 0.6],  # 19 mirror
        [.1, .6, .1],  # 20 floor mat
        [.8, 0.25, 0.55],  # 21 clothes
        [0, 1, 0],  # 22 ceiling
        [.75, 0.25, 0.50],  # 23 books
        [.7, 0.2, 0.45],  # 24 refridgerator
        [0.316852, 0.548847, 0.186899],  # 25 tvs
        [.65, 0.1, 0.35],  # 26 paper
        [.6, 0.1, 0.3],  # 27 towel
        [0.6, 0.6, .8],  # 28 shower curtain
        [.55, 0.1, 0.25],  # 29 box
        [.5, 0.05, 0.2],  # 30 whiteboard
        [.5, .5, .5],  # 31 person
        [.5, .25, .7],  # 32 night stand
        [.7, .7, .7],  # 33 toilet
        [.6, .6, .6],  # 34 sink
        [.9, .9, .8],  # 35 lamp
        [.7, .7, .7],  # 36 bathtub
        [.55, 0.1, 0.25],  # 37 bag
        [0, .6, 0],  # 38 other structure
        [.45, .2, .65],  # 39 otherfurniture
        [.5, 0.05, 0.2],  # 40 otherprop
        [0, 0, 0],  # 0 empty
    ])
    return (np.array(class_colors)*255).astype(np.uint8)

def get_class_colors():
    class_colors = np.array([
        [0, 1, 0],  # 0 ceiling
        [0, .7, 0],  # 1 floor
        [0.66, 0.66, 1],  # 2 wall
        [0.8, 0.8, 1],  # 3 window
        [1, 1, 0.0392201],  # 4 chair
        [1, 0.490452, 0.0624932],  # 5 bed
        [0.657877, 0.0505005, 1],  # 6 sofa
        [0.0363214, 0.0959549, 0.6],  # 7 table
        [0.316852, 0.548847, 0.186899],  # 8 tvs
        [.8, .5, .1],  # 9 furn
        [1, 0.241096, 0.718126],  # 10 objs
        [0, 0, 0],  # 11 empty
    ])
    return (np.array(class_colors)*255).astype(np.uint8)

def decode_input(input: Tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im_batch = input.transpose(1,3) # NCHW => NWHC
    im_batch = im_batch.transpose(1,2) # NWHC => NHWC
    im_batch = ((im_batch.cpu().numpy()*std + mean) * 255).astype(np.uint8)

    return im_batch

def decode_extra(input: Tensor):
    #mean = [0.485, 0.456, 0.406]
    #std = [0.229, 0.224, 0.225]

    im_batch = input.transpose(1,3) # NCHW => NWHC
    im_batch = im_batch.transpose(1,2) # NWHC => NHWC
    im_batch = (im_batch.cpu().numpy() * 255).astype(np.uint8)

    return im_batch

def decode_labels(input : Tensor, labels: Tensor):

    input_im = decode_input(input)

    class_colors = get_class_colors()

    labels_np = labels.cpu().numpy()

    labels_im = class_colors[labels_np]

    return (input_im * .3 + labels_im * .7).astype(np.uint8)


def decode_outputs(input: Tensor, output : Tensor, labels: Tensor):

    input_im = decode_input(input)

    class_colors = get_class_colors()

    out_im = output.transpose(1,3) # NCHW => NWHC
    out_im = out_im.transpose(1,2) # NWHC => NHWC
    out_im = out_im.cpu().detach().numpy()
    out_im = np.argmax(out_im,axis=-1)

    labels_np = labels.cpu().numpy()

    out_im[labels_np == 11] = 11

    out_im = class_colors[out_im]

    return (input_im * .3 + out_im * .7).astype(np.uint8)

