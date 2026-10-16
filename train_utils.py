import os
import logging
import datetime
import numpy as np
from torch import nn

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def free(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def setup_logger(name=None, output=None, for_file=False):
    # create a logger 

    logger = logging.getLogger(name) 
    logger.setLevel(logging.INFO) 

    # the handler format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S") 

    def beijing(sec):
        # change the time to beijing time
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()

    formatter.converter = beijing
    # create a handler to output to terminal window
    if not for_file:
        ch = logging.StreamHandler() 
        ch.setLevel(logging.DEBUG) 
        ch.setFormatter(formatter) 
        logger.addHandler(ch) 

    # write log file
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")

        # create a handler to write log file
        fh = logging.FileHandler(filename) 
        fh.setLevel(logging.DEBUG) 
        fh.setFormatter(formatter) 
        logger.addHandler(fh)

    return logger

def count_model_prameters(model):
    count = sum([p.data.nelement() for p in model.parameters()])
    return count


def record_image(writer, tag, epoch, image):
    writer.add_image(tag, image, epoch)


def record_images(writer, default_tag, epoch, images):
    for idx, image in enumerate(images):
        writer.add_image('{} {}'.format(default_tag, idx + 1), image, epoch)


def print_scatter(feature_set, loss_set, cycle, step, n_components=2):
    scaler = MinMaxScaler()
    tsne = TSNE(n_components=n_components)

    features = np.concatenate(feature_set, axis=0)
    features = tsne.fit_transform(features)

    loss = np.concatenate(loss_set, axis=0)
    loss = scaler.fit_transform(loss.reshape(-1, 1)).reshape(-1)

    plt.scatter(features[:, 0], features[:, 1], c=loss, s=5, cmap='Reds')

    plt.savefig(f'{cycle}-{step}.png', dpi=300)
