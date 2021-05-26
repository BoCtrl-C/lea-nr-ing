from modules.bouncing_balls import BouncingBalls
from modules.building import build_ARTransformerNum, count_parameters
from modules.utils import quantize

import argparse
import os
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


#----------
# Constants
#----------

# dataset
DATASET_ROOT = 'datasets/non-uniform_dots'
BATCH_SIZE = 16

# model
SZ = 'm'
ALLW_NUMS = range(1, 9)

# optimization
MAX_EPOCHS = 100
STEP_SIZE = 25
GAMMA = 0.1
PATIENCE = 5
DELTA = 1e-3
EXPERIMENT_STR = 'org'
CHECKPOINT_DIR = 'checkpoints'
METADATA_DIR = 'metadata'


#----------
# Functions
#----------

def transform(frames):
    """The function quantizes and flattens the input frames.
    """

    return quantize(frames.squeeze())\
        .view(frames.shape[0], - 1)\
        .T\
        .contiguous()\

def name_experiment(sz, lr, exp_str):
    """Depending on the provided parameters, the function returns a unique name
    for the experiment.
    """

    return 'artrnnum_' + sz + '_lr-' + str(lr) + '_' + exp_str


#--------
# Classes
#--------

class EarlyStopping:
    """The class implements a patience-based stopping criterion. The delta
    parameter represents the minimum change in the monitored loss qualified as
    an improvement.
    """

    def __init__(self, patience=1, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.count = 0
        self.best_loss = float('inf')

    def stop(self, loss):
        if loss < self.best_loss - self.delta:
            self.count = 0
            self.best_loss = loss
        else:
            self.count += 1
        
        # stop?
        return True if self.count == self.patience else False


#-----
# Main
#-----

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
    args = parser.parse_args()

    train_set = BouncingBalls(
        DATASET_ROOT,
        split='training',
        num=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_set = BouncingBalls(
        DATASET_ROOT,
        split='validation',
        num=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE
    )

    model = build_ARTransformerNum(
        sz=SZ,
        num_num=max(ALLW_NUMS) + 1,
        num_positions=train_set.res**2
    )
    print(
        'ARTransformerNum (size \'', SZ,
        '\', ', count_parameters(model), ' parameters)',
        sep=''
    )

    device = torch.device('cuda:0')
    model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    early_stopping = EarlyStopping(patience=PATIENCE, delta=DELTA)

    experiment_name = name_experiment(SZ, args.lr, EXPERIMENT_STR)
    
    # metadata
    epoch = 0
    train_losses = []
    val_losses = []
    
    exit = False
    while not exit and epoch < MAX_EPOCHS:
        epoch += 1
        print('=> Epoch:', epoch)

        print('    Training steps:')
        model.train()
        cum_loss = 0
        pbar = tqdm(train_loader)
        for batch in pbar:
            numerosities, frames = batch

            # preprocessing
            numerosities = numerosities.to(device)
            frames = transform(frames).to(device)
            
            # forward, backward and optimize
            optimizer.zero_grad()
            logits = model(numerosities, frames)
            loss = criterion(
                logits.view(- 1, logits.shape[-1]),
                frames.view(- 1)
            )
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            pbar.set_description('    Loss: %.3f' % loss.item())

        epoch_loss = cum_loss/len(train_loader)
        train_losses.append(epoch_loss)
        print('    Epoch training loss: %.3f' % epoch_loss)

        scheduler.step()

        print('    Validation steps:')
        model.eval()
        cum_loss = 0
        with torch.no_grad():
            pbar = tqdm(val_loader)
            for batch in pbar:
                numerosities, frames = batch

                # preprocessing
                numerosities = numerosities.to(device)
                frames = transform(frames).to(device)

                # forward
                logits = model(numerosities, frames)
                loss = criterion(
                    logits.view(- 1, logits.shape[-1]),
                    frames.view(- 1)
                )

                cum_loss += loss.item()
                pbar.set_description('    Loss: %.3f' % loss.item())

        epoch_loss = cum_loss/len(val_loader)
        val_losses.append(epoch_loss)
        print('    Epoch validation loss: %.3f' % epoch_loss)

        if early_stopping.stop(epoch_loss):
            exit = True
            print('Stopping criterion satisfied')

        if early_stopping.count == 0:
            # save the best checkpoint
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict()},
                os.path.join(CHECKPOINT_DIR, experiment_name)
            )

    # save training metadata
    torch.save(
        {'train_losses': train_losses, 'val_losses': val_losses},
        os.path.join(METADATA_DIR, experiment_name)
    )

    if epoch == MAX_EPOCHS:
        print('No convergence')