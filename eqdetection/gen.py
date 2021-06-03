import argparse
import os
import random
from datetime import datetime

# PyTorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel

# Misc. Imports
from tqdm import tqdm
import matplotlib.pyplot as plt

# EQDetection Imports
from eqdetection.dataset.stead import STEADDataset
from eqdetection.dataset.impulse import DownscaledImpulse, SpikeImpulse, NoisyImpulse
from eqdetection.genmodel import ParallelModel

# Create a command line argument parser
parser = argparse.ArgumentParser('Training program.')

# Device settings
parser.add_argument('--gpu',
                    type=int,
                    nargs='+',
                    required=True,
                    help='Specify one or more GPUs to use.')

# TensorBoard summary stuff
parser.add_argument(
    '--log_dir',
    dest='root',
    default='runs',
    help='Root directory in which to save run. Default: "./runs/"')
parser.add_argument(
    '--run',
    dest='run',
    default=datetime.now().isoformat(),
    help='Name of the current training run. Default: Current ISO time')

# Training/Testing Parameters
parser.add_argument('--epochs',
                    default=40,
                    type=int,
                    help='Number of epochs to train for. Default: 40')
parser.add_argument(
    '--batch',
    default=1024,
    type=int,
    help='The batch size used for training and testing. Default: 1024')
parser.add_argument('--dataset_frac',
                    default=1.0,
                    type=float,
                    help='Fraction of dataset to use. Default: 1.0')
parser.add_argument('--train_split',
                    default=0.8,
                    type=float,
                    help='Train/Test split fraction. Default: 0.8')
parser.add_argument('--examples',
                    default=5,
                    type=int,
                    help='Number of examples to save. Default: 5')

# Parse commmand line args
args = parser.parse_args()

# Training/Testing Parameters
IMPULSE_WIDTH = 10  # Impulse width
PRED_TOLERANCE = 10  # Time error tolerance

# Locations of STEAD Dataset
NPY_FILE = '/net/arius/scratch/cs101/STEAD/stead_full.npy'
CSV_FILE = '/net/arius/scratch/cs101/STEAD/stead_metadata_new.csv'

# Device is the destination device
device = torch.device('cuda', args.gpu[0])
torch.backends.cudnn.benchmark = True

# Set up the impulse signal
impulse = DownscaledImpulse(NoisyImpulse(SpikeImpulse(1.0), 0.05), 2)

# Load the dataset
full_dataset = STEADDataset(CSV_FILE,
                            NPY_FILE,
                            impulse,
                            crop=4096,
                            p_uncertainty=3.0,
                            s_uncertainty=5.0,
                            crop_keep_both=True)
full_dataset.filter(lambda df: (df['s_arrival_sample'] - df[
    'p_arrival_sample'] < 3000) | (df['trace_category'] != 'earthquake_local'))

# Split into train, test, and example sets
torch.manual_seed(42)
full_size = len(full_dataset)
fraction_size = int(full_size * args.dataset_frac)
train_size = int(fraction_size * args.train_split)
test_size = fraction_size - train_size - args.examples
leftover = full_size - fraction_size
train, test, examples, _ = random_split(
    full_dataset, [train_size, test_size, args.examples, leftover])
torch.manual_seed(torch.initial_seed())

# Set up the PyTorch DataLoaders
train_data = DataLoader(train,
                        batch_size=args.batch,
                        shuffle=True,
                        num_workers=4)
test_data = DataLoader(test,
                       batch_size=args.batch,
                       shuffle=True,
                       num_workers=4)

# TensorBoard writer
writer = SummaryWriter(os.path.join(args.root, args.run))

e_start = 0

# Set up the model that should be used.
model = ParallelModel(16, 2048).to(device)
# NOTE: If you want to load a model, do it here

# Set up the model for the GPU(s)
par_model = DataParallel(model, device_ids=args.gpu)
prior = distributions.MultivariateNormal(
    torch.zeros(2).to(device),
    torch.eye(2).to(device))

# Set up the optimizer and LR scheduler
optimizer = optim.SGD(model.parameters(), lr=4e-5, momentum=0.9, nesterov=True)
lr_sched = optim.lr_scheduler.CyclicLR(optimizer,
                                       base_lr=4e-5,
                                       max_lr=16e-5,
                                       mode='triangular2',
                                       step_size_up=4 * len(train_data))

pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Number of Trainable Params:', pytorch_total_params)

for e in range(args.epochs):
    print(f'Epoch {e}:')

    loss_total = 0

    # Run through the training set
    for idx, batch in enumerate(tqdm(train_data)):
        # Get inputs and labels, and move them to device
        trace = batch['trace']
        p_label = batch['p_impulse']
        s_label = batch['s_impulse']
        label = torch.hstack((p_label, s_label))

        # Zero gradients
        optimizer.zero_grad()

        # Run through the forward pass
        zp, log_det = par_model(label, trace)

        # Calculate the log probability of the transformation
        log_prob = prior.log_prob(zp.transpose(1, 2)).squeeze()
        prob = -(log_det + log_prob).mean()

        # Calculate loss
        loss = prob

        # Optimizer step
        loss.backward()
        optimizer.step()

        # LR scheduler step
        lr_sched.step()

        # Log loss to TensorBoard
        writer.add_scalar('Loss/Batch/Train', loss.item(),
                          (e_start + e) * len(train_data) + idx)
        loss_total += loss.item()

    # Inference only for now
    model.eval()

    # Save example predictions
    with torch.no_grad():

        for i in tqdm(range(args.examples)):
            # Make predictions for the ith trace.
            example = examples[i]
            trace = example['trace'].unsqueeze(0).to(device)  # ith trace
            p_label = example['p_impulse'].unsqueeze(0)
            s_label = example['s_impulse'].unsqueeze(0)
            label = torch.hstack((p_label, s_label)).to(device)

            sample = prior.sample((200, 2048)).transpose(1, 2).squeeze(dim=-1)

            estimation, _ = model.reverse(sample, trace.repeat(200, 1, 1))

            # Create a trace plot
            fig, axes = plt.subplots(nrows=3, sharex=True)

            # Plot the trace
            axes[0].plot(nn.functional.avg_pool1d(
                example['trace'].unsqueeze(0), 2, stride=2)[0, 0],
                         color='k')

            def probability(data, threshold=0.2):
                return (data > threshold).float().mean(dim=0)

            axes[1].plot(torch.arange(2048),
                         probability(estimation[:, 0]).detach().cpu())
            axes[2].plot(torch.arange(2048),
                         probability(estimation[:, 1]).detach().cpu())

            if example['p_idx'] > 0:
                for ax in range(3):
                    axes[ax].axvline(example['p_idx'] / 2,
                                     color='dodgerblue',
                                     label='P-Arrival',
                                     linestyle=':')

            if example['s_idx'] > 0:
                for ax in range(3):
                    axes[ax].axvline(example['s_idx'] / 2,
                                     color='orangered',
                                     label='S-Arrival',
                                     linestyle=':')

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right')

            writer.add_figure(f'Prediction/Trace{i}', fig, e_start + e)

            latent, _ = model(label, trace)
            inverse, _ = model.reverse(latent, trace)

            # Create Latent Space Visualization
            fig, axes = plt.subplots(nrows=3, sharex=True)

            # Plot the original p impulse
            axes[0].plot(example['p_impulse'][0], color='black')
            axes[1].plot(latent[0, 0].squeeze().cpu(), color='black')
            axes[2].plot(inverse[0, 0].squeeze().cpu(), color='black')

            writer.add_figure(f'LatentP/Trace{i}', fig, e_start + e)

            # Create Latent Space Visualization
            fig, axes = plt.subplots(nrows=3, sharex=True)

            # Plot the original p impulse
            axes[0].plot(example['s_impulse'][0], color='black')
            axes[1].plot(latent[0, 1].squeeze().cpu(), color='black')
            axes[2].plot(inverse[0, 1].squeeze().cpu(), color='black')

            writer.add_figure(f'LatentS/Trace{i}', fig, e)

    # Re-enable training
    model.train()

    # Save a model checkpoint
    if e % 20 == 19:
        torch.save(model.state_dict(), f'{args.run}_e{e_start + e}.pt')

# Make sure the whole TensorBoard log gets saved
writer.flush()
