import argparse
import os
import random
from datetime import datetime

# PyTorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Misc. Imports
from tqdm import tqdm
import matplotlib.pyplot as plt

# EQDetection Imports
from eqdetection.stead import STEADDataset, SquareImpulse, Standardizer
from eqdetection.model import Network
from eqdetection.util import Statistics


# Create a command line argument parser
parser = argparse.ArgumentParser('Training program.')

# TensorBoard summary stuff
parser.add_argument('--log_dir', dest='root', default='runs',
                    help='Root directory in which to save run. Default: "./runs/"')
parser.add_argument('--run', dest='run', default=datetime.now().isoformat(),
                    help='Name of the current training run. Default: Current ISO time')

# Device settings
parser.add_argument('--gpu', type=int,
                    help='Specify a GPU index to use. Defaults to CPU.')

# Training/Testing Parameters
parser.add_argument('--epochs', default=40, type=int,
                    help='Number of epochs to train for. Default: 40')
parser.add_argument('--batch', default=1024, type=int,
                    help='The batch size used for training and testing. Default: 1024')
parser.add_argument('--dataset_frac', default=1.0,
                    type=float, help='Fraction of dataset to use. Default: 1.0')
parser.add_argument('--train_split', default=0.8, type=float,
                    help='Train/Test split fraction. Default: 0.8')
parser.add_argument('--examples', default=5, type=int,
                    help='Number of examples to save. Default: 5')

# Parse commmand line args
args = parser.parse_args()

# Training/Testing Parameters
IMPULSE_WIDTH = 10  # Impulse width
PRED_TOLERANCE = 10  # Time error tolerance

# Locations of STEAD Dataset
NPY_FILE = '/scratch/cs101/STEAD/stead_full.npy'
CSV_FILE = '/scratch/cs101/STEAD/stead_metadata_new.csv'

# Set up the device to use
if args.gpu is not None:
    device = torch.device('cuda', args.gpu)
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

# Set up the impulse signal
impulse = SquareImpulse(IMPULSE_WIDTH, 1)

# Load the dataset
standardizer = Standardizer()
full_dataset = STEADDataset(
    CSV_FILE, NPY_FILE, impulse, transform=standardizer)

# Split into train, test, and example sets
full_size = len(full_dataset)
fraction_size = int(full_size * args.dataset_frac)
train_size = int(fraction_size * args.train_split)
test_size = fraction_size - train_size - args.examples
leftover = full_size - fraction_size
train, test, examples, _ = random_split(
    full_dataset, [train_size, test_size, args.examples, leftover])

# Set up the PyTorch DataLoaders
train_data = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=4)
test_data = DataLoader(test, batch_size=args.batch, shuffle=True, num_workers=4)

# TensorBoard writer
writer = SummaryWriter(os.path.join(args.root, args.run))

# Initialize the network and move to device
model = Network().to(device)

# Initialize criteria (loss) and set weights
criterion_p = nn.BCEWithLogitsLoss().to(device)
criterion_s = nn.BCEWithLogitsLoss().to(device)
criterion_e = nn.BCEWithLogitsLoss().to(device)
WEIGHT_P = 0.2
WIEGHT_S = 0.3
WEIGHT_E = 0.5
THRESHOLD_D = 0.5
THRESHOLD_P = 0.3
THRESHOLD_S = 0.3

# Set up the optimizer and the LR scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiplicativeLR(
    optimizer,
    lambda e: 0.1 if e % 20 == 19 else 1
)

# Each iteration is one full pass through the data
for e in range(args.epochs):
    print(f'Epoch {e}:')

    loss_total = 0

    # Run through the training set
    for idx, batch in enumerate(tqdm(train_data)):
        # Get inputs and labels, and move them to device
        trace = batch['trace'].to(device)
        p_label = batch['p_impulse'].to(device)
        s_label = batch['s_impulse'].to(device)
        e_label = batch['e_impulse'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Make predictions
        (p_pred, s_pred, e_pred) = model(trace)

        # Compute loss
        p_loss = criterion_p(p_pred, p_label)
        s_loss = criterion_s(s_pred, s_label)
        e_loss = criterion_e(e_pred, e_label)
        loss = WEIGHT_P * p_loss + WIEGHT_S * s_loss + WEIGHT_E * e_loss

        # Make optimization step
        loss.backward()
        optimizer.step()

        # Log loss to TensorBoard
        writer.add_scalar('Loss/Batch/Train', loss.item(),
                          e * len(train_data) + idx)
        loss_total += loss.item()

    # Log whole-epoch loss average
    writer.add_scalar('Loss/Train', loss_total / len(train_data), e)

    loss_total = 0
    p_mae = 0
    s_mae = 0
    p_stats = Statistics(device=device)
    s_stats = Statistics(device=device)
    d_stats = Statistics(device=device)

    # Inference only for now
    model.eval()

    # Run through the test set
    with torch.no_grad():
        for batch in tqdm(test_data):
            # Get inputs and labels, and move them to device
            trace = batch['trace'].to(device)
            p_label = batch['p_impulse'].to(device)
            s_label = batch['s_impulse'].to(device)
            e_label = batch['e_impulse'].to(device)
            p_idx = batch['p_idx'].to(device)
            s_idx = batch['s_idx'].to(device)

            # Is the signal noise?
            is_noise = (p_idx < 0) * (s_idx < 0)

            # Make predictions
            (p_pred, s_pred, e_pred) = model(trace)

            # Compute loss
            p_loss = criterion_p(p_pred, p_label)
            s_loss = criterion_s(s_pred, s_label)
            e_loss = criterion_e(e_pred, e_label)
            loss = WEIGHT_P * p_loss + WIEGHT_S * s_loss + WEIGHT_E * e_loss
            loss_total += loss.item()

            # Compute Accuracy
            p_mags, p_times = p_pred.squeeze().max(1)
            s_mags, s_times = s_pred.squeeze().max(1)
            d_mags, _ = e_pred.squeeze().max(1)
            
            # Prediction masks
            p_pred = p_mags > THRESHOLD_P
            s_pred = s_mags > THRESHOLD_S
            detection = d_mags > THRESHOLD_D

            # Absolute error in the time prediction
            p_pred_err = torch.abs(p_idx - p_times)
            s_pred_err = torch.abs(s_idx - s_times)

            # Mask of correct predictions in non-noise traces
            p_pred_corr = (p_pred_err < PRED_TOLERANCE) * p_pred * ~is_noise
            s_pred_corr = (s_pred_err < PRED_TOLERANCE) * s_pred * ~is_noise
            
            # Mask of correct prediction of noise traces
            p_noise_corr = ~p_pred * is_noise
            s_noise_corr = ~s_pred * is_noise
            
            # Mask of correct prediction overall
            p_corr = p_pred_corr + p_noise_corr
            s_corr = s_pred_corr + s_noise_corr
            
            # Accumulate statistics
            p_stats.add_corr_actual(p_corr, ~is_noise)
            s_stats.add_corr_actual(s_corr, ~is_noise)
            d_stats.add_pred_actual(detection, ~is_noise)
            
            p_err_valid = p_pred * ~is_noise
            p_valid_sum = p_err_valid.float().sum().item()
            if p_valid_sum == 0:
                p_valid_sum = 1.0
            p_mae += (p_pred_err * p_err_valid).float().sum() / p_valid_sum
                
            s_err_valid = s_pred * ~is_noise
            s_valid_sum = s_err_valid.float().sum().item()
            if s_valid_sum == 0:
                s_valid_sum = 1.0
            s_mae += (s_pred_err * s_err_valid).float().sum() / s_valid_sum
    
    loss_total /= len(test_data)
    p_mae /= len(test_data)
    s_mae /= len(test_data)

    # Log loss to TensorBoard
    writer.add_scalar('Loss/Test', loss_total, e)
    writer.add_scalar('P-Arrival/MAE', p_mae, e)
    writer.add_scalar('S-Arrival/MAE', s_mae, e)
    p_stats.log(writer, 'P-Arrival', e)
    s_stats.log(writer, 'S-Arrival', e)
    d_stats.log(writer, 'Detection', e)

    # Save example predictions
    with torch.no_grad():
        for i in tqdm(range(args.examples)):
            # Make predictions for the ith trace.
            example = examples[i]
            trace = example['trace'].unsqueeze(0).to(device) # ith trace
            (p_pred, s_pred, c_pred) = model(trace)          # prediction

            # Take sigmoid of prediction and scale to trace range
            p_pred = torch.sigmoid(p_pred).squeeze().cpu()
            s_pred = torch.sigmoid(s_pred).squeeze().cpu()
            c_pred = torch.sigmoid(c_pred).squeeze().cpu()

            # Get the trace plot
            fig, axes = plt.subplots(nrows=3, sharex=True)

            # Plot predictions
            for (idx, ax) in enumerate(axes):
                ax.plot(example['trace'][idx, :], color='k')
                
                y_min, y_max = ax.get_ylim()
                scaling = 0.9 * (y_max - y_min)
                shift = 0.9 * y_min

                ax.axvline(example['p_idx'], color='dodgerblue', label='P-Arrival', linestyle=':')
                ax.axvline(example['s_idx'], color='orangered', label='S-Arrival', linestyle=':')
                ax.axvline(example['e_idx'], color='green', label='Coda End', linestyle=':')

                ax.plot(scaling * p_pred + shift, color='deepskyblue',
                        label='Predicted P-Arrival')
                ax.plot(scaling * s_pred + shift, color='firebrick',
                        label='Predicted S-Arrival')
                ax.plot(scaling * c_pred + shift, color='limegreen',
                        label='Predicted Coda', linestyle='--')

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right')

            writer.add_figure(f'Prediction/Trace{i}', fig, e)

    # Re-enable training
    model.train()

    # Learning rate scheduler
    scheduler.step()

# Make sure the whole TensorBoard log gets saved
writer.flush()