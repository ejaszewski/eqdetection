import argparse
import os
from datetime import datetime

# PyTorch stuff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Misc. Imports
from tqdm import tqdm

# STEAD
from cs101.stead import STEADDataset, SquareImpulse

# Models
from cs101.models.eqtnetwork import EQTNetwork
from cs101.models.senetwork import SENetwork

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
parser.add_argument('--dataset_frac', default=1.0,
                    type=float, help='Fraction of dataset to use. Default: 1.0')
parser.add_argument('--train_split', default=0.8, type=float,
                    help='Train/Test split fraction. Default: 0.8')
parser.add_argument('--examples', default=5, type=int,
                    help='Number of examples to save. Default: 5')
parser.add_argument('--model', default='EQT', type=str,
                    help='The model to train. Options: EQT (default), SE, SE_LSTM')

args = parser.parse_args()

# Training/Testing Parameters
IMPULSE_WIDTH = 10  # Impulse width

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

# Load the full dataset
full_dataset = STEADDataset(CSV_FILE, NPY_FILE, impulse)
# Filter out non-earthquake signals
full_dataset.filter(lambda df: df['trace_category'] == 'earthquake_local')
# Pair down the dataset to the specified fraction
full_dataset, _ = full_dataset.split(args.dataset_frac)

# Create train/test/examples split
train, test = full_dataset.split(args.train_split)
test, examples = test.split(0, tail=args.examples)

# Set up the PyTorch DataLoaders
train_data = DataLoader(train, batch_size=1000, shuffle=True, num_workers=4)
test_data = DataLoader(test, batch_size=1000, shuffle=True, num_workers=4)

# TensorBoard writer
writer = SummaryWriter(os.path.join(args.root, args.run))

# Initialize the network and move to device
if args.model == 'EQT':
    model = EQTNetwork().to(device)
elif args.model == 'SE':
    model = SENetwork().to(device)
elif args.model == 'SE_LSTM':
    model = SENetwork(lstm=True).to(device)

# Initialize criteria (loss) and set weights
criterion_p = nn.BCEWithLogitsLoss().to(device)
criterion_s = nn.BCEWithLogitsLoss().to(device)
criterion_c = nn.BCEWithLogitsLoss().to(device)
WEIGHT_P = 0.2
WIEGHT_S = 0.3
WEIGHT_C = 0.5
THRESHOLD_D = 0.5
THRESHOLD_P = 0.3
THRESHOLD_S = 0.3

# Set up the optimizer and the LR scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiplicativeLR(
    optimizer,
    lambda e: 0.1 if e % 20 == 19 else 1
)

ZERO = torch.tensor(0).to(device)
ONE = torch.tensor(1).to(device)
ZERO_F = torch.tensor(0.0).to(device)
ONE_F = torch.tensor(1.0).to(device)

# Each iteration is one full pass through the data
for e in range(args.epochs):
    print(f'Epoch {e}:')

    loss_total = 0

    # Run through the training set
    for idx, batch in enumerate(tqdm(train_data)):
        # Get inputs and labels, and move them to device
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = [label.to(device) for label in labels]

        # Zero gradients
        optimizer.zero_grad()

        # Make predictions
        (p_pred, s_pred, c_pred) = model(inputs)

        # Compute loss
        p_loss = criterion_p(p_pred, labels[0])
        s_loss = criterion_s(s_pred, labels[1])
        c_loss = criterion_c(c_pred, labels[2])
        loss = WEIGHT_P * p_loss + WIEGHT_S * s_loss + WEIGHT_C * c_loss

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
    p_acc = 0
    s_acc = 0
    p_mae = 0
    s_mae = 0
    recall = 0

    # Inference only for now
    model.eval()

    # Run through the test set
    with torch.no_grad():
        for batch in tqdm(test_data):
            # Get inputs and labels, and move them to device
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = [label.to(device) for label in labels]

            # Make predictions
            (p_pred, s_pred, c_pred) = model(inputs)

            # Compute loss
            p_loss = criterion_p(p_pred, labels[0])
            s_loss = criterion_s(s_pred, labels[1])
            c_loss = criterion_c(c_pred, labels[2])
            loss = WEIGHT_P * p_loss + WIEGHT_S * s_loss + WEIGHT_C * c_loss
            loss_total += loss.item()

            # Compute Accuracy
            p_mags, p_times = p_pred.squeeze().max(1)
            s_mags, s_times = s_pred.squeeze().max(1)
            detection, _ = c_pred.squeeze().max(1)

            p_pred_times = torch.where(p_mags > THRESHOLD_P, p_times, ZERO)
            s_pred_times = torch.where(s_mags > THRESHOLD_S, s_times, ZERO)
            detection_pred = torch.where(
                detection > THRESHOLD_D, ONE_F, ZERO_F)

            p_abs_err = torch.abs(impulse.get_index(labels[0]) - p_pred_times)
            s_abs_err = torch.abs(impulse.get_index(labels[1]) - s_pred_times)
            p_acc += torch.where(p_abs_err < IMPULSE_WIDTH,
                                 ONE_F, ZERO_F).mean()
            p_mae += p_abs_err.float().mean()
            s_acc += torch.where(s_abs_err < IMPULSE_WIDTH,
                                 ONE_F, ZERO_F).mean()
            s_mae += s_abs_err.float().mean()
            recall += detection_pred.mean()

    # Log loss to TensorBoard
    writer.add_scalar('Loss/Test', loss_total / len(test_data), e)
    writer.add_scalar('Metrics/P-Arrival/Accuracy', p_acc / len(test_data), e)
    writer.add_scalar('Metrics/S-Arrival/Accuracy', s_acc / len(test_data), e)
    writer.add_scalar('Metrics/P-Arrival/MAE', p_mae / len(test_data), e)
    writer.add_scalar('Metrics/S-Arrival/MAE', s_mae / len(test_data), e)
    writer.add_scalar('Metrics/Detection/Recall', recall / len(test_data), e)

    # Save example predictions
    with torch.no_grad():
        for i in tqdm(range(args.examples)):
            # Make predictions for the ith trace.
            example = examples[i][0].unsqueeze(0).to(device)    # ith trace
            (p_pred, s_pred, c_pred) = model(example)           # prediction

            # Take sigmoid of prediction and scale to trace range
            p_pred = torch.sigmoid(p_pred).squeeze().cpu()
            s_pred = torch.sigmoid(s_pred).squeeze().cpu()
            c_pred = torch.sigmoid(c_pred).squeeze().cpu()

            # Get the trace plot
            fig = examples.show(i)
            axes = fig.get_axes()

            impulses = examples[i][1]

            # Plot predictions
            for (ax, trace) in zip(axes, example[0]):
                scaling = 0.8 * torch.max(trace).item()

                ax.plot(scaling * p_pred, color='deepskyblue',
                        label='Predicted P-Arrival')
                ax.plot(scaling * s_pred, color='firebrick',
                        label='Predicted S-Arrival')
                ax.plot(scaling * c_pred, color='limegreen',
                        label='Predicted Coda')

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center right')

            writer.add_figure(f'Prediction/Trace{i}', fig, e)

    # Re-enable training
    model.train()

    # Learning rate scheduler
    scheduler.step()

# Make sure the whole TensorBoard log gets saved
writer.flush()
