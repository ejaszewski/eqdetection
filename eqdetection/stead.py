import math
import random
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Impulse:
    def get_impulse(self, start, end, size):
        return torch.zeros(1, size)

class SpikeImpulse(Impulse):
    def __init__(self, magnitude):
        self.magnitude = magnitude

    def get_impulse(self, start, end, size):
        impulse = torch.zeros(1, size)
        impulse[0, start] = self.magnitude
        return impulse

class SquareImpulse(Impulse):
    def __init__(self, width, magnitude):
        self.width = width
        self.magnitude = magnitude
    
    def get_impulse(self, start, end, size):
        impulse = torch.zeros(1, size)
        lo = max(start - self.width, 0)
        hi = min(end + self.width, size - 1)
        impulse[0, lo:hi] = self.magnitude
        return impulse
        

class TriangleImpulse(Impulse):
    def __init__(self, width, magnitude):
        self.width = width
        self.magnitude = magnitude
    
    def get_impulse(self, start, end, size):
        impulse = torch.zeros(1, size)
        impulse[0, start:end] = self.magnitude
        for i in range(self.width):
            lo = max(start - i, 0)
            hi = min(end + i, size - 1)
            mag = self.magnitude * (1.0 - (i / self.width))
            impulse[0, lo] = mag
            impulse[0, hi] = mag
        return impulse

class ProbSquareImpulse(Impulse):
    def __init__(self, width, scale):
        self.sigma = width / math.sqrt(2 * math.pi)
        self.scale = scale
    
    def get_impulse(self, start, end, size):
        idxs = torch.arange(size).unsqueeze(0)
        probs = (-0.5 * ((idxs - start) / self.sigma).pow(2)).exp()
        return torch.bernoulli(probs) * self.scale

class NoisyImpulse(Impulse):
    def __init__(self, impulse, noise):
        self.impulse = impulse
        self.noise = noise
    
    def get_impulse(self, start, end, size):
        impulse = self.impulse.get_impulse(start, end, size)
        impulse += self.noise * torch.randn(1, size)
        return impulse

class BoundedNoisyImpulse(Impulse):
    def __init__(self, impulse, noise):
        self.impulse = impulse
        self.noise = noise

    def get_impulse(self, start, end, size):
        impulse = self.impulse.get_impulse(start, end, size)
        impulse += (1 - impulse * 2) * self.noise * torch.randn(1, size).abs()
        return 0.95 * impulse + 0.025

class Standardizer():
    def __call__(self, x):
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        std = torch.where(std == 0, torch.tensor(1.0), std)
        return (x - mean) / std


class STEADDataset(data.Dataset):
    def __init__(self, csv_file, npy_file, impulse, transform=None, init_data=None, crop=None, p_uncertainty=0, s_uncertainty=0):
        if init_data != None: # Create a dataset with pre-loaded data
            self.metadata = init_data[0]
            self.trace_data = init_data[1]
        else: # Load data from file
            print('Loading metadata...')
            self.metadata = pd.read_csv(csv_file)
            print('Reading trace files...')
            print('Loading data into memory...')
            self.trace_data = np.load(npy_file, mmap_mode='r')
            print('Done.')
        
        self.impulse = impulse
        self.transform = transform
        self.crop = crop
        self.p_uncertainty = p_uncertainty
        self.s_uncertainty = s_uncertainty
    
    def __len__(self):
        return len(self.metadata)
    
    def __get_trace(self, idx):
        # Get trace name
        trace_attrs = self.metadata.iloc[idx]
        trace_index = trace_attrs['trace_index']
        
        # Get the trace from the numpy file by name
        trace = self.trace_data[trace_index]
        
        return trace_attrs, trace.copy()
    
    def __getitem__(self, idx):
        # Get the trace and its metadata
        trace_attrs, trace = self.__get_trace(idx)
        
        # Convert to PyTorch tensor
        trace = torch.transpose(torch.from_numpy(trace), 0, 1)
        length = trace.size()[1]
        
        is_noise = trace_attrs['trace_category'] == 'noise'

        if is_noise:
            p_arrival_idx = -1
            s_arrival_idx = -1
            end_idx = -1

            p_impulse = torch.zeros(1, length)
            s_impulse = torch.zeros(1, length)
            end_impulse = torch.zeros(1, length)
        else:
            # Perturb p-arrival by uncertainty in measurement
            p_perturb = 0
            if self.p_uncertainty > 0:
                p_perturb = int(torch.normal(torch.zeros(1), self.p_uncertainty).item())

            # Perturb s-arrival by uncertainty in measurement
            s_perturb = 0
            if self.s_uncertainty > 0:
                s_perturb = int(torch.normal(torch.zeros(1), self.s_uncertainty).item())

            # Create the p_arrival impulse using the time index
            p_arrival_idx = int(trace_attrs['p_arrival_sample']) + p_perturb
            p_impulse = self.impulse.get_impulse(p_arrival_idx, p_arrival_idx, length)
            
            s_arrival_idx = int(trace_attrs['s_arrival_sample']) + s_perturb
            s_impulse = self.impulse.get_impulse(s_arrival_idx, s_arrival_idx, length)
            
            end_idx = int(trace_attrs['coda_end_sample'])
            end_impulse = self.impulse.get_impulse(p_arrival_idx, end_idx, length)

        if self.crop is not None:
            crop_min = 0
            crop_max = length - self.crop

            # If not a noise signal
            if p_arrival_idx > 0:
                # Whether to guarantee the P or the S
                keep_p = random.random() < 0.5

                # Guarantee the P arrival
                if keep_p:
                    crop_min = max(0, p_arrival_idx - self.crop)
                    crop_max = min(p_arrival_idx, crop_max)
                # Guarantee the S arrival
                else:
                    crop_min = max(0, s_arrival_idx - self.crop)
                    crop_max = min(s_arrival_idx, crop_max)

            crop_start = random.randint(crop_min, crop_max)

            crop_end = crop_start + self.crop

            trace = trace.narrow(1, crop_start, self.crop)

            p_impulse = p_impulse.narrow(1, crop_start, self.crop)
            s_impulse = s_impulse.narrow(1, crop_start, self.crop)
            end_impulse = end_impulse.narrow(1, crop_start, self.crop)

            if p_arrival_idx < crop_start or p_arrival_idx >= crop_end:
                p_arrival_idx = -1
            else:
                p_arrival_idx -= crop_start
            
            if s_arrival_idx < crop_start or s_arrival_idx >= crop_end:
                s_arrival_idx = -1
            else:
                s_arrival_idx -= crop_start

            if end_idx < crop_start or end_idx >= crop_end:
                end_idx = -1
            else:
                end_idx -= crop_start

        if self.transform:
            trace = self.transform(trace)

        return {
            'trace': trace,
            'p_impulse': p_impulse,
            's_impulse': s_impulse,
            'e_impulse': end_impulse,
            'p_idx': p_arrival_idx,
            's_idx': s_arrival_idx,
            'e_idx': end_idx
        }

    def filter(self, f):
        self.metadata = self.metadata[f(self.metadata)]

    def plot(self, idx, codastyle='-'):
        # Get the trace and its metadata
        trace_attrs, trace = self.__get_trace(idx)
        
        # Arrival times vertical lines
        p_time = trace_attrs['p_arrival_sample']
        s_time = trace_attrs['s_arrival_sample']
        e_time = trace_attrs['coda_end_sample']
        
        # Set up the plot
        fig, (ax_e, ax_n, ax_z) = plt.subplots(nrows=3, sharex=True)
        fig.set_size_inches(8, 6)
        
        # Plot the e signal
        ax_e.plot(trace[:,0], color='k')
        ax_e.axvline(p_time, color='dodgerblue', label='P-Arrival', linestyle=codastyle)
        ax_e.axvline(s_time, color='orangered', label='S-Arrival', linestyle=codastyle)
        ax_e.axvline(e_time, color='green', label='Coda End', linestyle=codastyle)
        
        # Plot the n signal
        ax_n.plot(trace[:,1], color='k')
        ax_n.axvline(p_time, color='dodgerblue', label='P-Arrival', linestyle=codastyle)
        ax_n.axvline(s_time, color='orangered', label='S-Arrival', linestyle=codastyle)
        ax_n.axvline(e_time, color='green', label='Coda End', linestyle=codastyle)
        
        # Plot the z signal
        ax_z.plot(trace[:,1], color='k')
        ax_z.axvline(p_time, color='dodgerblue', label='P-Arrival', linestyle=codastyle)
        ax_z.axvline(s_time, color='orangered', label='S-Arrival', linestyle=codastyle)
        ax_z.axvline(e_time, color='green', label='Coda End', linestyle=codastyle)
        
        # Set up the rest of the plot
        handles, labels = ax_e.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.suptitle(f'Trace {trace_attrs["trace_name"]}')

        return fig
