import random
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Impulse:
    def get_impulse(self, start, end, size):
        return torch.zeros(1, size)


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
            mag = self.magnitude * (i / self.width)
            impulse[0, lo] = mag
            impulse[0, hi] = mag
        return impulse


class Standardizer():
    def __call__(self, x):
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        std = torch.where(std == 0, torch.tensor(1.0), std)
        return (x - mean) / std


class STEADDataset(data.Dataset):
    def __init__(self, csv_file, npy_file, impulse, transform=None, init_data=None, crop=None):
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
    
    def __len__(self):
        return len(self.metadata)
    
    def __get_trace(self, idx):
        # Get trace name
        trace_attrs = self.metadata.iloc[idx]
        trace_index = trace_attrs['trace_index']
        
        # Get the trace from the numpy file by name
        trace = self.trace_data[trace_index]
        
        return trace_attrs, trace
    
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
            # Create the p_arrival impulse using the time index
            p_arrival_idx = int(trace_attrs['p_arrival_sample'])
            p_impulse = self.impulse.get_impulse(p_arrival_idx, p_arrival_idx, length)
            
            s_arrival_idx = int(trace_attrs['s_arrival_sample'])
            s_impulse = self.impulse.get_impulse(s_arrival_idx, s_arrival_idx, length)
            
            end_idx = int(trace_attrs['coda_end_sample'])
            end_impulse = self.impulse.get_impulse(p_arrival_idx, end_idx, length)

        if self.crop is not None:
            crop_start = random.randint(0, length - self.crop)
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
