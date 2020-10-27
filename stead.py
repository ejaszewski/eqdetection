import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

class Impulse:
    def get_impulse(self, start, end, size):
        return torch.zeros(1, size)

class SquareImpulse(Impulse):
    def __init__(self, width, magnitude):
        self.width = width
        self.magnitude = magnitude
    
    def get_impulse(self, start, end, size):
        impulse = torch.zeros(1, size)
        impulse[0, start - self.width:end + self.width] = self.magnitude
        return impulse

class STEADDataset(data.Dataset):
    def __init__(self, csv_file, npy_file, impulse, transform=None, init_data=None):
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
        
        # Create the p_arrival impulse using the time index
        p_arrival_idx = int(trace_attrs['p_arrival_sample'])
        p_impulse = self.impulse.get_impulse(p_arrival_idx, p_arrival_idx, trace.size()[1])
        
        s_arrival_idx = int(trace_attrs['s_arrival_sample'])
        s_impulse = self.impulse.get_impulse(s_arrival_idx, s_arrival_idx, trace.size()[1])
        
        coda_end_idx = int(trace_attrs['coda_end_sample'])
        c_impulse = self.impulse.get_impulse(p_arrival_idx, coda_end_idx, trace.size()[1])
        
        if self.transform:
            trace = self.transform(trace)
        
        return trace, (p_impulse, s_impulse, c_impulse)
    
    def filter(self, f):
        self.metadata = self.metadata[f(self.metadata)]
    
    def split(self, frac, random=True):
        count = int(len(self.metadata) * frac)
        indices = np.random.permutation(self.metadata.index) if random else self.metadata.index
        p1 = self.metadata.loc[indices[:count]]
        p2 = self.metadata.loc[indices[count:]]
        d1 = STEADDataset(None, None, self.impulse, self.transform, init_data=(p1, self.trace_data))
        d2 = STEADDataset(None, None, self.impulse, self.transform, init_data=(p2, self.trace_data))
        return (d1, d2)
    
    def show(self, idx):
        # Get the trace and its metadata
        trace_attrs, trace = self.__get_trace(idx)
        
        # Arrival times vertical lines
        p_time = trace_attrs['p_arrival_sample']
        s_time = trace_attrs['s_arrival_sample']
        c_time = trace_attrs['coda_end_sample']
        
        # Set up the plot
        fig, (ax_e, ax_n, ax_z) = plt.subplots(nrows=3, sharex=True)
        fig.set_size_inches(8, 6)
        
        # Plot the e signal
        ax_e.plot(trace[:,0], color='k')
        ax_e.axvline(p_time, color='dodgerblue', label='P-Arrival')
        ax_e.axvline(s_time, color='orangered', label='S-Arrival')
        ax_e.axvline(c_time, color='green', label='Coda End')
        
        # Plot the n signal
        ax_n.plot(trace[:,1], color='k')
        ax_n.axvline(p_time, color='dodgerblue', label='P-Arrival')
        ax_n.axvline(s_time, color='orangered', label='S-Arrival')
        ax_n.axvline(c_time, color='green', label='Coda End')
        
        # Plot the z signal
        ax_z.plot(trace[:,1], color='k')
        ax_z.axvline(p_time, color='dodgerblue', label='P-Arrival')
        ax_z.axvline(s_time, color='orangered', label='S-Arrival')
        ax_z.axvline(c_time, color='green', label='Coda End')
        
        # Set up the rest of the plot
        handles, labels = ax_e.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.suptitle(f'Trace {trace_attrs["trace_name"]}')