import random
import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class STEADDataset(data.Dataset):
    """A PyTorch Dataset for the STEAD dataset.

    This class implements a PyTorch map-style dataset for the STEAD dataset.
    The STEAD dataset has to be provided as a metadata CSV file and Numpy
    binary file containing the traces in the order specified in the metadata
    file. The metadata is loaded and kept in memory and the traces are mmapped
    to reduce total memory footprint.

    Attributes:
        metadata: The STEAD metadata.
        trace_data: The STEAD traces.
        impulse: The impulse class to use.
        crop: The width to crop to.
        crop_keep_both: Whether to preserve both P and S during cropping.
        p_uncertainty: Amount of uncertainty to add to P arrival time.
        s_uncertainty: Amount of uncertainty to add to S arrival time.
    """
    def __init__(self,
                 csv_file,
                 npy_file,
                 impulse,
                 standardize=True,
                 crop=None,
                 crop_keep_both=False,
                 p_uncertainty=0,
                 s_uncertainty=0):
        """Initializes a STEADDataset with the requested settings.

        Args:
            csv_file: Path to a CSV file with the STEAD metadata.
            npy_file: Path to a Numpy binary file with the STEAD traces.
            impulse: The impulse class to use when generating impulses.
            standardize: Whether to standardize the earthquake traces.
            crop: Optional; The length to crop traces/impulses to.
                Default: None (no crop)
            crop_keep_both: Optional; Whether to preserve both P and S during
                cropping. Default: False
            p_uncertainty: Optional; Amount of uncertainty to add to P arrival
                time. Default: 0
            s_uncertainty: Optional; Amount of uncertainty to add to S arrival
                time. Default: 0
        """
        print('Loading metadata...')
        self.metadata = pd.read_csv(csv_file)
        print('Reading trace files...')
        print('Loading data into memory...')
        self.trace_data = np.load(npy_file, mmap_mode='r')
        print('Done.')

        self.impulse = impulse
        self.standardize = standardize
        self.crop = crop
        self.crop_keep_both = crop_keep_both
        self.p_uncertainty = p_uncertainty
        self.s_uncertainty = s_uncertainty

    def __len__(self):
        return len(self.metadata)

    def __get_trace(self, idx):
        """Retrieves a copy of the trace with the given index.
        
        Args:
            idx: Index of the trace to retrieve.
        
        Returns:
            A tuple containing the trace attributes and the trace as a Numpy
            array.
        """
        # Get trace name
        trace_attrs = self.metadata.iloc[idx]
        trace_index = trace_attrs['trace_index']

        # Get the trace from the numpy file by name
        trace = self.trace_data[trace_index]

        return trace_attrs, trace.copy()

    def __getitem__(self, idx):
        """Returns the dataset entry with the given index.

        This function implements the __getitem__ function required by the
        PyTorch Dataset class. The function will retrieve the trace at the
        given index, add uncertainty, crop it, and standardize it before
        formatting it as a dict and returning it.

        Args:
            idx: Index of the trace to retrieve.
        
        Returns:
            A dict containing the trace, impulses, and some metadata. The
            contents are as follows:
            - trace: The three (E, N, Z) traces as a torch.Tensor
            - p_impulse: The P arrival impulse as a torch.Tensor
            - s_impulse: The S arrival impulse as a torch.Tensor
            - e_impulse: The coda end impulse as a torch.Tensor
            - p_idx: The index of the P arrival.
            - s_idx: The index of the S arrival.
            - e_idx: The index of the coda end.
        """
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
        else:
            # Perturb p-arrival by uncertainty in measurement
            p_perturb = 0
            if self.p_uncertainty > 0:
                p_perturb = int(
                    torch.normal(torch.zeros(1), self.p_uncertainty).item())

            # Perturb s-arrival by uncertainty in measurement
            s_perturb = 0
            if self.s_uncertainty > 0:
                s_perturb = int(
                    torch.normal(torch.zeros(1), self.s_uncertainty).item())

            # Create the p_arrival impulse using the time index
            p_arrival_idx = int(trace_attrs['p_arrival_sample']) + p_perturb
            s_arrival_idx = int(trace_attrs['s_arrival_sample']) + s_perturb
            end_idx = int(trace_attrs['coda_end_sample'])

        if self.crop is not None:
            crop_min = 0
            crop_max = length - self.crop

            # If not a noise signal
            if p_arrival_idx > 0:
                if self.crop_keep_both:
                    crop_min = max(0, s_arrival_idx - self.crop + 1)
                    crop_max = min(p_arrival_idx, crop_max)
                else:
                    # Whether to guarantee the P or the S
                    keep_p = random.random() < 0.0

                    # Guarantee the P arrival
                    if keep_p:
                        crop_min = max(0, p_arrival_idx - self.crop + 1)
                        crop_max = min(p_arrival_idx, crop_max)
                    # Guarantee the S arrival
                    else:
                        crop_min = max(0, s_arrival_idx - self.crop + 1)
                        crop_max = min(s_arrival_idx, crop_max)

            crop_start = random.randint(crop_min, crop_max)

            crop_end = crop_start + self.crop

            trace = trace.narrow(1, crop_start, self.crop)

            # Adjust P arrival for crop
            if p_arrival_idx < crop_start or p_arrival_idx >= crop_end:
                p_arrival_idx = -1
            else:
                p_arrival_idx -= crop_start

            # Adjust S arrival for crop
            if s_arrival_idx < crop_start or s_arrival_idx >= crop_end:
                s_arrival_idx = -1
            else:
                s_arrival_idx -= crop_start

            # Adjust coda end for crop
            if end_idx < crop_start or end_idx >= crop_end:
                end_idx = -1
            else:
                end_idx -= crop_start

            # Create impulses
            p_impulse = self.impulse.get_impulse(p_arrival_idx, p_arrival_idx,
                                                 self.crop)
            s_impulse = self.impulse.get_impulse(s_arrival_idx, s_arrival_idx,
                                                 self.crop)
            end_impulse = self.impulse.get_impulse(p_arrival_idx, end_idx,
                                                   self.crop)

        # Standardize the trace along the time dimension.
        if self.standardize:
            std, mean = torch.std_mean(trace, dim=-1, keepdim=True)
            std = torch.where(std == 0, torch.tensor(1.0), std)
            trace = (trace - mean) / std

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
        """Apply a filtering function to the dataset.

        This function allows for the application of a Pandas dataframe
        filtering function to the dataset to remove undesired options. The
        provided filtering function should accept a Pandas dataframe as an
        argument. The STEAD metadata will be passed into the filtering
        function, and output of the filter will be used to select which items
        should be kept.

        Args:
            f: The filtering function to apply.
        """
        self.metadata = self.metadata[f(self.metadata)]

    def plot(self, idx, codastyle='-'):
        """Plot a plot of the trace at the given index.

        This function retrieves and plots the trace at the given index. The
        plot includes the E, N, and Z components (in that order), as well as
        vertical lines corresponding to the P and S arrivals and the "coda
        end". The line style for the vertical lines can be specified.

        Args:
            idx: Index of the trace to plot.
        
        Returns:
            A figure containing three axes, arranged vertically, that
            correspond to the E, N, and Z components of the trace at the given
            index.
        """

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
        ax_e.plot(trace[:, 0], color='k')
        ax_e.axvline(p_time,
                     color='dodgerblue',
                     label='P-Arrival',
                     linestyle=codastyle)
        ax_e.axvline(s_time,
                     color='orangered',
                     label='S-Arrival',
                     linestyle=codastyle)
        ax_e.axvline(e_time,
                     color='green',
                     label='Coda End',
                     linestyle=codastyle)

        # Plot the n signal
        ax_n.plot(trace[:, 1], color='k')
        ax_n.axvline(p_time,
                     color='dodgerblue',
                     label='P-Arrival',
                     linestyle=codastyle)
        ax_n.axvline(s_time,
                     color='orangered',
                     label='S-Arrival',
                     linestyle=codastyle)
        ax_n.axvline(e_time,
                     color='green',
                     label='Coda End',
                     linestyle=codastyle)

        # Plot the z signal
        ax_z.plot(trace[:, 1], color='k')
        ax_z.axvline(p_time,
                     color='dodgerblue',
                     label='P-Arrival',
                     linestyle=codastyle)
        ax_z.axvline(s_time,
                     color='orangered',
                     label='S-Arrival',
                     linestyle=codastyle)
        ax_z.axvline(e_time,
                     color='green',
                     label='Coda End',
                     linestyle=codastyle)

        # Set up the rest of the plot
        handles, labels = ax_e.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.suptitle(f'Trace {trace_attrs["trace_name"]}')

        return fig
