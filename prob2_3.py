import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from scipy.ndimage import rotate
import ujson as json

import torch

from DisplayVolume import DisplayVolume 

def myfilteredbackprojection(Proj, angles, N, fc):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input to PyTorch tensors and transfer to GPU if available
    Proj = torch.tensor(Proj, dtype=torch.float).to(device)
    angles = torch.tensor(angles, dtype=torch.float).to(device)

    # Number of projections
    r, c, d = Proj.shape

    # Initialize output 3D matrix
    reconstructed = np.zeros((N, N, d))

    # Frequency axis for ramp filter
    freqs = torch.fft.fftfreq(r).to(device)

    # Create ramp filter
    ramp_filter = (torch.abs(freqs) <= fc) * torch.abs(freqs)

    # Perform FFT on all projections
    proj_fft = torch.fft.fft(Proj, dim=0)

    # Apply ramp filter
    filtered_proj_fft = proj_fft * ramp_filter[:, None, None]

    # Inverse FFT
    filtered_proj = torch.fft.ifft(filtered_proj_fft, dim=0).real

    # Coordinates for backprojection
    x, y = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2)

    # Backprojection
    for slice_index in range(d):
        slice_reconstructed = np.zeros((N, N))
        print(slice_index)
        for i, angle in enumerate(angles):
            theta = -angle.item()
            rot_x = np.cos(theta) * x - np.sin(theta) * y + r // 2

            # Interpolate using numpy
            interp_func = interp1d(np.arange(r), filtered_proj[:, i, slice_index].cpu(), bounds_error=False, fill_value=0)
            slice_reconstructed += interp_func(rot_x)

        reconstructed[:, :, slice_index] = slice_reconstructed

    return reconstructed


cutoff = 0.05

with open('Project5.json', 'r') as f:
    data = json.load(f)

projections = np.array(data['Projections'])
angles = np.array(data['angles'])

print(projections.shape)

reconstructed_images = myfilteredbackprojection(projections, angles, 512, cutoff)
print(reconstructed_images.shape)

#save numpy
np.save("reconstructed_images_" + str(cutoff) + ".npy", reconstructed_images)

# load numpy for display 
reconstructed_images = np.load("reconstructed_images_" + str(cutoff) + ".npy")

d = DisplayVolume()
# Set the image with default voxel size
voxsz = [1, 1, 1]  # Adjust as per your actual voxel size
d.SetImage(reconstructed_images, voxsz)

# Display the image
d.Display()