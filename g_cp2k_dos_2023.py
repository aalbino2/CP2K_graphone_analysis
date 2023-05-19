#!/home/andrea/.pyenv/bin/python3
"""
Prints the PDOS and the TDOS from the eigenvalues in CP2K file format
for different values of sigma of the gaussian fit.
Data are in eV, Fermi level is set to zero.
"""

from glob import glob
import sys
from linecache import getline
from pylab import loadtxt
import numpy as np
# from scipy.stats import norm
import click
import os

HA2EV = 27.211  # Hartree to eV


def gaussian(center, height, width):
    '''
    Compute a Gaussian curve for a given set of points.

    Parameters
    ----------
    center: float
        The center of the Gaussian curve.
    height: float
        The height of the Gaussian curve.
    width: float
        The width of the Gaussian curve.

    Returns
    -------
    curve: callable
        A callable function that calculates the Gaussian curve for given x values.
    '''
    return lambda x: height * np.exp(-(((center - x) / float(width))**2))


def histogram(data_set, delta_energy, sigma, accuracy):
    '''
    Compute a histogram from a set of data points.

    Parameters
    ----------
    data_set: numpy array
        The data set to be used to compute the histogram.
    delta_energy: float
        The step size for energy integration.
    sigma: float
        The standard deviation of the Gaussian curve.
    accuracy: float
        The accuracy of the Gaussian curve.

    Returns
    -------
    gaussian_mesh: numpy array
        An array of x values for the Gaussian curve.
    gaussian_curve: numpy array
        An array of y values for the Gaussian curve,
        (one column for each orbital + one column for the sum of orbitals).
    energy_min: float
        The minimum energy value.
    energy_max: float
        The maximum energy value.
    '''
    data_set_np = np.array(data_set)
    energy_min = np.min(data_set_np[:, 0])
    energy_max = np.max(data_set_np[:, 0])
    if energy_max - energy_min % delta_energy != 0:
        energy_max = energy_min + (((energy_max - energy_min) // delta_energy) + 1) * delta_energy
    x_mesh = np.arange(energy_min + delta_energy / 2, energy_max, delta_energy)
    y_mesh = np.zeros((data_set_np.shape[1] - 1, x_mesh.shape[0]))
    if ((energy_max - energy_min) % delta_energy):
        energy_max = energy_min + delta_energy * (int((energy_max - energy_min) / delta_energy) + 1)
    print(f"# Number of lines:       {data_set_np.shape[0]}\n"
          f"# Number of columns:      {y_mesh.shape[0]}\n# Integration step:      {delta_energy}\n"
          f"# sigma:                 {sigma}\n# Minimum energy:       {energy_min}\n"
          f"# Maximum energy:        {energy_max}\n")
    print(f"# Number of mesh points: {x_mesh.shape[0]}")
    for i in range(data_set_np.shape[0]):
        eigenvalue = data_set_np[i, :]
        if eigenvalue[0] >= energy_min:
            mesh_index = int((eigenvalue[0] - energy_min) // delta_energy)
            for j in range(data_set_np.shape[1] - 1):
                y_mesh[j, mesh_index] += eigenvalue[j + 1]
    print(f"# Integral area is {np.sum(y_mesh)}")
    gaussian_mesh = np.arange(energy_min, energy_max + delta_energy / accuracy, delta_energy / accuracy)
    print(f"# Number of Gauss mesh points: {gaussian_mesh.shape[0]}")
    gaussian_curve = np.zeros((y_mesh.shape[0] + 1, gaussian_mesh.shape[0]))
    gaussian_curve_sum = np.zeros((1, gaussian_mesh.shape[0]))
    for i in range(y_mesh.shape[0]):
        print("# column ", i + 1, "...")
        for j in range(x_mesh.shape[0] - 1):
            gaussian_curve[i] += gaussian(x_mesh[j], y_mesh[i, j], sigma)(gaussian_mesh)
        gaussian_curve_sum += gaussian_curve[i]
    print("# column ", y_mesh.shape[0] + 1, "...")
    gaussian_curve[y_mesh.shape[0]] = gaussian_curve_sum
    return gaussian_mesh, gaussian_curve, energy_min, energy_max


def get_eigenvalues(data_set, fermi_level):
    """
    Computes the eigenvalues from a .pdos file.
    The values are translated so that the Fermi level is set to zero,
    and they are converted to eV.
    Usage: get_eigenvalues(data_set, fermi_level)
    """
    print(f"# Fermi level: {fermi_level} a.u. ({fermi_level * HA2EV} eV)")
    print("# Generating eigenvalues...")
    sys.stdout.flush()
    num_projections = len(data_set[0]) - 3
    new_eigenvalues = [[(i[1] - fermi_level) * HA2EV] + [i[j] for j in range(3, num_projections + 3)] for i in data_set]
    print(" Done")
    return new_eigenvalues


def compute_dos(input_filename, delta_energy, sigma, accuracy):
    '''
    Returns the DOS from a .pdos file.
    Usage: compute_dos( FileName, delta_energy, sigma, Xmin, Xmax )
    FileName is the name of the input file.
    delta_energy is the width of the intervals.
    sigma is the sigma of the gaussian curve.
    Xmin and Xmax define the range of energies to compute.
    '''
    if getline(input_filename, 1):
        fermi_level = float(getline(input_filename, 1).split(" ")[-2])
    dos_file = f"{input_filename.replace('.pdos', '')}-Dos-{str(sigma)}-{str(delta_energy)}.dat"
    with open(dos_file, 'w', encoding='utf8') as output_file:
        print(f"# Input file: {input_filename}")
        print(f"# Output file: {dos_file}")
        new_eigenvalues = get_eigenvalues(loadtxt(input_filename), fermi_level)
        gaussian_x, gaussian_curves, xmin, xmax = histogram(new_eigenvalues,
                                                            delta_energy, sigma, accuracy)
        output_file.write(f"# sigma={sigma:.6f}, delta_energy={delta_energy:.6f}. Xmin={xmin:.6f}, Xmax={xmax:.6f}\n")
        for index, abscissa_point in enumerate(gaussian_x):
            output_file.write(f"{abscissa_point:.6f}\t")
            for gaussian_curve in gaussian_curves:
                output_file.write(f"{gaussian_curve[index]:.6f}\t")
            output_file.write('\n')


@click.command()
@click.option(
    '-e', '--delta-energy',
    default=0.02,
    help='Energy width for the hystogram. Decreasing the value increses the accuracy of the plot.',
    # required=True,
    # prompt='Your name please',
)
@click.option(
    '-a', '--accuracy',
    default=10,
    help='Accuracy for the gaussian plot'
)
@click.option(
    '-s', '--sigma',
    default=0.1,
    help='Gaussian width.'
)
@click.option(
    '-f', '--filename',
    multiple=True,
    help='Filename. The option can be repeated.'
)
def launch_tool(filename, delta_energy, accuracy, sigma):
    '''
    Main function that launches the tools.
    '''
    if not filename:
        filename = glob('*.pdos')
    elif not len(filename) == len(glob('*.pdos')):
        raise ValueError("\n\nThere are more .pdos files in the folder ! Consider to remove -f arg to grab all of them\n")

    for file in filename:
        compute_dos(file, delta_energy, sigma, accuracy)


if __name__ == '__main__':
    launch_tool().parse()  # pylint: disable=no-value-for-parameter
