#!/home/andrea/.pyenv/bin/python3
'''
Main file of graphone post-processing tool.

This script defines Fermi Energy(beta) = Fermi Energy(alpha).
Optimized geometry is parsed from *-DOS.restart file.
The used pdos file are the unrestricted (ALPHA and BETA) files present in the cwd.

'''

# Test data is in: /home/andrea/Data/Uni/5_DOTTORATO/GRAFENE/
# GRAFONE/GRAFONE+4Au/MACCHIA-200H-4Au-BUFFER/1_old_not converged_THESIS

import os
import re
import sys
import shutil
import subprocess
from glob import glob
from typing import List, Tuple, Dict  # , Union
import numpy as np
import click
# import ase
import matplotlib.pyplot as plt

SCRIPT1 = "gCp2kMergeDos_New.sh"
SCRIPT2 = "g_cp2k_dos_2023.py"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()
NEW_DIR_NAME = 'files'
NEW_DIR = os.path.join(CWD, NEW_DIR_NAME)


def read_xyz(input_file: str) -> Tuple[int, List[str], np.ndarray, np.ndarray]:
    '''
    Reads an XYZ file and returns information about the atoms present in the file.

    Parameters:
    -----------
    input_file : str
        The file path of the XYZ file to be read.

    Returns:
    --------
    Tuple[int, List[str], np.ndarray, np.ndarray]:
        A tuple of the number of atoms (int), the list of atomic symbols (List[str]),
        the coordinates of the atoms (np.ndarray), and the cell parameters (np.ndarray).

    '''
    # atoms = ase.io.read("file.xyz", format="xyz")  # non supporta C1, C2 ecc come atom type
    with open(input_file, "r", encoding='utf8') as file:
        natoms = int(file.readline())
        atom_kinds: List[str] = []
        coords = np.zeros((int(natoms + 1), 3))
        title = file.readline().split()  # TODO fix this second line of xyz
        if len(title) > 8:
            cellpar = np.array([title for i in range(3, 12)])
        else:
            cellpar = np.array([0 for i in range(3, 12)])
        for coord in range(1, natoms + 1):
            line = file.readline().split()
            atom_kinds.append(line[0])
            coords[coord, :] = [line[1], line[2], line[3]]
    return natoms, atom_kinds, coords, cellpar


def write_xyz(output_file, natoms, atom_kinds, coords, cellpar):
    '''
    Write a xyz file.
    Writes coordinates in xyz format.
    '''
    with open(output_file, "w", encoding='utf8') as file:
        file.write(f"{natoms}\n")
        file.write(f"{cellpar}\n")
        for index in range(natoms):
            file.write(f"{atom_kinds[index]} {coords[index,:]}")


def extract_output_file_info(cp2k_output_file: str) -> Tuple[int, int]:
    '''
    Extracts information about the number of steps and last step number,
    as well as convergence criteria, from a CP2K output file.

    Parameters:
    ----------
    cp2k_output_file : str
        The file path of the CP2K output file to be read.

    Returns:
    -------
    Tuple[int, int]:
        A tuple of the number of steps (int) in the file and the last step number (int).
    '''
    with open(cp2k_output_file, 'r', encoding='utf8') as file:
        pattern = " --------  Informations at step =(.*?)---------------------------------------------------"
        file_chunk = re.findall(pattern, file.read(), re.DOTALL)
        steps_in_file = len(file_chunk)
        if file_chunk:
            last_step_no = int(file_chunk[-1].split('\n')[0].split('-')[0])

        # pattern = r'.*(DFT\| Spin unrestricted \(spin-polarized\) Kohn-Sham calculation)(.*)'
        # kohn_sham_type = re.search(pattern, file.read())
            for line in file_chunk[-1].split('\n'):
                for conv_param in ["Convergence in step size",
                                   "Convergence in RMS step",
                                   "Conv. for gradients",
                                   "Conv. in RMS gradients"]:
                    if conv_param in line:
                        if re.search("YES", line):
                            print('         ', line.replace('\n', ' ').strip())
                        else:
                            print('WARNING: ', line.replace('\n', ' ').strip())
        else:
            last_step_no = "{not an optimization, it's a signle point}"
    return steps_in_file, last_step_no  # , kohn_sham_type


def extract_mulliken(cp2k_output_file: str, xyz_file: str, last_step_no: int, atoms_number: int):  # -> np.ndarray:
    '''
    Extracts Mulliken charges from a CP2K output file for the specified
    optimization step and the number of atoms.

    Parameters:
    -----------
    cp2k_output_file : str
        The file path of the CP2K output file to be processed.
    last_step_no : int
        The optimization step number for which to extract the Mulliken charges.
    atoms_number : int
        The number of atoms in the system.

    Returns:
    --------
    np.ndarray:
        An array of the Mulliken charges (np.ndarray).
    '''

    pattern1 = f"OPTIMIZATION STEP:\\s*{last_step_no}(.*?)Informations at step =\\s*{last_step_no}"  # GEO_OPT / CELL_OPT
    pattern2 = f"SCF WAVEFUNCTION OPTIMIZATION(.*?)\\s*Informations at step =\\s*{last_step_no}"  # one-step GEO_OPT / CELL_OPT
    pattern4 = r"SCF WAVEFUNCTION OPTIMIZATION.*?ENERGY\| Total FORCE_EVAL \( QS \) energy \(a\.u\.\):"  # ENERGY_FORCE

    with open(cp2k_output_file, 'r', encoding='utf8') as file:
        rks = re.search(r"DFT\| Spin restricted Kohn-Sham \(RKS\) calculation", file.read())
        file.seek(0)
        uks = re.search(r"DFT\| Spin unrestricted \(spin-polarized\) Kohn-Sham calculation", file.read())
        if rks and not uks:
            sys.stdout.write("\n\n the calculation is RKS! No spin Mulliken file printed\n\n")
        elif uks and not rks:
            sys.stdout.write("\n\n the calculation is UKS! Spin Mulliken file will be printed\n\n")
            file.seek(0)
            file_chunk = re.findall(pattern4, file.read(), re.DOTALL)
            match_found = False
            mulliken_list: List[str] = []
            mulliken = np.zeros((int(atoms_number)))
            carbon_mulliken = np.zeros((int(atoms_number)))
            for line in file_chunk[-1].split('\n'):
                if match_found:
                    mulliken_list.append(line)
                    atoms_number -= 1
                    if atoms_number == 0:
                        break
                elif re.search("Mulliken Population Analysis", line):
                    match_found = True
            for atom in mulliken_list[2:]:
                fields = atom.split()
                if fields[1] in ['Au']:
                    break  # ######### !!! I assume the relevant C and H are placed before the Au in the xyz file - see also further #############à
                if fields[1] in ['C', 'C1', 'C2']:
                    mulliken[int(fields[0])] = float(fields[6])
                    carbon_mulliken[int(fields[0])] = float(fields[6])
                elif fields[1] in ['H', 'F']:
                    mulliken[int(fields[0])] = float(fields[6])
                    carbon_mulliken[int(fields[0])] = float(0)
                else:
                    mulliken[int(fields[0])] = float(0)
                    carbon_mulliken[int(fields[0])] = float(0)
            mulliken_file = f"{cp2k_output_file.split('.restart')[0]}_MULLIKEN.txt"
            with open(mulliken_file, 'w', encoding='utf8') as file:
                file.write("The Mulliken charges are extracted from the last optimization step in:\n")
                file.write(f"{cp2k_output_file}\n")
                file.write("#\n")
                positive = np.sum(mulliken[mulliken > 0])
                negative = np.sum(mulliken[mulliken < 0])
                total = positive + negative
                total_abs_value = positive - negative
                carbon_positive = np.sum(carbon_mulliken[carbon_mulliken > 0])
                carbon_negative = np.sum(carbon_mulliken[carbon_mulliken < 0])
                carbon_total = carbon_positive + carbon_negative
                carbon_total_abs_value = carbon_positive - carbon_negative
                natoms, atom_kinds, _, _ = read_xyz(xyz_file)
                counts: Dict[str, int] = {}
                for item in atom_kinds:
                    if item in counts:
                        counts[item] += 1
                    else:
                        counts[item] = 1
                csp2atoms = 0
                carbon_atoms = 0
                for i in counts.items():
                    if "C" in i[0]:
                        csp2atoms += i[1]
                        carbon_atoms += i[1]
                    if "H" in i[0]:
                        csp2atoms -= i[1]
                    if "Au" in i[0]:
                        break
                ncells = carbon_atoms / 2
                file.write(f"Total number of atoms: {natoms}\n")
                file.write(f"Number of Carbon atoms: {carbon_atoms}\n")
                file.write(f"Number of graphene unit cells: {ncells}\n")
                file.write(f"Number of SP2 Carbon atoms: {csp2atoms}\n")
                file.write(f"Total of Mulliken charges: {total}\n")
                file.write(f"Positive Mulliken charges: {positive}\n")
                file.write(f"Negative Mulliken charges: {negative}\n")
                file.write(f"Total of Mulliken charges ABS value: {total_abs_value}\n")
                file.write(f"Positive Mulliken charges per unit cell: {positive / ncells}\n")
                file.write(f"Negative Mulliken charges per unit cell: {negative / ncells}\n")
                file.write(f"Mulliken charges per unit cell ABS value: {total_abs_value / ncells}\n")
                file.write(f"Positive Mulliken charges per carbon atom: {positive / carbon_atoms}\n")
                file.write(f"Negative Mulliken charges per carbon atom: {negative / carbon_atoms}\n")
                file.write(f"Mulliken charges per carbon atom ABS value: {total_abs_value / carbon_atoms}\n")
                file.write(f"Positive Mulliken charges per SP2 carbon atom: {positive / csp2atoms}\n")
                file.write(f"Negative Mulliken charges per SP2 carbon atom: {negative / csp2atoms}\n")
                file.write(f"Mulliken charges per SP2 carbon atom ABS value: {total_abs_value / csp2atoms}\n")
                file.write("\nAgain the same parameters, counting only Mulliken on CARBON ATOMS\n\n")
                file.write(f"Total number of atoms: {natoms}\n")
                file.write(f"Number of Carbon atoms: {carbon_atoms}\n")
                file.write(f"Number of graphene unit cells: {ncells}\n")
                file.write(f"Number of SP2 Carbon atoms: {csp2atoms}\n")
                file.write(f"Total of Mulliken charges: {carbon_total}\n")
                file.write(f"Positive Mulliken charges: {carbon_positive}\n")
                file.write(f"Negative Mulliken charges: {carbon_negative}\n")
                file.write(f"Total of Mulliken charges ABS value: {carbon_total_abs_value}\n")
                file.write(f"Positive Mulliken charges per unit cell: {carbon_positive / ncells}\n")
                file.write(f"Negative Mulliken charges per unit cell: {carbon_negative / ncells}\n")
                file.write(f"Mulliken charges per unit cell ABS value: {carbon_total_abs_value / ncells}\n")
                file.write(f"Positive Mulliken charges per carbon atom: {carbon_positive / carbon_atoms}\n")
                file.write(f"Negative Mulliken charges per carbon atom: {carbon_negative / carbon_atoms}\n")
                file.write(f"Mulliken charges per carbon atom ABS value: {carbon_total_abs_value / carbon_atoms}\n")
                file.write(f"Positive Mulliken charges per SP2 carbon atom: {carbon_positive / csp2atoms}\n")
                file.write(f"Negative Mulliken charges per SP2 carbon atom: {carbon_negative / csp2atoms}\n")
                file.write(f"Mulliken charges per SP2 carbon atom ABS value: {carbon_total_abs_value / csp2atoms}\n")
            # return mulliken
        else:
            print("\n\n the calculation is neither RKS nor UKS! CHECK CP2K OUTPUT FILE \n\n")


def extract_restart_file_info(cp2k_restart_file: str) -> Tuple[int, int]:
    '''
    Extracts information about a CP2K restart file.

    Parameters:
    -----------
    cp2k_restart_file : str
        The file path of the CP2K restart file to be processed.

    Returns:
    --------
    Tuple[int, int]:
        A tuple of the restart step number (int) and the number of atoms in the simulation (int).

    '''
    with open(cp2k_restart_file, 'r', encoding='utf8') as restart_file:
        for line in restart_file:
            if re.search("RUN_TYPE", line):
                if "GEO_OPT" in line:
                    pattern = "&GEO_OPT.*\n.*&END GEO_OPT"
                    file_chunk = re.findall(pattern, restart_file.read(), re.DOTALL)[0]
                elif "CELL_OPT" in line:
                    pattern = "&CELL_OPT.*\n.*&END CELL_OPT"
                    file_chunk = re.findall(pattern, restart_file.read(), re.DOTALL)[0]
                elif "ENERGY_FORCE" in line:
                    sys.stdout.write('\nIT IS AN ENERGY_FORCE SINGLE POINT\n')
                    file_chunk = "ENERGY_FORCE"
                else:
                    raise Exception('GEO_OPT or CELL_OPT or ENERGY_FORCE not found')
                if re.search("STEP_START_VAL", file_chunk):
                    restart_step_no = int(file_chunk.split("STEP_START_VAL")[1].split('\n')[0])
                else:
                    restart_step_no = 1
        restart_file.seek(0)
        for line in restart_file:
            if re.search("NUMBER_OF_ATOMS", line):
                atoms_number = int(line.split("NUMBER_OF_ATOMS")[1])
        return restart_step_no, atoms_number


def extract_optimized_geo(input_file, pattern, lines_no, output_file):
    '''
    Extract `lines_no` lines after a certain matched string from an input file
    and write the extracted lines to an output file.

    Parameters:
    -----------
    input_file : str
        The file path of the input file.
    pattern : str
        The pattern to search for in the input file.
    lines_no : int
        The number of lines to extract after the pattern.
    output_file : str
        The file path of the output file.

    '''
    with open(input_file, 'r', encoding='utf8') as restart_file:
        match_found = False
        with open(output_file, 'w', encoding='utf8') as opt_xyz_file:
            opt_xyz_file.write(f"{lines_no}\n")
            opt_xyz_file.write(' \n')
            for line in restart_file:
                if match_found:
                    opt_xyz_file.write(line)
                    lines_no -= 1
                    if lines_no == 0:
                        break
                elif re.search(pattern, line):
                    match_found = True


def validate_files(cp2k_output_filename, cp2k_restart_filename):  # -> str:
    '''
    Validate the CP2K output and restart files and extract optimized geometry from the restart file.

    Parameters:
    -----------
    cp2k_output_filename : str
        The file path of the CP2K output file.
    cp2k_restart_filename : str
        The file path of the CP2K restart file.

    Raises:
    -------
    AssertionError: If the last step number of the optimization in the output file and
    the step number in the restart file are discordant.

    '''
    if not os.path.exists(NEW_DIR):
        os.makedirs(NEW_DIR)
        sys.stdout.write(f"Directory '{NEW_DIR_NAME}' created successfully\n")
    else:
        sys.stdout.write(f"Directory '{NEW_DIR_NAME}' already exists\n")
    cp2k_output_file = os.path.join(NEW_DIR, cp2k_output_filename)
    cp2k_restart_file = os.path.join(NEW_DIR, cp2k_restart_filename)
    shutil.copyfile(cp2k_output_filename, cp2k_output_file)
    shutil.copyfile(cp2k_restart_filename, cp2k_restart_file)
    sys.stdout.write("------ START check of .out and .restart files ------\n")
    steps_in_file, last_step_no = extract_output_file_info(cp2k_output_file)
    sys.stdout.write(f"Optimization steps performed in this .out file:   {steps_in_file}\n")
    sys.stdout.write(f"Last optimization step:   no. {last_step_no}\n")
    restart_step_no, atoms_number = extract_restart_file_info(cp2k_restart_file)
    if last_step_no != restart_step_no:
        sys.stdout.write('\n!!! WARNING !!!\n')
        sys.stdout.write('Restart and Output file have discordant info')
        sys.stdout.write(f'(last step is no. {restart_step_no} and no. {last_step_no}, respectively)\n')
        sys.stdout.write('!!! !!! !!! !!!\n\n')
    else:
        sys.stdout.write('Restart and Output file have same last step index')
    opt_xyz_file = f"{cp2k_restart_file.split('.restart')[0]}_OPT.xyz"
    extract_optimized_geo(cp2k_restart_file, "&COORD", atoms_number, opt_xyz_file)
    extract_mulliken(cp2k_output_file, opt_xyz_file, last_step_no, atoms_number)
    # _, atom_kinds, coords_opt, cellpar_opt = read_xyz(opt_xyz_file)
    sys.stdout.write("------ END check of .out and .restart files ------\n")
    # return str(kohn_sham_type).strip()


def g_cp2k_merge_dos() -> dict:
    '''
    Merge the alpha and beta partial density of states (PDOS) files and return
    a dictionary with the resulting merged files and their type.

    Returns:
    --------
    dict:
        A dictionary with the keys 'alpha', 'beta' and 'merged' containing
        the respective lists of filenames for each type.

    '''
    found_files: Dict[str, List[str]] = {'alpha': [], 'beta': [], 'merged': []}
    if not os.path.isdir(NEW_DIR):
        os.mkdir(NEW_DIR)
    restricted_suffix = re.compile('-k[0-9]{1}-[0-9]{1}.pdos')
    unrestricted_suffix = re.compile('-ALPHA_k[0-9]{1}-[0-9]{1}.pdos|-BETA_k[0-9]{1}-[0-9]{1}.pdos')
    for file in [f for f in os.listdir(NEW_DIR) if os.path.isfile(os.path.join(NEW_DIR, f))]:
        if re.search(restricted_suffix, file):
            os.remove(os.path.join(NEW_DIR, file))
    for file in [f for f in os.listdir(CWD) if os.path.isfile(os.path.join(CWD, f))]:
        if re.search(unrestricted_suffix, file):
            suffix = re.search(unrestricted_suffix, file).group(0)
            clean_filename = file.split(suffix)[0].replace('-', '') + suffix
            if '-ALPHA_k' in clean_filename:
                found_files['alpha'].append(clean_filename)
            elif '-BETA_k' in clean_filename:
                found_files['beta'].append(clean_filename)
            shutil.copyfile(os.path.join(CWD, file), os.path.join(NEW_DIR, clean_filename))
            sys.stdout.write(f" -> copied {clean_filename} in \n {NEW_DIR_NAME} folder \n")
        # found_files['alpha'].sort()
        # found_files['beta'].sort()
    sys.stdout.write(f"------ START {SCRIPT1} ------\n")
    # shutil.copyfile(SCRIPT1, os.path.join(NEW_DIR, SCRIPT1))
    subprocess.run(os.path.join(SCRIPT_DIR, SCRIPT1), shell=True, cwd=NEW_DIR, check=True)
    sys.stdout.write(f"------ END {SCRIPT1} ------\n")
    for file in [f for f in os.listdir(CWD) if os.path.isfile(os.path.join(CWD, f))]:
        if re.search(restricted_suffix, file):
            suffix = re.search(restricted_suffix, file).group(0)
            clean_filename = file.split(suffix)[0].replace('-', '') + suffix
            found_files['merged'].append(clean_filename)
    # found_files['merged'].sort()
    return found_files


def build_atomkind_dict():
    '''
    Build a dictionary useful for plotting the data
    '''
    found_files: Dict[str, List[str]] = {'alpha': [], 'beta': [], 'merged': [], 'kind': [], 'orbitals': []}
    unrestricted_suffix_dat = re.compile('-ALPHA_k[0-9]{1}-[0-9]{1}-Dos-|-BETA_k[0-9]{1}-[0-9]{1}-Dos-')
    restricted_suffix_dat = re.compile('-k[0-9]{1}-[0-9]{1}-Dos-')
    unrestricted_suffix_pdos = re.compile('-ALPHA_k[0-9]{1}-[0-9]{1}.pdos|-BETA_k[0-9]{1}-[0-9]{1}.pdos')
    restricted_suffix_pdos = re.compile('-k[0-9]{1}-[0-9]{1}.pdos')
    for file_dat in [f for f in os.listdir(NEW_DIR) if os.path.isfile(os.path.join(NEW_DIR, f))]:
        for file_pdos in [f for f in os.listdir(NEW_DIR) if os.path.isfile(os.path.join(NEW_DIR, f))]:
            if re.search(unrestricted_suffix_dat, file_dat) \
                and re.search(unrestricted_suffix_pdos, file_pdos) \
                and file_pdos.replace('.pdos', '') == file_dat.split('-Dos-')[0] \
                and '.png' not in file_dat:
                if '-ALPHA_k' in file_dat:
                    found_files['alpha'].append(file_dat)
                    with open(os.path.join(NEW_DIR, file_pdos), 'r', encoding='utf8') as alpha_file:
                        line = alpha_file.readline()
                        detected_kind = line.split('atomic kind')[1].split('at iteration')[0].strip()
                        found_files['kind'].append(detected_kind)
                        line = alpha_file.readline()
                        detected_orbitals = ['Energy']
                        detected_orbitals += line.split('Occupation')[1].split()
                        detected_orbitals += ['Tot']
                        found_files['orbitals'].append(detected_orbitals)
                elif '-BETA_k' in file_dat:
                    found_files['beta'].append(file_dat)
            if re.search(restricted_suffix_dat, file_dat) \
                    and re.search(restricted_suffix_pdos, file_pdos) \
                    and file_pdos.replace('.pdos', '') == file_dat.split('-Dos-')[0] \
                    and '.png' not in file_dat:
                found_files['merged'].append(file_dat)
    sorted_found_files = {}
    for key, value in found_files.items():
        if key in ['kind', 'orbitals']:
            sorted_found_files[key] = [v for _, v in sorted(zip(found_files['alpha'], value))]
        else:
            sorted_found_files[key] = sorted(value)
    return sorted_found_files


def check_fermi_energy(unrestricted_files: Dict[str, List[str]]) -> None:
    '''
    Sets Fermi Energy equal to the one in the ALPHA file.

    Parameters:
    -----------
    unrestricted_files : Dict[str, List[str]]
        A dictionary containing the names of the ALPHA and BETA files. The keys of the dictionary
        should be 'alpha' and 'beta', the values should be lists of the corresponding file names.

    '''
    sys.stdout.write("------ START check_fermi_energy ------\n")
    for index, _ in enumerate(unrestricted_files['alpha']):
        with open(os.path.join(NEW_DIR, unrestricted_files['alpha'][index]),
                  'r', encoding='utf8') as file:
            line = file.readline()  # line = file.readlines()[0]
            fermi_alpha = line.split('E(Fermi) =')[1].split('a.u.')[0].strip()
        with open(os.path.join(NEW_DIR, unrestricted_files['beta'][index]),
                  'r', encoding='utf8') as file:
            line = file.readline()
            fermi_beta = line.split('E(Fermi) =')[1].split('a.u.')[0].strip()
        if fermi_alpha != fermi_beta:
            with open(os.path.join(NEW_DIR, unrestricted_files['beta'][index]),
                      'r+', encoding='utf8') as file:
                old_fermi = file.read()
                new_fermi = old_fermi.replace(fermi_beta, fermi_alpha)
                file.write(new_fermi)
                sys.stdout.write(f"E Fermi = (changed) {fermi_beta} -> {fermi_alpha} "
                                 f"in {unrestricted_files['beta'][index]} (path: {NEW_DIR_NAME})\n")
        else:
            sys.stdout.write(f"E Fermi = {fermi_alpha} "
                             f"in {unrestricted_files['alpha'][index]} (path: {NEW_DIR_NAME})\n")
    sys.stdout.write("------ END check_fermi_energy ------\n")


def plot_data(atomkind_files):

    dpi = 600

    linestyles_alpha = [
        "#000000",
        "#8F0D0D",  # Red
        "#B52626",
        "#D34747",
        "#FF8E00",
        "#FF9F00",  # Yellow
        "#FFAE00",
        "#FFBC00",
        "#FFCA00",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#000000"
    ]
    linestyles_beta = [
        "#000000",
        "#0014D3",  # Red
        "#0029D1",
        "#003BD0",
        "#0070CC",
        "#0084CB",  # Yellow
        "#009DC9",
        "#00BEC7",
        "#00CAAE",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#1B9A00",
        "#000000"
    ]

    data_alpha_c1 = None
    data_alpha_c2 = None
    data_alpha_bs = None

    for index, atomkind in enumerate(atomkind_files['kind']):
        if atomkind == 'C':
            data_alpha_c = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['alpha'][index]))
            data_beta_c = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['beta'][index]))
            data_c = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['merged'][index]))
        if atomkind == 'C1':
            data_alpha_c1 = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['alpha'][index]))
            data_beta_c1 = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['beta'][index]))
            data_c1 = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['merged'][index]))
        if atomkind == 'C2':
            data_alpha_c2 = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['alpha'][index]))
            data_beta_c2 = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['beta'][index]))
            data_c2 = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['merged'][index]))
    if np.any(data_alpha_c1) and not np.any(data_alpha_c2):
        alpha_bs = data_alpha_c[:, 1:] + data_alpha_c1[:, 1:]
        beta_bs = data_beta_c[:, 1:] + data_beta_c1[:, 1:]
        bs = data_c[:, 1:] + data_c1[:, 1:]
        data_alpha_bs = np.concatenate((data_alpha_c[:, 0].reshape(-1, 1), alpha_bs), axis=1)
        data_beta_bs = np.concatenate((data_beta_c[:, 0].reshape(-1, 1), beta_bs), axis=1)
        data_bs = np.concatenate((data_c[:, 0].reshape(-1, 1), bs), axis=1)
    if not np.any(data_alpha_c1) and np.any(data_alpha_c2):
        data_alpha_bs = data_alpha_c[:, 1:] + data_alpha_c2[:, 1:]
        data_beta_bs = data_beta_c[:, 1:] + data_beta_c2[:, 1:]
        data_bs = data_c[:, 1:] + data_c2[:, 1:]
    if np.any(data_alpha_c1) and np.any(data_alpha_c2):
        alpha_bs = data_alpha_c[:, 1:] + data_alpha_c1[:, 1:] + data_alpha_c2[:, 1:]
        beta_bs = data_beta_c[:, 1:] + data_beta_c1[:, 1:] + data_beta_c2[:, 1:]
        bs = data_c[:, 1:] + data_c1[:, 1:] + data_c2[:, 1:]
        data_alpha_bs = np.concatenate((data_alpha_c[:, 0].reshape(-1, 1), alpha_bs), axis=1)
        data_beta_bs = np.concatenate((data_beta_c[:, 0].reshape(-1, 1), beta_bs), axis=1)
        data_bs = np.concatenate((data_c[:, 0].reshape(-1, 1), bs), axis=1)

    for index, atomkind in enumerate(atomkind_files['kind']):
        data_alpha = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['alpha'][index]))
        data_beta = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['beta'][index]))
        # Create the figure and axes objects
        fig = plt.figure(figsize=(4, 3))
        alpha_beta_plot = fig.add_subplot(111)
        plt.xlim(-7.5, 7.5)
        # Get the current tick positions and labels
        ticks, labels = plt.xticks()
        # Exclude the extreme tick positions and labels
        ticks = ticks[1:-1]
        labels = labels[1:-1]
        # Set the modified ticks and labels
        plt.xticks(ticks, labels)
        # Set the axis labels and title
        alpha_beta_plot.set_xlabel('Energy (eV)')
        alpha_beta_plot.set_ylabel('Pdos (a. u.)')
        alpha_beta_plot.set_title(f"Pdos for atom kind {atomkind}")
        # Customize the tick marks and grid lines
        alpha_beta_plot.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        alpha_beta_plot.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # Plot the data
        for orbital in range((data_alpha.shape[1] - 1), 0, -1):
            inverted_index = (data_alpha.shape[1] - 1 - orbital)
            red = int(linestyles_alpha[inverted_index][1:3], 16) / 255
            green = int(linestyles_alpha[inverted_index][3:5], 16) / 255
            blue = int(linestyles_alpha[inverted_index][5:7], 16) / 255
            alpha_beta_plot.plot(data_alpha[:, 0], data_alpha[:, orbital],
                                 linestyle='-', linewidth=1,
                                 color=(red, green, blue),
                                 label=f"{atomkind_files['orbitals'][index][orbital]}")
        for orbital in range((data_beta.shape[1] - 1), 0, -1):
            inverted_index = (data_beta.shape[1] - 1 - orbital)
            red = int(linestyles_beta[inverted_index][1:3], 16) / 255
            green = int(linestyles_beta[inverted_index][3:5], 16) / 255
            blue = int(linestyles_beta[inverted_index][5:7], 16) / 255
            alpha_beta_plot.plot(data_beta[:, 0], -data_beta[:, orbital],
                                 linestyle='-', linewidth=1,
                                 color=(red, green, blue),
                                 label=f"{atomkind_files['orbitals'][index][orbital]}")
        # Add a legend
        alpha_beta_plot.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., handlelength=0.5)
        # Save as PNG
        fig.savefig(f"{os.path.join(NEW_DIR, atomkind_files['alpha'][index]).replace('ALPHA','alpha-beta')}.png", dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    for index, atomkind in enumerate(atomkind_files['kind']):
        data = np.loadtxt(os.path.join(NEW_DIR, atomkind_files['merged'][index]))
        # Create the figure and axes objects
        fig = plt.figure(figsize=(4, 3))
        merged_plot = fig.add_subplot(111)
        plt.xlim(-10, 10)
        # Get the current tick positions and labels
        # ticks, labels = plt.xticks()
        # Exclude the extreme tick positions and labels
        # ticks = ticks[1:-1]
        # labels = labels[1:-1]
        # # Set the modified ticks and labels
        # plt.xticks(ticks, labels)
        # Set the axis labels and title
        merged_plot.set_xlabel('Energy (eV)')
        merged_plot.set_ylabel('Pdos (a. u.)')
        merged_plot.set_title(f"Pdos for atom kind {atomkind}")
        # Customize the tick marks and grid lines
        merged_plot.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        merged_plot.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # Plot the data
        for orbital in range((data.shape[1] - 1), 0, -1):
            inverted_index = (data.shape[1] - 1 - orbital)
            red = int(linestyles_alpha[inverted_index][1:3], 16) / 255
            green = int(linestyles_alpha[inverted_index][3:5], 16) / 255
            blue = int(linestyles_alpha[inverted_index][5:7], 16) / 255
            merged_plot.plot(data[:, 0], data[:, orbital], linestyle='-', linewidth=1, color=(red, green, blue), label=f"{atomkind_files['orbitals'][index][orbital]}")
        # Add a legend
        merged_plot.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., handlelength=0.5)
        # Save as PNG
        fig.savefig(f"{os.path.join(NEW_DIR, atomkind_files['merged'][index])}.png", dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    if np.any(data_alpha_bs):
        for index, atomkind in enumerate(atomkind_files['kind']):
            if atomkind == 'C':
                carbon_index = index
        # Create the figure and axes objects
        fig = plt.figure(figsize=(4, 3))
        bs_plot = fig.add_subplot(111)
        plt.xlim(-7.5, 7.5)
        # Get the current tick positions and labels
        ticks, labels = plt.xticks()
        # Exclude the extreme tick positions and labels
        ticks = ticks[1:-1]
        labels = labels[1:-1]
        # Set the modified ticks and labels
        plt.xticks(ticks, labels)
        # Set the axis labels and title
        bs_plot.set_xlabel('Energy (eV)')
        bs_plot.set_ylabel('Pdos (a. u.)')
        bs_plot.set_title(f"Pdos for atom kind {atomkind_files['kind'][carbon_index]} (BS)")
        # Customize the tick marks and grid lines
        bs_plot.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        bs_plot.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # Plot the data
        for orbital in range((data_alpha_bs.shape[1] - 1), 0, -1):
            inverted_index = (data_alpha_bs.shape[1] - 1 - orbital)
            red = int(linestyles_alpha[inverted_index][1:3], 16) / 255
            green = int(linestyles_alpha[inverted_index][3:5], 16) / 255
            blue = int(linestyles_alpha[inverted_index][5:7], 16) / 255
            bs_plot.plot(data_alpha_bs[:, 0], data_alpha_bs[:, orbital], linestyle='-', linewidth=1, color=(red, green, blue), label=f"{atomkind_files['orbitals'][carbon_index][orbital]}")
        for orbital in range((data_beta_bs.shape[1] - 1), 0, -1):
            inverted_index = (data_beta_bs.shape[1] - 1 - orbital)
            red = int(linestyles_beta[inverted_index][1:3], 16) / 255
            green = int(linestyles_beta[inverted_index][3:5], 16) / 255
            blue = int(linestyles_beta[inverted_index][5:7], 16) / 255
            bs_plot.plot(data_beta_bs[:, 0], -data_beta_bs[:, orbital], linestyle='-', linewidth=1, color=(red, green, blue), label=f"{atomkind_files['orbitals'][carbon_index][orbital]}")
        # Add a legend
        bs_plot.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., handlelength=0.5)
        # Save as PNG
        fig.savefig(f"{os.path.join(NEW_DIR, atomkind_files['alpha'][carbon_index]).replace('ALPHA','')}_BS.png", dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

        # Create the figure and axes objects
        fig = plt.figure(figsize=(4, 3))
        bs_plot_merged = fig.add_subplot(111)
        plt.xlim(-10, 10)
        # # Get the current tick positions and labels
        # ticks, labels = plt.xticks()
        # # Exclude the extreme tick positions and labels
        # ticks = ticks[1:-1]
        # labels = labels[1:-1]
        # # Set the modified ticks and labels
        # plt.xticks(ticks, labels)
        # # Set the axis labels and title
        bs_plot_merged.set_xlabel('Energy (eV)')
        bs_plot_merged.set_ylabel('Pdos (a. u.)')
        bs_plot_merged.set_title(f"Pdos for atom kind {atomkind}")
        # Customize the tick marks and grid lines
        bs_plot_merged.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        bs_plot_merged.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # Plot the data
        for orbital in range((data_bs.shape[1] - 1), 0, -1):
            inverted_index = (data_bs.shape[1] - 1 - orbital)
            red = int(linestyles_alpha[inverted_index][1:3], 16) / 255
            green = int(linestyles_alpha[inverted_index][3:5], 16) / 255
            blue = int(linestyles_alpha[inverted_index][5:7], 16) / 255
            bs_plot_merged.plot(data_bs[:, 0], data_bs[:, orbital], linestyle='-', linewidth=1, color=(red, green, blue), label=f"{atomkind_files['orbitals'][carbon_index][orbital]}")
        # Add a legend
        bs_plot_merged.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., handlelength=0.5)
        # Save as PNG
        fig.savefig(f"{os.path.join(NEW_DIR, atomkind_files['merged'][carbon_index])}_BS.png", dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close(fig)


@click.command()
@click.option(
    '-s', '--sigma',
    # required=True,
    default=0.1,
    # prompt='Your name please',
    help='Gaussian width.'
)
@click.option(
    '-f', '--filename',
    multiple=True,
    help='Filename. The option can be repeated.'
)
@click.option(
    '-o', '--output',
    help='Output filename. The file where the last optimization step is stored.'
)
@click.option(
    '-r', '--restart',
    help='Restart filename. The file where the last optimization step is stored.'
)
@click.option(
    '-d', '--dos-process',
    is_flag=True,
    help='Go through the DOS processing only if the flag is present.'
)
def launch_tool(sigma, filename, output, dos_process, restart):
    '''
    Main function that launches the tools.
    '''
    if output and restart:
        validate_files(output, restart)
    else:
        raise ValueError("Either output file or restart file were not provided via -o and -r args")
    atomkind_pdos_files = g_cp2k_merge_dos()
    check_fermi_energy(atomkind_pdos_files)

    if dos_process:
        sys.stdout.write(f"------ START {SCRIPT2} ------\n")
        # shutil.copyfile(SCRIPT2, os.path.join(NEW_DIR, SCRIPT2))
        if filename:
            subprocess.run(f"python {os.path.join(SCRIPT_DIR, SCRIPT2)} -s {sigma} -f {filename[0]}", cwd=NEW_DIR,
                           check=True, text=True, shell=True)
        else:
            subprocess.run(f"python {os.path.join(SCRIPT_DIR, SCRIPT2)} -s {sigma}", cwd=NEW_DIR,
                           check=True, text=True, shell=True)
        sys.stdout.write(f"------ END {SCRIPT2} ------\n")
    else:
        sys.stdout.write(f"------ {SCRIPT2} was skipped ------\n")

    sys.stdout.write("------ START PLOTTING ------\n")
    atomkind_dat_files = build_atomkind_dict()
    plot_data(atomkind_dat_files)
    sys.stdout.write("------  END PLOTTING  ------\n")


if __name__ == '__main__':
    launch_tool().parse()  # pylint: disable=no-value-for-parameter
