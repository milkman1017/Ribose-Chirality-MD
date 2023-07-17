import numpy as np 
import json 
from openmm.unit import *
import glob
import matplotlib.pyplot as ax
from tqdm import tqdm as pbar
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import ripser
import mdtraj as md
from scipy.stats import gaussian_kde
from scipy.fft import fft, ifft
import seaborn as sns
import configparser
import pandas as pd

def get_config():
    config = configparser.ConfigParser()
    config.read('analysis_config.ini')
    return config

def compute_heights(traj, sheet_resnames, mol):
    sheet_atoms = traj.topology.select(f'resn {" or resn ".join(sheet_resnames)}')

    traj_heights = []

    for residue in traj.topology.residues: 
        if residue.name == mol:

            mol_atoms = [atom.index for atom in residue.atoms]
            traj_height = np.mean(md.compute_distances(traj, np.array([[sheet_atom, test_mol_atom] for sheet_atom in sheet_atoms for test_mol_atom in mol_atoms])))
            traj_heights.append(traj_height)

    return traj_heights
    
def graph_heights(heights, config):
    test_resnames = config.get('Input Setup','test resnames').split(',')

    fig, ax = plt.subplots()

    for i, mol_height in enumerate(heights):
        sns.kdeplot(data=mol_height, linewidth=1, label = test_resnames[i])

    plt.legend()
    plt.title('PDF of Distance to Sheet')
    plt.xlabel('Distance to Sheet')
    plt.ylabel('Probability Density')
    plt.show()

def compute_hbonds(chunk, hbond_data, sim_number, config):
    test_resnames = config.get('Input Setup', 'test resnames').split(',')
    sheet_mols = config.get('Input Setup', 'crystal resnames').split(',')

    total_mols = test_resnames + sheet_mols

    for i, frame in enumerate(chunk):
        hbonds = md.baker_hubbard(frame, exclude_water=True)

        for mol_1 in test_resnames:
            for mol_2 in total_mols:
                count = 0  # Initialize the hydrogen bond count for each pair of molecules

                for bond in hbonds:
                    donor, acceptor = bond[0], bond[2]
                    res1, res2 = frame.topology.atom(donor).residue, frame.topology.atom(acceptor).residue

                    if res1.name == mol_1 and res2.name == mol_2:
                        count += 1  # Increment the count if a hydrogen bond is found

                # Assign the count to the corresponding cell in the DataFrame
                hbond_data.loc[i, (sim_number, mol_1, mol_2)] = count

    return hbond_data

def hbond_counts_distribution(hbonds, config):
    test_mols = config.get('Input Setup','test resnames').split(',')
    sheet_mols = config.get('Input Setup','crystal resnames').split(',')
    num_sims = int(config.get('Analyses','number of sims'))

    sheet_hbonds = {}

    #aggregate data from all sims
    for i in range(num_sims):
        for test_mol in test_mols:
            for sheet_mol in sheet_mols:
                
                if (test_mol, sheet_mol) not in sheet_hbonds:
                    sheet_hbonds[test_mol, sheet_mol] = hbonds[i, test_mol, sheet_mol].to_numpy()
                else:
                    sheet_hbonds[test_mol, sheet_mol] = np.append(sheet_hbonds[test_mol, sheet_mol], hbonds[i,test_mol, sheet_mol].to_numpy())
    
    fig, ax = plt.subplots(len(sheet_mols))

    for i, sheet_mol in enumerate(sheet_mols):
        for test_mol in test_mols:
            ax[i].hist(sheet_hbonds[test_mol, sheet_mol], label=test_mol, linewidth=1, bins='auto', histtype='step', density=True)
            ax[i].set_title(f'Hydrogen bonding to residue: {sheet_mol}')
            ax[i].set_xlabel('Number of H bonds')
            ax[i].set_ylabel('Density')

    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_nematic_order(traj, nematic_order, test_resnames, sim_num):
    frames = traj.time

    for i, mol in enumerate(test_resnames):
        mol_atoms = []
        
        for residue in traj.topology.residues:
            if residue.name == mol:
                mol_atoms.append([atom.index for atom in residue.atoms])

        param = md.compute_nematic_order(traj, indices=mol_atoms)
        
        nematic_order[traj.time, i, sim_num] += param

    return nematic_order

def graph_nematic_order(nematic_order, test_resnames):
    nematic_order = np.concatenate(nematic_order, axis=1)
    
    fig, ax = plt.subplots()

    for i, mol in enumerate(test_resnames):

        sns.kdeplot(data=nematic_order[i,:], linewidth=1, label=mol)
    
    plt.title('Nematic Order Distribution')
    plt.xlabel('Nematic Order Parameter')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()


        
def compute_sasa(traj, test_res_names, SASA):
    sasa = md.shrake_rupley(traj, mode='residue')

    for i, mol in enumerate(test_res_names):
        mol_res_indices = []
        for residue in traj.topology.residues:
            if residue.name == mol:
                mol_res_indices.append(residue.index)
    
        mol_sasa = sasa[:,mol_res_indices]

        SASA[i].extend(mol_sasa)

    return SASA

def graph_sasa(SASA, test_resnames):
    fig, ax = plt.subplots()

    for i, mol in enumerate(test_resnames):
        #SASA is a list of num_res arrays. each array is of length num_of_mols X frames so need to concatentate to convert into a 1D array
        mol_sasa = np.concatenate(SASA[i])

        sns.kdeplot(data=mol_sasa, linewidth=1, label=mol)

    plt.yscale('log')
    plt.xlabel('Solvent Accessible Surface Area (nm^2)')
    plt.ylabel('log density')
    plt.title('KDE of SASA')
    plt.legend()
    plt.show()


def main():
    config = get_config()
    sims = int(config.get('Analyses', 'number of sims'))
    nsteps = int(config.get('Input Setup', 'number steps'))
    cell_name = config.get('Input Setup', 'crystal structure')
    sheet_resnames = config.get('Input Setup', 'crystal resnames').split(',')
    test_resnames = config.get('Input Setup', 'test resnames').split(',')
    report = int(config.get('Input Setup', 'report interval'))
    total_mols = test_resnames + sheet_resnames

    file_path = config.get('Input Setup', 'file path')
    outdir = config.get('Output Parameters', 'output directory')

    # these need to be initialized as lists of lists since list lengths are dynamic
    # if the number of each mols are different then an array can't be used as well since the columns would be of different lengths
    heights = [[] for _ in range(len(test_resnames))]
    SASA = [[] for _ in range(len(test_resnames))]

    # initialize hbonds as pandas dataframe
    hbond_columns = pd.MultiIndex.from_product([range(sims), test_resnames, total_mols],
                                               names=['sim_number', 'mol_1', 'mol_2'])
    hbonds = pd.DataFrame(columns=hbond_columns)

    # these can be initialized as arrays as each mol type will give only one output so the size of the array is easily known prior to calculating
    # these give outputs of length one no matter the number of mols so an array works
    nematic_order = np.zeros((int(nsteps / report), len(test_resnames), sims))

    contact_maps = np.zeros((sims, len(test_resnames), len(test_resnames)))

    for sim_number in range(sims):
        print('Analyzing sim number', sim_number)

        traj = md.iterload(f'{file_path}/traj_{sim_number}_sheet_{cell_name[:-4]}_mols_{"_".join(test_resnames)}_steps_{nsteps}.dcd',
                           top=f'{file_path}/traj_{sim_number}_sheet_{cell_name[:-4]}_mols_{"_".join(test_resnames)}_steps_{nsteps}.pdb')

        for chunk in traj:

            if config.get('Analyses', 'height density') == 'True':
                for i, mol in enumerate(test_resnames):
                    heights[i].extend(compute_heights(chunk, sheet_resnames, mol))

            if config.get('Analyses', 'Nematic Order Parameter') == 'True':
                nematic_order = compute_nematic_order(chunk, nematic_order, test_resnames, sim_number)

            if config.get('Analyses', 'Solvent Accessible Surface Area') == 'True':
                SASA = compute_sasa(chunk, test_resnames, SASA)

            if config.get('Analyses', 'H-Bond Counts') == 'True':
                hbonds = compute_hbonds(chunk, hbonds, sim_number, config)
                
    if config.get('Analyses', 'height density') == 'True':
        graph_heights(heights, config)

    if config.get('Analyses', 'Nematic Order Parameter') == 'True':
        graph_nematic_order(nematic_order, test_resnames)

    if config.get('Analyses', 'Solvent Accessible Surface Area') == 'True':
        graph_sasa(SASA, test_resnames)

    if config.get('Analyses', 'H-Bond Counts') == 'True':
        hbond_counts_distribution(hbonds, config)


if __name__ == '__main__':
    main()