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

def compute_hbonds(chunk, hbond_counts):

    D_G,D_C,D_B,L_G,L_C,L_B,D_D,D_L,L_L = [],[],[],[],[],[],[],[],[]

    for frame in chunk:
        DG = 0
        DC = 0
        LG = 0
        LC = 0
        DD = 0
        DL = 0
        LL = 0
        hbonds = md.baker_hubbard(frame,exclude_water=True)
        
        #get hbond counts 
        for hbond in hbonds:
            atom1, atom2 = hbond[0], hbond[2]
            res1, res2 = frame.topology.atom(atom1).residue, frame.topology.atom(atom2).residue
            atom1_index, atom2_index = frame.topology.atom(atom1).index % res1.n_atoms, frame.topology.atom(atom2).index % res2.n_atoms

            if (res1.name == 'G' and res2.name == 'DRI') or (res1.name == 'DRI' and res2.name == 'G'):
                DG += 1
            elif (res1.name == 'C' and res2.name == 'DRI') or (res1.name == 'DRI' and res2.name == 'C'):
                DC+= 1
            elif (res1.name == 'G' and res2.name == 'LRI') or (res1.name == 'LRI' and res2.name == 'G'):
                LG += 1
            elif (res1.name == 'C' and res2.name == 'LRI') or (res1.name == 'LRI' and res2.name == 'C'):
                LC += 1
            elif (res1.name == 'DRI' and res2.name == 'DRI'):
                DD += 1
            elif (res1.name == 'DRI' and res2.name == 'LRI') or (res1.name == 'LRI' and res2.name == 'DRI'):
                DL += 1
            elif (res1.name == 'LRI' and res2.name == 'LRI'):
                LL += 1

            hbond_key = f'{res1.name}-{res2.name}'
            hbond_count_key = f"{atom1_index}-{atom2_index}"

            if hbond_key not in hbond_counts:
                hbond_counts[hbond_key] = dict()
            if hbond_count_key not in hbond_counts[hbond_key]:
                hbond_counts[hbond_key][hbond_count_key]=0
            hbond_counts[hbond_key][hbond_count_key] += 1

        D_G.append(DG)
        D_C.append(DC)
        D_B.append(DG+DC)
        L_G.append(LG)
        L_C.append(LC)
        L_B.append(LG+LC)
        D_D.append(DD)
        D_L.append(DL)
        L_L.append(LL)
            
    return hbond_counts, D_G, D_C, D_B, L_G, L_C, L_B, D_D, D_L, L_L

def ribose_label_sort(item):
    if item.startswith('DRI'):
        return (0, int(item[3:]))
    elif item.startswith('LRI'):
        return (1, int(item[3:]))
    elif item.startswith('C'):
        return (2, int(item[1:]))
    elif item.startswith('G'):
        return (3, int(item[1:]))
    else:
        return (4, int(item[2:]))

def hbond_heatmap(hbond_counts):

    dribose_donor_labels = set()
    dribose_acceptor_labels = set()

    lribose_donor_labels = set()
    lribose_acceptor_labels = set()

    hbonds = hbond_counts

    for residue, bond_dict in hbonds.items():
        if residue == 'DRI-G' or residue == 'DRI-C' or residue == 'G-DRI' or residue == 'C-DRI':
            donor_residue, acceptor_residue = residue.split('-')
            for atom in bond_dict.keys():
                dribose_donor_labels.add(f"{donor_residue}-{atom.split('-')[0]}")
                dribose_acceptor_labels.add(f"{acceptor_residue}-{atom.split('-')[1]}")

        elif residue == 'LRI-C' or residue == 'LRI-G' or residue == 'G-LRI' or residue == 'C-LRI':
            donor_residue, acceptor_residue = residue.split('-')
            for atom in bond_dict.keys():
                lribose_donor_labels.add(f"{donor_residue}-{atom.split('-')[0]}")
                lribose_acceptor_labels.add(f"{acceptor_residue}-{atom.split('-')[1]}")

    dribose_donor_labels = sorted(dribose_donor_labels, key=ribose_label_sort)
    dribose_acceptor_labels = sorted(dribose_acceptor_labels, key=ribose_label_sort,reverse=True)

    lribose_donor_labels = sorted(lribose_donor_labels,key=ribose_label_sort)
    lribose_acceptor_labels = sorted(lribose_acceptor_labels, key=ribose_label_sort,reverse=True)
    
    dribose_bond_data = np.zeros((len(dribose_donor_labels), len(dribose_acceptor_labels)))
    lribose_bond_data = np.zeros((len(lribose_donor_labels), len(lribose_acceptor_labels)))
    
    for residue, bond_dict in hbonds.items():
        if residue == 'DRI-G' or residue == 'DRI-C' or residue == 'G-DRI' or residue == 'C-DRI':
            donor_residue, acceptor_residue = residue.split('-')

            for atom, count in bond_dict.items():
                donor, acceptor = atom.split('-')
                donor_label = f"{donor_residue}-{donor}"
                acceptor_label = f"{acceptor_residue}-{acceptor}"
                donor_index = dribose_donor_labels.index(donor_label)
                acceptor_index = dribose_acceptor_labels.index(acceptor_label)
                dribose_bond_data[donor_index, acceptor_index] += count

        elif residue == 'LRI-C' or residue == 'LRI-G' or residue == 'G-LRI' or residue == 'C-LRI':
            donor_residue, acceptor_residue = residue.split('-')
            for atom, count in bond_dict.items():
                donor, acceptor = atom.split('-')
                donor_label = f"{donor_residue}-{donor}"
                acceptor_label = f"{acceptor_residue}-{acceptor}"
                donor_index = lribose_donor_labels.index(donor_label)
                acceptor_index = lribose_acceptor_labels.index(acceptor_label)
                lribose_bond_data[donor_index, acceptor_index] += count

    fig, ax = plt.subplots(1,2)
    im1 = ax[0].imshow(dribose_bond_data, cmap="hot")

    ax[0].set_xticks(np.arange(len(dribose_acceptor_labels)))
    ax[0].set_xlabel('Acceptors')
    ax[0].set_yticks(np.arange(len(dribose_donor_labels)))
    ax[0].set_xticklabels(dribose_acceptor_labels)
    ax[0].set_yticklabels(dribose_donor_labels)
    ax[0].set_ylabel('Donors')
    ax[0].set_title('D-Ribose Contact Map')

    im2 = ax[1].imshow(lribose_bond_data, cmap='hot')
    ax[1].set_xticks(np.arange(len(lribose_acceptor_labels)))
    ax[1].set_xlabel('Acceptors')
    ax[1].set_yticks(np.arange(len(lribose_donor_labels)))
    ax[1].set_xticklabels(lribose_acceptor_labels)
    ax[1].set_yticklabels(lribose_donor_labels)
    ax[1].set_ylabel('Donors')
    ax[1].set_title('L-Ribose Contact Map')

    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    plt.tight_layout()
    plt.suptitle('Hydrogen Bond Heat Map')
    plt.show()

def hbond_order(D_G,D_C,D_B,L_G,L_C,L_B,D_D,D_L,L_L):
    D_G = np.mean(D_G,axis=0)
    D_C = np.mean(D_C, axis=0)
    D_B = np.mean(D_B, axis=0)
    L_G = np.mean(L_G, axis=0)
    L_C = np.mean(L_C, axis=0)
    L_B = np.mean(L_B, axis=0)
    D_D = np.mean(D_D, axis=0)
    D_L = np.mean(D_L, axis=0)
    L_L = np.mean(L_L, axis=0)

    time = np.arange(len(D_G)) * 0.004

    fig, ax = plt.subplots(3,3)
    ax[0,0].plot(time, D_G, linewidth=1, color='b', label='D-Ribose')
    ax[0,0].plot(time, L_G, linewidth=1, color='r', label='L-Ribose')
    ax[0,0].set_title('Guanine H-Bonds')
    ax[0,0].set_xlabel('Time (ns)')
    ax[0,0].set_ylabel('Count')

    ax[1,0].plot(time, D_C, linewidth=1, color='b', label='D-Ribose')
    ax[1,0].plot(time, L_C, linewidth=1, color='r', label='L-Ribose')
    ax[1,0].set_title('Cytosine H-Bonds')
    ax[1,0].set_xlabel('Time (ns)')
    ax[1,0].set_ylabel('Count')

    ax[2,0].plot(time, D_B, linewidth=1, color='b', label='D-Ribose')
    ax[2,0].plot(time, L_B, linewidth=1, color='r', label='L-Ribose')
    ax[2,0].set_title('Guanine and Cytosine H-Bonds')
    ax[2,0].set_xlabel('Time (ns)')
    ax[2,0].set_ylabel('Count')


    ax[0,1].hist(D_G, histtype='step', density=True, bins='auto', color='b', label='D-Ribose')
    ax[0,1].hist(L_G, histtype='step', density=True, bins='auto', color='r', label='L-Ribose')
    ax[0,1].set_title('Distribution of Guanine H-Bonds')
    ax[0,1].set_xlabel('Number of H-Bonds')

    ax[1,1].hist(D_C, histtype='step', density=True, bins='auto', color='b', label='D-Ribose')
    ax[1,1].hist(L_C, histtype='step', density=True, bins='auto', color='r', label='L-Ribose')
    ax[1,1].set_title('Distribution of Cytosine H-Bonds')
    ax[1,1].set_xlabel('Number of H-Bonds')

    ax[2,1].hist(D_B, histtype='step', density=True, bins='auto', color='b', label='D-Ribose')
    ax[2,1].hist(L_B, histtype='step', density=True, bins='auto', color='r', label='L-Ribose')
    ax[2,1].set_title('Distribution of Sheet H-Bonds')
    ax[2,1].set_xlabel('Number of H-Bonds')

    ax[0,2].plot(time, D_D, linewidth=1, color='b', label='D-Ribose')
    ax[0,2].plot(time, L_L, linewidth=1, color='r', label='L-Ribose')
    ax[0,2].set_title('Self-Ribose H-Bonds')
    ax[0,2].set_xlabel('Time (ns)')
    ax[0,2].set_ylabel('Count')

    ax[1,2].plot(time, D_L, linewidth=1, color='m')
    ax[1,2].set_title('D to L Ribose H-Bonds')
    ax[1,2].set_xlabel('Time (ns)')
    ax[1,2].set_ylabel('Count')

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

    # plt.yscale('log')
    plt.xlabel('Solvent Accessible Surface Area (nm^2)')
    plt.ylabel('log density')
    plt.title('KDE of SASA')
    plt.legend()
    plt.show()

def main():
    config = get_config()
    sims = int(config.get('Analyses','number of sims'))
    nsteps = int(config.get('Input Setup','number steps'))
    cell_name = config.get('Input Setup','crystal structure')
    sheet_resnames = config.get('Input Setup','crystal resnames').split(',')
    test_resnames = config.get('Input Setup','test resnames').split(',')
    report = int(config.get('Input Setup','report interval'))

    file_path = config.get('Input Setup','file path')
    outdir = config.get('Output Parameters','output directory')

    #these need to be initialized as lists of lists since list lengths are dynamic
    #if the number of each mols are different then an array cant be used as well since the columns would be of different legnths
    heights = [[] for _ in range(len(test_resnames))]
    SASA = [[] for _ in range(len(test_resnames))]


    #these can be initalized as arrays as each mol type will give only one output so the size of the array is easily known prior to calculating
    #these give outputs of length one no matter the number of mols so an array works
    nematic_order = np.zeros((int(nsteps/report), len(test_resnames), sims))

    for sim_number in range(sims):
        print('Analyzing sim number', sim_number)

        traj = md.iterload(f'{file_path}/traj_{sim_number}_sheet_{cell_name[:-4]}_mols_{"_".join(test_resnames)}_steps_{nsteps}.dcd', 
                            top=f'{file_path}/traj_{sim_number}_sheet_{cell_name[:-4]}_mols_{"_".join(test_resnames)}_steps_{nsteps}.pdb')

        for chunk in traj:

            if config.get('Analyses','height density') == 'True':
                for i, mol in enumerate(test_resnames):
                    heights[i].extend(compute_heights(chunk, sheet_resnames, mol))

            if config.get('Analyses','Nematic Order Parameter') == 'True':
                nematic_order = compute_nematic_order(chunk, nematic_order, test_resnames, sim_number)

            if config.get('Analyses','Solvent Accessible Surface Area') == 'True':
                SASA = compute_sasa(chunk, test_resnames, SASA)

        #     hbond_counts, traj_D_G, traj_D_C, traj_D_B, traj_L_G, traj_L_C, traj_L_B, traj_D_D, traj_D_L, traj_L_L = compute_hbonds(chunk,hbond_counts)
        #     D_G.extend(traj_D_G)
        #     D_C.extend(traj_D_C)
        #     D_B.extend(traj_D_B)
        #     L_G.extend(traj_L_G)
        #     L_C.extend(traj_L_C)
        #     L_B.extend(traj_L_B)
        #     D_D.extend(traj_D_D)
        #     D_L.extend(traj_D_L)
        #     L_L.extend(traj_L_L)

        # sim_D_G.append(D_G)
        # sim_D_C.append(D_C)
        # sim_D_B.append(D_B)
        # sim_L_G.append(L_G)
        # sim_L_C.append(L_C)
        # sim_L_B.append(L_B)
        # sim_D_D.append(D_D)
        # sim_D_L.append(D_L)
        # sim_L_L.append(L_L)

    # hbond_heatmap(hbond_counts)
    # hbond_order(sim_D_G,sim_D_C,sim_D_B,sim_L_G,sim_L_C,sim_L_B,sim_D_D,sim_D_L,sim_L_L)

    if config.get('Analyses','height density') == 'True':
        graph_heights(heights, config)

    if config.get('Analyses','Nematic Order Parameter') == 'True':
        graph_nematic_order(nematic_order, test_resnames)

    if config.get('Analyses','Solvent Accessible Surface Area') == 'True':
        graph_sasa(SASA, test_resnames)


if __name__ == '__main__':
    main()