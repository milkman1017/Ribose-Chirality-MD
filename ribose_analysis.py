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
import argparse
from scipy.stats import gaussian_kde

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsims', type=int, default=1, help='number of simulations to analyze')
    parser.add_argument('--nsteps', type=int, default=10000, help='the length of sims to analyze')
    parser.add_argument('--lconc', type=int, default=0, help='the amount of the l-ribose in the simulation to analye')
    parser.add_argument('--filepath', type=str, default='.', help='the file path from which to load sim data')
    parser.add_argument('--outdir', type=str, default='.', help='output directory for results')
    args = parser.parse_args()
    return args

def compute_heights(traj):
    sheet_atoms = traj.topology.select('resn "G" or resn "C"')
    
    try:
        dribose_atoms = traj.topology.select('resn "DRI"')
        if len(dribose_atoms) > 0:
            dribose_heights = md.compute_distances(traj, np.array([[sheet_atom, dribose_atom] for sheet_atom in sheet_atoms for dribose_atom in dribose_atoms]))[:, 0]
        else:
            print('No D-ribose in this sim')
            dribose_heights = None
    except:
        print('No D-ribose in this sim')
        dribose_heights = None
    
    try:
        lribose_atoms = traj.topology.select('resn "LRI"')
        if len(lribose_atoms) > 0:
            lribose_heights = md.compute_distances(traj, np.array([[sheet_atom, lribose_atom] for sheet_atom in sheet_atoms for lribose_atom in lribose_atoms]))[:, 0]
        else:
            print('No L-ribose in this sim')
            lribose_heights = None
    except:
        print('No L-ribose in this sim')
        lribose_heights = None
    
    return dribose_heights, lribose_heights

def graph_heights(dribose_heights, lribose_heights):
    dribose_heights = np.array(dribose_heights)
    lribose_heights = np.array(lribose_heights)

    d_kde = gaussian_kde(dribose_heights)
    l_kde = gaussian_kde(lribose_heights)

    dx = np.linspace(dribose_heights.min(), dribose_heights.max(), 1000)
    lx = np.linspace(lribose_heights.min(), lribose_heights.max(), 1000)

    d_kde_vals = d_kde.evaluate(dx)
    l_kde_vals = l_kde.evaluate(lx)

    fig, ax = plt.subplots()

    ax.plot(dx,d_kde_vals,linewidth=1, color='r')
    ax.plot(lx,l_kde_vals,linewidth=1, color='b')
    ax.legend(['D-Ribose','L-Ribose'])
    ax.set_xlabel('Height Above Sheet (nm)')
    ax.set_ylabel('PDF')
    ax.set_title('Probability Density of height of ribose')
    plt.show()

def compute_hbonds(chunk, hbond_counts):

    for frame in chunk:
        hbonds = md.baker_hubbard(frame,exclude_water=True)
        
        #get hbond counts 
        for hbond in hbonds:
            atom1, atom2 = hbond[0], hbond[2]
            res1, res2 = frame.topology.atom(atom1).residue, frame.topology.atom(atom2).residue
            atom1_index, atom2_index = frame.topology.atom(atom1).index % res1.n_atoms, frame.topology.atom(atom2).index % res2.n_atoms

            hbond_key = f'{res1.name}-{res2.name}'
            hbond_count_key = f"{atom1_index}-{atom2_index}"

            if hbond_key not in hbond_counts:
                hbond_counts[hbond_key] = dict()
            if hbond_count_key not in hbond_counts[hbond_key]:
                hbond_counts[hbond_key][hbond_count_key]=0
            hbond_counts[hbond_key][hbond_count_key] += 1

    return hbond_counts

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

def nematic_order(traj):
    dribose_indices_list = []  
    lribose_indices_list = []  

    for residue in traj.topology.residues:
        if residue.name == 'DRI':
            dribose_indices = [atom.index for atom in residue.atoms]
            dribose_indices_list.append(dribose_indices) 
        elif residue.name == 'LRI':
            lribose_indices = [atom.index for atom in residue.atoms]
            lribose_indices_list.append(lribose_indices) 

    dribose_order_list = [] 
    lribose_order_list = []  

    for dribose_indices in dribose_indices_list:
        dribose_order = md.compute_nematic_order(traj, indices=dribose_indices_list)
        dribose_order_list.append(dribose_order) 

    for lribose_indices in lribose_indices_list:
        lribose_order = md.compute_nematic_order(traj, indices=lribose_indices_list)
        lribose_order_list.append(lribose_order)

    dribose_order_list = np.mean(dribose_order_list, axis=0)
    lribose_order_list = np.mean(lribose_order_list, axis=0)

    return dribose_order_list, lribose_order_list

def graph_nematic_order(dribose_order, lribose_order):
    dribose_order = np.mean(dribose_order, axis=0)
    lribose_order = np.mean(lribose_order, axis=0)

    time = np.arange(len(dribose_order)) * 0.004

    fig, ax = plt.subplots(2,1)
    ax[0].plot(time, dribose_order, color='b', linewidth=1, label='D-ribose')
    ax[0].plot(time, lribose_order, color='r', linewidth=1, label='L-ribose')
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Nematic Order Parameter')
    ax[0].legend()

    ax[1].hist(dribose_order, color='blue', histtype='step', label='D-Ribose',density=True, bins='auto')
    ax[1].hist(lribose_order, color='r', histtype='step', label='L-ribose',density=True, bins='auto')
    ax[1].set_xlabel('Nematic Order Parameter')
    ax[1].legend()
    
    plt.suptitle('Nematic Order of Ribose Enantiomers')
    plt.show()

def main():
    args  = parse_args()
    sims = args.nsims
    sim_length = args.nsteps

    dribose_heights, lribose_heights = [],[]
    dribose_order, lribose_order = [],[]
    hbond_counts = dict()

    for sim_number in range(sims):
        print('Analyzing sim number', sim_number)

        traj = md.iterload(f'traj_{sim_number}_lconc_18_steps_{sim_length}.dcd', 
                            top=f'topology_{sim_number}_lconc_18_steps_{sim_length}.pdb')
        
        traj_d_order, traj_l_order = [],[]
        for chunk in traj:
            # dheight, lheight = compute_heights(chunk)
            # dribose_heights.extend(dheight)
            # lribose_heights.extend(lheight)

            # hbond_counts = compute_hbonds(chunk,hbond_counts)

            traj_d_ord, traj_l_ord = nematic_order(chunk)
            traj_d_order.extend(traj_d_ord)
            traj_l_order .extend(traj_l_ord)
        dribose_order.append(traj_d_order)
        lribose_order.append(traj_l_order)

    # hbond_heatmap(hbond_counts)

    graph_nematic_order(dribose_order, lribose_order)
    # graph_heights(dribose_heights, lribose_heights)

if __name__ == '__main__':
    main()