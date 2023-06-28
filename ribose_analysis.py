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

def compute_rdf(traj):
    sheet_atoms = traj.topology.select('resn "G" or resn "C"')
    
    try:
        dribose_atoms = traj.topology.select('resn "DRI"')
        if len(dribose_atoms) > 0:
            d_rdf, d_bins = md.compute_rdf(traj, np.array([[sheet_atom, dribose_atom] for sheet_atom in sheet_atoms for dribose_atom in dribose_atoms]))
        else:
            print('No D-ribose in this sim')
            d_rdf, d_bins = None,None
    except:
        print('No D-ribose in this sim')
        d_rdf, d_bins = None,None
    
    try:
        lribose_atoms = traj.topology.select('resn "LRI"')
        if len(lribose_atoms) > 0:
            l_rdf, l_bins = md.compute_rdf(traj, np.array([[sheet_atom, lribose_atom] for sheet_atom in sheet_atoms for lribose_atom in lribose_atoms]))
        else:
            print('No L-ribose in this sim')
            l_rdf, l_bins = None,None
    except:
        print('No L-ribose in this sim')
        l_rdf, l_bins = None,None

    return d_rdf, d_bins, l_rdf, l_bins

def graph_rdf(d_rdf_final, d_bins_final, l_rdf_final, l_bins_final):
    d_rdf_final = np.mean(d_rdf_final,axis=0)
    d_bins_final = np.mean(d_bins_final,axis=0)
    l_rdf_final = np.mean(l_rdf_final, axis=0)
    l_bins_final = np.mean(l_bins_final, axis=0)

    fig, ax = plt.subplots()
    ax.plot(d_bins_final, d_rdf_final, color='r', linewidth=1, label='D-Ribose')
    ax.plot(l_bins_final, l_rdf_final, color='b', linewidth=1, label='L-Ribose')
    ax.legend()
    ax.set_xlabel('Height above sheet (nm)')
    ax.set_ylabel('g(r)')
    ax.set_title('RDF')
    plt.show()

def main():
    args  = parse_args()
    sims = args.nsims
    sim_length = args.nsteps

    dribose_heights, lribose_heights = [],[]
    d_rdf_final, d_bins_final, l_rdf_final, l_bins_final = [],[],[],[]
    
    for sim_number in range(sims):
        print('Analyzing sim number', sim_number)

        traj = md.iterload(f'traj_{sim_number}_lconc_18_steps_{sim_length}.dcd', 
                            top=f'topology_{sim_number}_lconc_18_steps_{sim_length}.pdb')
        for chunk in traj:

            dheight, lheight = compute_heights(chunk)
            dribose_heights.extend(dheight)
            lribose_heights.extend(lheight)

            d_bins, d_rdf, l_bins, l_rdf = compute_rdf(chunk)
            d_rdf_final.append(d_rdf)
            d_bins_final.append(d_bins)
            l_rdf_final.append(l_rdf)
            l_bins_final.append(l_bins)

    graph_rdf(d_rdf_final, d_bins_final, l_rdf_final, l_bins_final)

    graph_heights(dribose_heights, lribose_heights)

if __name__ == '__main__':
    main()