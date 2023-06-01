import numpy as np 
import json 
from openmm.unit import *
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm as pbar
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import ripser

def rdf(lconc, nsims, filepath):
    filenames = glob.glob(f"{filepath}/*.json")[0:nsims]

    dribose_rdf = []
    lribose_rdf = []
    for filename in pbar(filenames, desc="Loading Data"):
        with open(filename) as f:
            traj = json.load(f)
        f.close()
        for frame in traj:
            for res in frame:
                    if res[0] == 'D':
                        dribose_rdf.append(np.average(np.array(frame[res]['positions'])[:,-1]))
                    if res[0] == 'L':
                        lribose_rdf.append(np.average(np.array(frame[res]['positions'])[:, -1]))                      

     #Calculate D-Ribose RDF
    d_rdf = np.histogram(dribose_rdf, bins='auto')
    d_gr = d_rdf[0]/((32/(8*8*3))*8*8*(d_rdf[1][1]-d_rdf[1][0]))


    # Calculate L-Ribose RDF
    l_rdf = np.histogram(lribose_rdf, bins='auto')
    l_gr = l_rdf[0]/((32/(8*8*3))*8*8*(l_rdf[1][1]-l_rdf[1][0]))

    fig, ax = plt.subplots()

    ax.step(d_rdf[1][0:-1], d_gr, label='D-Ribose')
    ax.step(l_rdf[1][0:-1], l_gr, label='L-Ribose')
    ax.legend()

    plt.ylim(0,None)
    plt.xlim(0,2)
    plt.show()

def compute_2D_rdf(lconc, nsims, filepath):
    filenames = glob.glob(f"{filepath}/*.json")[0:nsims]
    dribose_dx = []
    lribose_dx = []

    dribose_dy = []
    lribose_dy = []

    for filename in pbar(filenames, desc="Loading Data"):
            # traj is a list of frames
            # frames are dicts of residues
            with open(filename) as f:
                traj = json.load(f)
            f.close()
            for frame in traj:
                for res in frame:
                        if res[0] == 'D':
                            dribose_dx.append(np.array(frame[res]['positions'])[:,0])
                            dribose_dy.append(np.array(frame[res]['positions'])[:,1])
                        if res[0] == 'L':
                            lribose_dx.append(np.array(frame[res]['positions'])[:,0])
                            lribose_dy.append(np.array(frame[res]['positions'])[:,1])

    
    fig, ax = plt.subplots(1,2)
    fig.suptitle('X-Y RDF')
    fig.supxlabel('X coordinate (nm)')
    fig.supylabel('Y Coordinate (nm)')

    ax[0].hist2d(np.array(dribose_dx).flatten(), np.array(dribose_dy).flatten(), bins=500, density=True)
    ax[0].set_title('D-Ribose')
    ax[1].hist2d(np.array(lribose_dx).flatten(), np.array(lribose_dy).flatten(), bins=500, density=True)
    ax[1].set_title('L-Ribose')

    plt.show()

def compute_angles(lconc, nsims, filepath):
    filenames = glob.glob(f"{filepath}/*.json")[:nsims]

    dribose_angle_x = []
    dribose_angle_y = []
    dribose_angle_z = []

    lribose_angle_x = []
    lribose_angle_y = []
    lribose_angle_z = []

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    angle_lists = [(dribose_angle_x, dribose_angle_y, dribose_angle_z), (lribose_angle_x, lribose_angle_y, lribose_angle_z)]

    for filename in pbar(filenames, desc="Loading Data"):
        with open(filename) as f:
            traj = json.load(f)
        
            for frame in traj:
                for res in frame:
                    if res[0] in ['D', 'L']:
                            mol = np.array(frame[res]['positions'])

                            centroid = np.average(mol, axis=0)
                            mol -= centroid

                            cov_matrix = np.cov(mol, rowvar=False)

                            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                            principle_axes = eigenvectors.T

                            dot_products = np.dot(principle_axes, [x_axis, y_axis, z_axis])

                            angles = np.degrees(np.arctan2(dot_products[:, 1], dot_products[:, 0]))+180
                            angle_x, angle_y, angle_z = angles

                            angle_list = angle_lists[res[0] == 'L']

                            angle_list[0].append(angle_x)
                            angle_list[1].append(angle_y)
                            angle_list[2].append(angle_z)
                        

    fig, ax = plt.subplots(2, subplot_kw={'projection': 'polar'})

    dribose_x_hist = np.histogram(np.radians(dribose_angle_x), bins=100, density=True)
    dribose_y_hist = np.histogram(np.radians(dribose_angle_y), bins=100, density=True)
    dribose_z_hist = np.histogram(np.radians(dribose_angle_z), bins=100, density=True)

    lribose_x_hist = np.histogram(np.radians(lribose_angle_x), bins=100, density=True)
    lribose_y_hist = np.histogram(np.radians(lribose_angle_y), bins=100, density=True)
    lribose_z_hist = np.histogram(np.radians(lribose_angle_z), bins=100, density=True)

    ax[0].plot(dribose_x_hist[1][:-1], dribose_x_hist[0], label='x rotation')
    ax[0].plot(dribose_y_hist[1][:-1], dribose_y_hist[0], label='y rotation')
    ax[0].plot(dribose_z_hist[1][:-1], dribose_z_hist[0], label='z rotation')
    ax[0].legend()

    ax[1].plot(lribose_x_hist[1][:-1], lribose_x_hist[0], label='x rotation')
    ax[1].plot(lribose_y_hist[1][:-1], lribose_y_hist[0], label='y rotation')
    ax[1].plot(lribose_z_hist[1][:-1], lribose_z_hist[0], label='z rotation')
    ax[1].legend()

    plt.show()


def hydrogen_bonds(lconc, nsims, filepath):
    filenames = glob.glob(f"{filepath}/*.json")[0:nsims]
    
    donor_labels = set()
    acceptor_labels = set()

    for filename in pbar(filenames, desc="Loading Data"):
        with open(filename) as f:
            traj = json.load(f)
        
        hbonds = traj[-1]['hbonds']


        for residue, bond_dict in hbonds.items():
            donor_residue, acceptor_residue = residue.split('-')
            
            for atom in bond_dict.keys():
                donor_labels.add(f"{donor_residue}-{atom.split('-')[0]}")
                acceptor_labels.add(f"{acceptor_residue}-{atom.split('-')[1]}")

        f.close()
    
    donor_labels = sorted(donor_labels)
    acceptor_labels = sorted(acceptor_labels, reverse=True)
    
    bond_data = np.zeros((len(donor_labels), len(acceptor_labels)))

    for filename in pbar(filenames, desc="Processing Data"):
        with open(filename) as f:
            traj = json.load(f)
        
        hbonds = traj[-1]['hbonds']
        
        for residue, bond_dict in hbonds.items():
            donor_residue, acceptor_residue = residue.split('-')

            for atom, count in bond_dict.items():
                donor, acceptor = atom.split('-')
                donor_label = f"{donor_residue}-{donor}"
                acceptor_label = f"{acceptor_residue}-{acceptor}"
                donor_index = donor_labels.index(donor_label)
                acceptor_index = acceptor_labels.index(acceptor_label)
                bond_data[donor_index, acceptor_index] += count
        
        f.close()

    bond_data = (bond_data - np.mean(bond_data))/np.std(bond_data)

    fig, ax = plt.subplots()
    im = ax.imshow(bond_data, cmap="hot")

    ax.set_xticks(np.arange(len(acceptor_labels)))
    ax.set_xlabel('Acceptors')
    ax.set_yticks(np.arange(len(donor_labels)))
    ax.set_xticklabels(acceptor_labels)
    ax.set_yticklabels(donor_labels)
    ax.set_ylabel('Donors')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_title('Hydrogen Bond Heatmap')

    plt.show()

    return bond_data

hbonds = hydrogen_bonds(32,10,'.')

def hbond_TDA(hbonds):
     pass
     



