import csv 
import numpy as np 
import mdtraj as md
from openmm.unit import *

def compute_rdf():
    print('loading input')
    traj = md.load('output.pdb')

    radii_out = []
    density_out = []

    # need to do this for rdf
    box_length = 10.0 * angstrom
    box_vectors = np.eye(3) * box_length
    traj.unitcell_vectors = np.array([box_vectors] * traj.n_frames) 

    pairs = np.array([(123,920)])

    print('Computing RDF')
    radii, density = md.compute_rdf(traj, pairs, r_range = (1,6), bin_width = 0.05)
    radii_out.append(radii)
    density_out.append(density)

    radii_out = np.mean(radii_out, axis = 0)
    density_out = np.mean(density_out, axis = 0)

    print('saving data')
    with open('rdf.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(radii_out)
        writer.writerow(density_out)

compute_rdf()

