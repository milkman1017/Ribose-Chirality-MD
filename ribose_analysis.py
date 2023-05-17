import numpy as np 
import json 
from openmm.unit import *
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm as pbar

def rdf(lconc, nsims, filepath):

   filenames = glob.glob(f"{filepath}/*.json")[0:nsims]

   dribose_rdf = []
   lribose_rdf = []
   for filename in pbar(filenames, desc="Loading Data"):
         # traj is a list of frames
         # frames are dicts of residues
         with open(filename) as f:
            traj = json.load(f)
         f.close()
         for frame in traj:
            for res in frame:
                  if res[0] == 'D':
                     dribose_rdf.append(np.array(frame[res]['positions'])[:,-1])
                  if res[0] == 'L':
                     lribose_rdf.append(np.array(frame[res]['positions'])[:,-1])


   d_vals, d_bins = np.histogram(dribose_rdf, bins='auto')
   l_vals, l_bins = np.histogram(lribose_rdf, bins='auto')

   d_rdf = [d_vals/np.sum(d_vals), d_bins]
   l_rdf = [l_vals/np.sum(l_vals), l_bins]

   fig, ax = plt.subplots(2,1)

   ax[0].step(d_rdf[1][0:-1], d_rdf[0])
   ax[0].set_title('D-Ribose RDF')
   ax[0].set_xlim([0, 4])
   ax[0].set_xticks(np.arange(0, 4, 0.5))


   ax[1].set_title('L-Ribose RDF')
   ax[1].step(l_rdf[1][0:-1], l_rdf[0])
   ax[1].set_xlim([0, 4])
   ax[1].set_xticks(np.arange(0, 4, 0.5))


   fig.supxlabel('Distance Above Sheet (Nanometer)')
   fig.supylabel('Count')

   plt.savefig(f"ribose_rdf{lconc}_{nsims}.png")

   
def compute_2D_rdf(nsims, filepath):
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

   ax[0].hist2d(np.array(dribose_dx).flatten(), np.array(dribose_dy).flatten(), bins=500)
   ax[1].hist2d(np.array(lribose_dx).flatten(), np.array(lribose_dy).flatten(), bins=100)


   plt.show()

