import numpy as np 
import json 
from openmm.unit import *
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm as pbar
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

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
                  dribose_rdf.append(np.array(frame[res]['positions'])[:, -1])
               if res[0] == 'L':
                  lribose_rdf.append(np.array(frame[res]['positions'])[:, -1])

    #Calculate D-Ribose RDF
   d_rdf = np.histogram(dribose_rdf, bins='auto', density=True)

   # Calculate L-Ribose RDF
   l_rdf = np.histogram(lribose_rdf, bins='auto', density=True)

   fig, ax = plt.subplots(2, 1)

   ax[0].step(d_rdf[1][0:-1], d_rdf[0])
   ax[0].set_title('D-Ribose RDF')
   ax[0].set_xlim([0, 4])
   ax[0].set_xticks(np.arange(0, 4, 0.5))

   ax[1].set_title('L-Ribose RDF')
   ax[1].step(l_rdf[1][0:-1], l_rdf[0])
   ax[1].set_xlim([0, 4])
   ax[1].set_xticks(np.arange(0, 4, 0.5))

   fig.suptitle('Z RDF')
   fig.supxlabel('Distance Above Sheet (Nanometer)')
   fig.supylabel('g(r)')

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
   ax[1].hist2d(np.array(lribose_dx).flatten(), np.array(lribose_dy).flatten(), bins=500)
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

   dribose_x_hist = np.histogram(np.radians(dribose_angle_x), bins='auto', density=True)
   dribose_y_hist = np.histogram(np.radians(dribose_angle_y), bins='auto', density=True)
   dribose_z_hist = np.histogram(np.radians(dribose_angle_z), bins='auto', density=True)

   lribose_x_hist = np.histogram(np.radians(lribose_angle_x), bins='auto', density=True)
   lribose_y_hist = np.histogram(np.radians(lribose_angle_y), bins='auto', density=True)
   lribose_z_hist = np.histogram(np.radians(lribose_angle_z), bins='auto', density=True)

   ax[0].step(dribose_x_hist[1][0:-1],dribose_x_hist[0], label='x rotation')
   ax[0].step(dribose_y_hist[1][0:-1],dribose_y_hist[0], label='y rotation')
   ax[0].step(dribose_z_hist[1][0:-1],dribose_z_hist[0], label='z rotation')
   ax[0].legend()


   ax[1].step(lribose_x_hist[1][0:-1],lribose_x_hist[0], label='x rotation')
   ax[1].step(lribose_y_hist[1][0:-1],lribose_y_hist[0], label='y rotation')
   ax[1].step(lribose_z_hist[1][0:-1],lribose_z_hist[0], label='z rotation')
   ax[1].legend()

   plt.show()

def rotational_order(lconc, nsims, filepath):
    pass

compute_angles(32,2,'.')



