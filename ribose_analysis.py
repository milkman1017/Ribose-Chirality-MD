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

   fig.suptitle('Z RDF')
   fig.supxlabel('Distance Above Sheet (Nanometer)')
   fig.supylabel('Count')

   plt.show()

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
   fig.suptitle('X-Y RDF')
   fig.supxlabel('X coordinate (nm)')
   fig.supylabel('Y Coordinate (nm)')

   ax[0].hist2d(np.array(dribose_dx).flatten(), np.array(dribose_dy).flatten(), bins=500)
   ax[0].set_title('D-Ribose')
   ax[1].hist2d(np.array(lribose_dx).flatten(), np.array(lribose_dy).flatten(), bins=100)
   ax[1].set_title('L-Ribose')

   plt.show()

def compute_angles(lcon, nsims, filepath):
   filenames = glob.glob(f"{filepath}/*.json")[0:nsims]

   dribose_angle_x = []
   dribose_angle_y = []
   dribose_angle_z = []

   lribose_angle_x = []
   lribose_angle_y = []
   lribose_angle_z = []

   #define axes 
   x_axis = np.array([1,0,0])
   y_axis = np.array([0,1,0])
   z_axis = np.array([0,0,1])

   for filename in pbar(filenames, desc="Loading Data"):
         # traj is a list of frames
         # frames are dicts of residues
         with open(filename) as f:
            traj = json.load(f)
         f.close()
         for frame in traj:
            for res in frame:
                  
                  #define configuration of molecuels by getting three non colinear points
                  #arbitrarly chose the first, last, and middle indexed atoms 
                  if res[0] == 'D':
                     first_atom = frame[res]['positions'][0]
                     last_atom = frame[res]['positions'][-1]
                     middle_atom = frame[res]['positions'][len(frame[res]['positions'])//2]
                     config = np.array([first_atom, middle_atom, last_atom])

                     #obtain normal vector and normalize

                     vector1 = config[1] - config[0]
                     vector2 = config[2] - config[0]
                     normal_orientation = np.cross(vector1, vector2)/np.linalg.norm(np.cross(vector1, vector2))

                     dribose_angle_x.append(np.degrees(np.arccos(np.dot(x_axis, normal_orientation))))
                     dribose_angle_y.append(np.degrees(np.arccos(np.dot(y_axis, normal_orientation))))
                     dribose_angle_z.append(np.degrees(np.arccos(np.dot(z_axis, normal_orientation))))


                  if res[0] == 'L':
                     lribose_pos = []
                     first_atom = frame[res]['positions'][0]
                     last_atom = frame[res]['positions'][-1]
                     middle_atom = frame[res]['positions'][len(frame[res]['positions'])//2]
                     config = [first_atom, middle_atom, last_atom]
                     lribose_pos.append(np.array(config))

   fig, ax = plt.subplots(1,3)

   dribose_x_hist = np.histogram(dribose_angle_x, bins='auto')
   dribose_y_hist = np.histogram(dribose_angle_y, bins='auto')
   dribose_z_hist = np.histogram(dribose_angle_z, bins='auto')

   ax[0].step(dribose_x_hist[1][0:-1], dribose_x_hist[0])
   ax[1].step(dribose_y_hist[1][0:-1], dribose_y_hist[0])
   ax[2].step(dribose_z_hist[1][0:-1], dribose_z_hist[0])

   plt.show()

compute_angles(2,2,'.')

