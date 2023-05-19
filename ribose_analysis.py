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

   ax[0].hist2d(np.array(dribose_dx).flatten(), np.array(dribose_dy).flatten(), bins=500)
   ax[0].set_title('D-Ribose')
   ax[1].hist2d(np.array(lribose_dx).flatten(), np.array(lribose_dy).flatten(), bins=500)
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
                     index = len(frame[res]['positions'])//3 

                     first_atom = frame[res]['positions'][index]
                     middle_atom = frame[res]['positions'][index*2]
                     last_atom = frame[res]['positions'][index*3]

                     config = np.array([first_atom, middle_atom, last_atom])

                     #obtain normal vector and normalize

                     vector1 = config[1] - config[0]
                     vector2 = config[2] - config[0]

                     normal_orientation = np.cross(vector1, vector2)/np.linalg.norm(np.cross(vector1, vector2))

                     angle_x = np.degrees(np.arctan2(np.dot(normal_orientation, np.cross(y_axis, z_axis)), np.dot(normal_orientation, y_axis))) + 180
                     angle_y = np.degrees(np.arctan2(np.dot(normal_orientation, np.cross(z_axis, x_axis)), np.dot(normal_orientation, z_axis))) + 180
                     angle_z = np.degrees(np.arctan2(np.dot(normal_orientation, np.cross(x_axis, y_axis)), np.dot(normal_orientation, x_axis))) + 180

                     dribose_angle_x.append(angle_x)
                     dribose_angle_y.append(angle_y)
                     dribose_angle_z.append(angle_z)

                  if res[0] == 'L':
                     index = len(frame[res]['positions'])//3 

                     first_atom = frame[res]['positions'][index]
                     middle_atom = frame[res]['positions'][index*2]
                     last_atom = frame[res]['positions'][index*3]

                     config = np.array([first_atom, middle_atom, last_atom])

                     #obtain normal vector and normalize

                     vector1 = config[1] - config[0]
                     vector2 = config[2] - config[0]
                     
                     normal_orientation = np.cross(vector1, vector2)/np.linalg.norm(np.cross(vector1, vector2))

                     angle_x = np.degrees(np.arctan2(np.dot(normal_orientation, np.cross(y_axis, z_axis)), np.dot(normal_orientation, y_axis))) + 180
                     angle_y = np.degrees(np.arctan2(np.dot(normal_orientation, np.cross(z_axis, x_axis)), np.dot(normal_orientation, z_axis))) + 180
                     angle_z = np.degrees(np.arctan2(np.dot(normal_orientation, np.cross(x_axis, y_axis)), np.dot(normal_orientation, x_axis))) + 180

                     lribose_angle_x.append(angle_x)
                     lribose_angle_y.append(angle_y)
                     lribose_angle_z.append(angle_z)


   fig, ax = plt.subplots(2,3, subplot_kw={'projection':'polar'})

   dribose_x_hist = np.histogram(dribose_angle_x, bins= 500)
   dribose_y_hist = np.histogram(dribose_angle_y, bins=500)
   dribose_z_hist = np.histogram(dribose_angle_z, bins=500)

   lribose_x_hist = np.histogram(lribose_angle_x, bins=500)
   lribose_y_hist = np.histogram(lribose_angle_y, bins=500)
   lribose_z_hist = np.histogram(lribose_angle_z, bins=500)

   ax[0][0].step(dribose_x_hist[1][0:-1], dribose_x_hist[0])
   ax[0][1].step(dribose_y_hist[1][0:-1], dribose_y_hist[0])
   ax[0][2].step(dribose_z_hist[1][0:-1], dribose_z_hist[0])

   ax[1][0].step(lribose_x_hist[1][0:-1], lribose_x_hist[0])
   ax[1][1].step(lribose_y_hist[1][0:-1], lribose_y_hist[0])
   ax[1][2].step(lribose_z_hist[1][0:-1], lribose_z_hist[0])

   plt.show()

   #normalize coordinate vectors 

compute_angles(32,2,'.')


