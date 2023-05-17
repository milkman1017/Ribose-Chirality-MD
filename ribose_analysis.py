import numpy as np 
import json 
from openmm.unit import *
import glob
import matplotlib.pyplot as plt
data = []

for f in glob.glob('*.json'):
   input = open(f)
   values = json.load(input)
   data.append(values)


dribose_positions = []
lribose_positions = []

for sim in data:
    for step in sim:
        for res in step:
            if res[:1] == 'D':
                dribose_positions.append(step[str(res)]['positions'])
            if res[:1] == 'L':
                lribose_positions.append(step[str(res)]['positions'])


def compute_rdf(dribose_positions, lribose_positions):

   dribose_rdf = []
   lribose_rdf = []
   for mol in dribose_positions:
      for atom in mol:
         dribose_rdf.append(atom[-1])
         pass
      pass

   for mol in lribose_positions:
      for atom in mol:
         lribose_rdf.append(atom[-1])
         pass
      pass
   
   d_rdf = np.histogram(dribose_rdf, bins = 'auto')
   l_rdf = np.histogram(lribose_positions, bins='auto')

   fig, ax = plt.subplots(2,1)

   ax[0].step(d_rdf[1][0:-1], d_rdf[0])
   ax[0].set_title('D-Ribose RDF')

   ax[1].set_title('L-Ribose RDF')
   ax[1].step(l_rdf[1][0:-1], l_rdf[0])

   fig.supxlabel('Distance Above Sheet (Nanometer)')
   fig.supylabel('Count')

   plt.show()
    
compute_rdf(dribose_positions, lribose_positions)