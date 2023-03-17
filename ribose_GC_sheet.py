from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.units.openmm import to_openmm
import numpy as np
import random as rand

def translate(mol, step, axis='x'):
    if (axis == 'x'):
        mol += [step, 0, 0] * angstrom
    elif (axis == 'y'):
        mol += [0,step,0] * angstrom
    else:
        mol += [0,0,step] *angstrom
    return mol 

def rotate(mol, angle, axis = 'x'):
    com = [np.average(mol[:,0]), np.average(mol[:,1]), np.average(mol[:,2])]
    mol = translate(mol, -com[0], axis = 'x') 
    mol = translate(mol, -com[1], axis = 'y')
    mol = translate(mol, -com[2], axis = 'z')
    if axis == 'x':
        x = np.array([[1,0,0],
                      [0,np.cos(angle), -np.sin(angle)],
                      [0,np.sin(angle), np.cos(angle)]])
        mol = mol[:,:]@x 
        mol = mol * angstrom
    elif axis == 'y':
        y = np.array([[np.cos(angle),0,np.sin(angle)],
                      [0,1,0],
                      [-np.sin(angle),0,np.cos(angle)]])
        mol = mol[:,:]@y
        mol = mol * angstrom
    else:
        z = np.array([[np.cos(angle),-np.sin(angle),0],
                      [np.sin(angle),np.cos(angle),0],
                      [0,0,1]])
        mol = mol[:,:]@z
        mol = mol * angstrom

    mol = translate(mol, com[0], axis = 'x') 
    mol = translate(mol, com[1], axis = 'y')
    mol = translate(mol, com[2], axis = 'z')
    return mol


def make_GC_sheet(height, width, g, guanine_top, c, cytosine_top, model):
    start_model_index = model.topology.getNumAtoms()
    for i in range(width):
        for j in range(height):
            g = translate(g,10,'x')
            model.add(guanine_top,g)
            c = translate(c,10,'x')
            model.add(cytosine_top,c)
        # move back to start for next row
        g = translate(g,16,'y')
        g = translate(g, -10*height,'x')
        c = translate(c,16,'y')
        c = translate(c, -10*height, 'x')
    end_model_index = model.topology.getNumAtoms()
    return start_model_index, end_model_index

def import_molecule(mol_name, format):
    mol = Molecule.from_file(mol_name, format)
    mol_top = mol.to_topology().to_openmm()
    mol.generate_conformers()
    mol_con = to_openmm(mol.conformers[0])
    
    return mol, mol_top, mol_con

#import molecules 
ad_ribose, ad_ribose_top, ad_ribose_conformer = import_molecule('aD-ribopyro.sdf', 'sdf')
al_ribose, al_ribose_top, al_ribose_conformer = import_molecule('aL-ribopyro.sdf', 'sdf')
guanine, guanine_top, guanine_conformer = import_molecule('guanine.sdf', 'sdf')
cytosine, cytosine_top, cytosine_conformer = import_molecule('cytosine.sdf', 'sdf')

#generate residue template 
gaff = GAFFTemplateGenerator(molecules = [ad_ribose, al_ribose, guanine, cytosine])
#move above and to middle of sheet
ad_ribose_conformer = translate(ad_ribose_conformer, 10, 'z')
ad_ribose_conformer = translate(ad_ribose_conformer, 10, 'y')

al_ribose_conformer = translate(al_ribose_conformer, 10, 'z')
al_ribose_conformer = translate(al_ribose_conformer, 20, 'y')
print("Building molecules")

#line up the guanine and cytosines so that the molecules face eachother
c = rotate(cytosine_conformer, 4.9, axis = 'z') 
c = translate(c, 8, 'y')
g = rotate(guanine_conformer, 180, axis = 'z')
model = Modeller(guanine_top, g)
model.delete(model.topology.atoms())

#make the sheet (height, width, make sure to pass in the guanine and cytosine confomrers (g and c) and their topologies)

GC_start, GC_stop = make_GC_sheet(4,4,g,guanine_top,c,cytosine_top, model)
(print("Molecules added"))

# add, at random, either an L or D ribose 
ribose_chirality = []
for i in range(1):
    randRibose = rand.choice(['ad_ribose_conformer', 'al_ribose_conformer'])
    if randRibose == 'ad_ribose_conformer':
        ribose_chirality.append('D')
        ad = rotate(ad_ribose_conformer, rand.randrange(0,360), axis = rand.choice(['x','y','z']))
        ad = translate(ad, (i+1)*30, 'x')
        model.add(ad_ribose_top, ad)
    elif randRibose == 'al_ribose_conformer':
        ribose_chirality.append('L')
        al = rotate(al_ribose_conformer, rand.randrange(0,10), axis = rand.choice(['x','y','z']))
        al = translate(al, (i+1)*30, 'x')
        model.add(al_ribose_top, al)
print('Ribose Chirlaity for this simulation is: ' + str(ribose_chirality))

print("Building system")
forcefield = ForceField('amber14-all.xml', 'implicit/obc2.xml')
forcefield.registerTemplateGenerator(gaff.generator)
# model.addSolvent(forcefield=forcefield, model='tip3p', boxSize = Vec3(10.5, 6, 3)*nanometers)
system = forcefield.createSystem(model.topology,nonbondedMethod=NoCutoff, nonbondedCutoff=1*nanometer, constraints=HBonds)


# create position restraints (thanks peter eastman https://gist.github.com/peastman/ad8cda653242d731d75e18c836b2a3a5)
restraint = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
system.addForce(restraint)
restraint.addGlobalParameter('k', 100.0*kilojoules_per_mole/angstrom**2)
restraint.addPerParticleParameter('x0')
restraint.addPerParticleParameter('y0')
restraint.addPerParticleParameter('z0')

for i in range(GC_start, GC_stop):
    restraint.addParticle(i, model.positions[i])

integrator = LangevinMiddleIntegrator(300*kelvin, 6/picosecond, 0.004*picoseconds)
model.addExtraParticles(forcefield)
simulation = Simulation(model.topology, system, integrator)
simulation.context.setPositions(model.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)
preEnergyMinPositions = simulation.context.getState(getPositions = True).getPositions()
PDBFile.writeFile(simulation.topology, model.positions, open('preEnergyMin.pdb','w'))
print('Saved Pre-Energy Minimization Positions')
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('output.pdb', 20))
simulation.reporters.append(StateDataReporter(stdout, 20, step=True,
        potentialEnergy=True, temperature=True))
simulation.step(10000)
