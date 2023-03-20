from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.units.openmm import to_openmm
import numpy as np

def translate(mol, step, axis='x'):
    if (axis == 'x'):
        mol += [step, 0, 0] * angstrom
    elif (axis == 'y'):
        mol += [0,step,0] * angstrom
    else:
        mol += [0,0,step] *angstrom
    return mol 

def rotate(mol, angle, axis='x'):
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


def make_sheet(height, width, tops, poss, model, step=5):
    """Creates an evenly spaced sheet of given molecules and attaches it to openmm modeler.
    Params
    ======
    height (int) - dimension in the x direction to build 2d sheet
    width  (int) - dimension in the y direction to build 2d sheet
    top    (list)(openmm.topology) - topology object of molecule
    pos    (list)(np.array, shape=(n,3)) - starting position of the molecule
    model  (openmm.modeler)
    (step) (int) - space between each molecule in sheet
    
    Returns
    =======
    index_coords (list) - (starting index, ending index) of sheet in modeler"""
    sheet_starting_index = model.topology.getNumAtoms()
    xspacing = 0
    spacing = step * len(tops)
    
    for j in range(width):
        for k in range(len(tops)):
            # x axis
            pos = translate(poss[k], spacing * xspacing, 'x')
            model.add(tops[k], pos)
            for i in range(height):
                # y axis
                pos = translate(pos, spacing, 'y')
                model.add(tops[k], pos)
            
            xspacing += 1
    return [sheet_starting_index, model.topology.getNumAtoms()]

def make_sheet_random(height, width, tops, poss, model, step=5):
    """Creates an evenly spaced sheet of molecules randomly picked from given list
        and attaches it to openmm modeler.
    Gives molecule random rotation.
    Params
    ======
    height (int) - dimension in the x direction to build 2d sheet
    width  (int) - dimension in the y direction to build 2d sheet
    top    (list)(openmm.topology) - topology object of molecule
    pos    (list)(np.array, shape=(n,3)) - starting position of the molecule
    model  (openmm.modeler)
    (step) (int) - space between each molecule in sheet
    
    Returns
    =======
    index_coords (list) - (starting index, ending index) of sheet in modeler"""
    sheet_starting_index = model.topology.getNumAtoms()
    # precalculate random variables
    idx = np.random.choice(np.arange(0, len(tops)), size=height*width)
    axis_rotation = np.random.choice(['x','y','z'], size=height*width)
    angle = np.deg2rad(np.random.randint(0,360,size=height*width))

    for i in range(height):
        for j in range(width):
            ij = i+j
            # x axis
            pos = rotate(poss[idx[ij]], angle[ij], axis=axis_rotation[ij])
            pos = translate(pos, step * j, 'y')
            pos = translate(pos, step * i, 'x')
            model.add(tops[idx[ij]], pos)
    return [sheet_starting_index, model.topology.getNumAtoms()]

def load_mols(filenames):
    """Loads a molecule from file.
    Args
    ====
    filenames (list) - list of molecule sdf files
    """
    mols = {}
    for name in filenames:
        mol = Molecule.from_file(name, file_format="sdf")
        mol.generate_conformers()
        conf = to_openmm(mol.conformers[0])
        top = mol.to_topology().to_openmm()
        mols[name[:-4]] = {
            "mol":mol,
            "topology": top,
            "positions": conf
        }
    return mols

#import molecules 
mols = load_mols(["aD-ribopyro.sdf", 'aL-ribopyro.sdf', 'D-glyceraldehyde.sdf', 'L-glyceraldehyde.sdf', 'guanine.sdf', 'cytosine.sdf'])

#generate residue template 
gaff = GAFFTemplateGenerator(molecules = [mols[name]["mol"] for name in mols.keys()])
#move above and to middle of sheet
ad_ribose_conformer = translate(mols["aD-ribopyro"]["positions"], 10, 'z')
# ad_ribose_conformer = translate(ad_ribose_conformer, 10, 'y')

al_ribose_conformer = translate(mols["aL-ribopyro"]["positions"], 10, 'z')
# al_ribose_conformer = translate(al_ribose_conformer, 10, 'y')
print("Building molecules")

#line up the guanine and cytosines so that the molecules face eachother
# c = rotate(mols["cytosine"]["positions"], np.deg2rad(180), axis = 'z') 
c = rotate(mols["cytosine"]["positions"], np.deg2rad(0), axis='y')
c = rotate(c, np.deg2rad(190), axis='x')
# c = translate(c, 8, 'y')
g = rotate(mols["guanine"]["positions"], np.deg2rad(-50), axis = 'z')
# initializing the modeler requires a topology and pos
# we immediately empty the modeler for use later
model = Modeller(mols["guanine"]["topology"], g) 
model.delete(model.topology.atoms())

#make the sheet (height, width, make sure to pass in the guanine and cytosine confomrers (g and c) and their topologies)
sheet_indices = []
sheet_indices.append(make_sheet(8, 8, [mols["guanine"]["topology"], mols["cytosine"]["topology"]], [g, c], model, step=3.5))

# sheet_indices.append(make_sheet(4, 4, guanine_top, g, model))
# offset the cytosine in the x dim to prevent putting the mols on top of each other
# c = translate(c, 10, 'y')
# sheet_indices.append(make_sheet(4, 4, cytosine_top, c, model, step=10))
print("Molecules added")

# add, at random, either an L or D ribose 
ad_ribose_conformer = translate(ad_ribose_conformer, 4,'z')
ad_ribose_conformer = translate(ad_ribose_conformer, 20, axis='x')
ad_ribose_conformer = translate(ad_ribose_conformer, 20, axis='y')
al_ribose_conformer = translate(al_ribose_conformer, 4,'z')
al_ribose_conformer = translate(al_ribose_conformer, 20, axis='x')
al_ribose_conformer = translate(al_ribose_conformer, 20, axis='y')
make_sheet_random(4, 4, [mols["aD-ribopyro"]["topology"], mols["aL-ribopyro"]["topology"]], [ad_ribose_conformer, al_ribose_conformer], model, step=8)

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

for start, stop in sheet_indices:
    for i in range(start, stop):
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