import warnings
warnings.filterwarnings("ignore", message="importing 'simtk.openmm' is deprecated. Import 'openmm' instead.")

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.units.openmm import to_openmm
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import argparse
import multiprocessing as mp
from tqdm import tqdm
import json
from simtk.openmm import app

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nprocs', type=int, default=8, help='number of simultaneously processes to use')
    parser.add_argument('--outdir', type=str, default='.', help='output directory')
    parser.add_argument('--nsims', type=int, default=8, help='number of simulations to run per umbrella')
    parser.add_argument('--ngpus', type=int, default=1, help='number of gpus on the system')
    parser.add_argument('--nsteps', type=int, default=100000, help='number of steps')
    parser.add_argument('--report', type=int, default=500, help='report interval')
    parser.add_argument('--verbose', action='store_true', help='print verbose output')
    parser.add_argument('--height_start', type=int, default=1, help='starting target z coordinate for umbrella sampling')
    parser.add_argument('--height_end', type=int, default=10, help='ending z coordinate for umbrella sampling')
    parser.add_argument('--z_increment', type=int, default=0.5, help='how much to increase the target z coordinate for each umbrella')
    parser.add_argument('--ribose', choices=['D','L','both'], help='chose which ribose to simulate: D, L, or one of each')
    args = parser.parse_args()
    return args

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


def make_sheet(height, width, tops, poss, model, step=5.0):
    """Creates an evenly spaced sheet of given molecules and attaches it to openmm modeler.
    Params
    ======
    height (int) - dimension in the x direction to build 2d sheet
    width  (int) - dimension in the y direction to build 2d sheet
    top    (list)(openmm.topology) - topology object of molecule
    pos    (list)(np.array, shape=(n,3)) - starting position of the molecule
    model  (openmm.modeler)
    (step) (float) - space between each molecule in sheet
    
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

def spawn_sugar(tops, poss, model, ribose_type):
    sheet_starting_index = model.topology.getNumAtoms()

    poss[0] = translate(poss[0], 2, 'x')
    poss[0] = translate(poss[0], 2, 'y')

    if ribose_type == 'D':
        model.add(tops[0], poss[0])

    if ribose_type == 'L':
        model.add(tops[1], poss[1])

    return [sheet_starting_index, model.topology.getNumAtoms()]

def load_mols(filenames, resnames):
    """Loads a molecule from file.
    Args
    ====
    filenames (list) - list of molecule sdf files
    """
    mols = {}
    for filename, resname in zip(filenames, resnames):
        mol = Molecule.from_file(filename, file_format='sdf')
        mol.generate_conformers()
        conf = to_openmm(mol.conformers[0])
        top = mol.to_topology().to_openmm()
        top = md.Topology.from_openmm(top)
        top.residue(0).name = resname
        top = top.to_openmm()
        mols[filename[:-4]] = {
            "mol":mol,
            "topology": top,
            "positions": conf,
            'resname': resname
        }
    return mols

def simulate(jobid, device_idx, start_z, end_z, args):
    mols = load_mols(["aD-ribopyro.sdf", 'aL-ribopyro.sdf', 'guanine.sdf', 'cytosine.sdf'], 
                    ['DRIB', 'LRIB', 'GUA', "CYT"])

    end_z = end_z/10

    target = start_z

    #generate residue template 
    gaff = GAFFTemplateGenerator(molecules = [mols[name]["mol"] for name in mols.keys()])
    #move above and to middle of sheet
    ad_ribose_conformer = translate(mols["aD-ribopyro"]["positions"], target, 'z')
    # ad_ribose_conformer = translate(ad_ribose_conformer, 20, 'y')
    # ad_ribose_conformer = translate(ad_ribose_conformer, 20, 'x')

    al_ribose_conformer = translate(mols["aL-ribopyro"]["positions"], target, 'z')
    # al_ribose_conformer = translate(al_ribose_conformer, 20, 'y')
    # al_ribose_conformer = translate(al_ribose_conformer, 20, 'x')
    if(args.verbose):
        print("Building molecules:", jobid)

    #line up the guanine and cytosines so that the molecules face eachother
    c = rotate(mols["cytosine"]["positions"], np.deg2rad(300), axis = 'z') 
    c = rotate(c, np.deg2rad(180), axis='y')
    c = rotate(c, np.deg2rad(190), axis='x')
    c = translate(c,1,'z')
    c = translate(c,4,'x')
    c = translate(c,4,'y')
    # c = translate(c, 8, 'y')

    g = rotate(mols["guanine"]["positions"], np.deg2rad(-50), axis = 'z')
    g = translate(g, 4.7, axis='x')
    g = translate(g, 4, 'y')
    g = translate(g, 1, 'z')


    # initializing the modeler requires a topology and pos
    # we immediately empty the modeler for use later

    model = Modeller(mols["guanine"]["topology"], g) 
    model.delete(model.topology.atoms())

    #make the sheet (height, width, make sure to pass in the guanine and cytosine confomrers (g and c) and their topologies)
    sheet_indices = []
    sugar_indices = []

    sheet_indices.append(make_sheet(1,1, [mols["guanine"]["topology"], mols["cytosine"]["topology"]], [g, c], model, step=3.3))

    sugar_indices.append(spawn_sugar([mols["aD-ribopyro"]["topology"], mols["aL-ribopyro"]["topology"]], [ad_ribose_conformer, al_ribose_conformer], model, 'D'))
    if(args.verbose):
        print("Building system:", jobid)
    forcefield = ForceField('amber14-all.xml', 'tip3p.xml')
    forcefield.registerTemplateGenerator(gaff.generator)

    box_size = [
        Vec3(1.5,0,0),
        Vec3(0,1.5,0),
        Vec3(0,0,end_z * 2)
    ]

    model.addSolvent(forcefield=forcefield, model='tip3p', boxSize=Vec3(1.5,1.5,end_z *2))
    model.topology.setPeriodicBoxVectors(box_size)

    system = forcefield.createSystem(model.topology, nonbondedMethod=PME, nonbondedCutoff=0.5*nanometer, constraints=HBonds)

    # create position sheet_restraints (thanks peter eastman https://gist.github.com/peastman/ad8cda653242d731d75e18c836b2a3a5)
    sheet_restraint = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    system.addForce(sheet_restraint)
    sheet_restraint.addGlobalParameter('k', 100.0*kilojoules_per_mole/angstrom**2)
    sheet_restraint.addPerParticleParameter('x0')
    sheet_restraint.addPerParticleParameter('y0')
    sheet_restraint.addPerParticleParameter('z0')

    for start, stop in sheet_indices:
        for i in range(start, stop):
            sheet_restraint.addParticle(i, model.positions[i])

    #add in bias potential for umbrella sampling 
    custom_force = CustomExternalForce('j*((x-x)^2+(y-y)^2+(z-target)^2)')
    system.addForce(custom_force)
    custom_force.addGlobalParameter("target", 25*angstrom)  
    custom_force.addGlobalParameter("j", 10*kilojoules_per_mole/nanometer**2) 
    custom_force.addPerParticleParameter('z0')
    custom_force.addPerParticleParameter('x0')
    custom_force.addPerParticleParameter('y0')


    for start, stop in sugar_indices:
        for i in range(start, stop):
            custom_force.addParticle(i, model.positions[i])

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    model.addExtraParticles(forcefield)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaDeviceIndex': str(device_idx), 'CudaPrecision': 'single'}

    simulation = Simulation(model.topology, system, integrator, platform, properties)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    # save pre-minimized positions as pdb
    PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open("pre_energy_min.pdb", 'w'))

    simulation.minimizeEnergy()

    simulation.reporters.append(PDBReporter('umbrella.pdb', args.report))
    simulation.reporters.append(StateDataReporter(stdout, args.report, step=True,
        potentialEnergy=True, temperature=True, speed=True))
    simulation.step(args.nsteps)

    # simulation.reporters.append(StateDataReporter(f"{args.outdir}/output{jobid}.txt", args.report, step=True, potentialEnergy=True, temperature=True, speed=True))

    # trajectory = []

    # trajectory.append(simulation.reporters.append(StateDataReporter(potentialEnergy=True)))
          
    # with open(f'{args.outdir}/umbrella-{current_z_target}_job{jobid}.json', 'w') as f:
    #     f.write(json.dumps(trajectory))

args = parse_args()    
simulate(1, 0, 5, 25, args)
'The absolute lowest D ribose can go is 4.1'


# def main():
#     args = parse_args()
#     sims = args.nsims
#     gpus = args.ngpus
#     proc = args.nprocs
#     jobs = 
#     processes = []
#     height_start = args.height_start
#     height_end = args.height_end
#     dz = args.z_increment
#     current_z = height_start 

#     while z < height_end:
#         while jobs < sims:
#             if len(processes) < proc: 
#                 print('starting process', jobs)
#                 p = mp.Process(target=simulate, args=(jobs, (jobs%gpus), current_z, args))
#                 p.start()
#                 processes.append(p)
#         z += dz

#     # with tqdm(total=total_sims) as pbar:
#     #     while jobs < total_sims:
#     #         if(len(processes) < proc):
#     #             print("Starting process", jobs)
#     #             p = mp.Process(target=simulate, args=(jobs, (jobs % gpus), args))
#     #             p.start()
#     #             processes.append(p)
#     #             jobs += 1
#     #         for p in processes:
#     #             if not p.is_alive():
#     #                 processes.remove(p)
#     #                 pbar.update(1)

#     # # Wait for all processes to finish
#     # for p in processes:
#     #     p.join()

# if __name__ == "__main__":
#     main()