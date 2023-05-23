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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nprocs', type=int, default=8, help='number of simultaneously processes to use')
    parser.add_argument('--outdir', type=str, default='.', help='output directory')
    parser.add_argument('--nsims', type=int, default=8, help='number of simulations to run')
    parser.add_argument('--ngpus', type=int, default=1, help='number of gpus on the system')
    parser.add_argument('--nsteps', type=int, default=100000, help='number of steps')
    parser.add_argument('--report', type=int, default=500, help='report interval')
    parser.add_argument('--verbose', action='store_true', help='print verbose output')
    parser.add_argument('--sh', type=int, default=5, help='sheet height')
    parser.add_argument('--sw', type=int, default=5, help='sheet width')
    parser.add_argument('--lconc', type=int, default=5, help='number of lribose molecules in a given sheet')
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

def make_sheet_random(height, width, tops, poss, model, lconc, step=5, mindist=5.0):
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
    # create a list of lconc tops[1]
    ls = [1] * lconc
    ds = [0] * (height*width - lconc)
    idx = [*ls, *ds]
    np.random.shuffle(idx)
    idx = np.array(idx)

    # precalculate random variables
    xpos = (np.arange(0, width*height)*(step+mindist) - np.random.uniform(-mindist, 0, size=height*width)).flatten()
    ypos = (np.arange(0, width*height)*(step+mindist) - np.random.uniform(-mindist, 0, size=height*width)).flatten()
    z_offset = np.random.uniform(-4.5, 1, size=height*width)

    axis_rotation = np.random.choice(['x','y','z'], size=height*width)
    angle = np.deg2rad(np.random.randint(0,360,size=height*width))
    k = 0
    for i in range(height):
        for j in range(width):    
            # x axis
            pos = rotate(poss[idx[k]], angle[k], axis=axis_rotation[k])
            pos = translate(pos, xpos[k], 'y')
            pos = translate(pos, ypos[k], 'x')
            pos = translate(pos, z_offset[k], 'z')
            model.add(tops[idx[k]], pos)
            k+=1
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


def simulate(jobid, device_idx, args):
    print(device_idx)
    mols = load_mols(["aD-ribopyro.sdf", 'aL-ribopyro.sdf', 'guanine.sdf', 'cytosine.sdf'], 
                    ['DRIB', 'LRIB', 'GUA', "CYT"])

    #generate residue template 
    gaff = GAFFTemplateGenerator(molecules = [mols[name]["mol"] for name in mols.keys()])
    #move above and to middle of sheet
    ad_ribose_conformer = translate(mols["aD-ribopyro"]["positions"], 14, 'z')
    # ad_ribose_conformer = translate(ad_ribose_conformer, 20, 'y')
    # ad_ribose_conformer = translate(ad_ribose_conformer, 20, 'x')

    al_ribose_conformer = translate(mols["aL-ribopyro"]["positions"], 14, 'z')
    # al_ribose_conformer = translate(al_ribose_conformer, 20, 'y')
    # al_ribose_conformer = translate(al_ribose_conformer, 20, 'x')
    if(args.verbose):
        print("Building molecules:", jobid)

    #line up the guanine and cytosines so that the molecules face eachother
    c = rotate(mols["cytosine"]["positions"], np.deg2rad(300), axis = 'z') 
    c = rotate(c, np.deg2rad(180), axis='y')
    c = rotate(c, np.deg2rad(190), axis='x')
    # c = translate(c, 8, 'y')
    g = rotate(mols["guanine"]["positions"], np.deg2rad(-50), axis = 'z')
    g = translate(g, .7, axis='x')

    # initializing the modeler requires a topology and pos
    # we immediately empty the modeler for use later

    model = Modeller(mols["guanine"]["topology"], g) 
    model.delete(model.topology.atoms())

    #make the sheet (height, width, make sure to pass in the guanine and cytosine confomrers (g and c) and their topologies)
    sheet_indices = []

    sheet_indices.append(make_sheet(args.sh, args.sw//2 + 1, [mols["guanine"]["topology"], mols["cytosine"]["topology"]], [g, c], model, step=3.3))


    make_sheet_random(args.sh, args.sw, [mols["aD-ribopyro"]["topology"], mols["aL-ribopyro"]["topology"]], [ad_ribose_conformer, al_ribose_conformer], model, args.lconc, step=8)
    if(args.verbose):
        print("Building system:", jobid)
    forcefield = ForceField('amber14-all.xml', 'tip3p.xml')
    forcefield.registerTemplateGenerator(gaff.generator)
    model.addSolvent(forcefield=forcefield, model='tip3p', boxSize = Vec3(args.sh + 1, args.sw + 1 , 3)*nanometers)
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

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    model.addExtraParticles(forcefield)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaDeviceIndex': str(device_idx), 'CudaPrecision': 'single'}

    simulation = Simulation(model.topology, system, integrator, platform, properties)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    # save pre-minimized positions as pdb
    PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open(f"{args.outdir}/preminimized{jobid}.pdb", 'w'))


    simulation.minimizeEnergy()

    simulation.reporters.append(StateDataReporter(f"{args.outdir}/output{jobid}.txt", args.report, step=True, potentialEnergy=True, temperature=True, speed=True))
    trajectory = []

    model_top = model.getTopology()
    for _ in range(0,args.nsteps, args.report):
        simulation.step(args.report)
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        positions = state.getPositions(asNumpy=True).tolist()
        velocities = state.getVelocities(asNumpy=True).tolist()
        frame = dict()
        
        for atom in model_top.atoms():
            resname = atom.residue.name + str(atom.residue.index)
            if("HOH" in resname):
                continue
            if(resname not in frame.keys()):
                frame[resname] = dict()
                frame[resname]['positions'] = []
                frame[resname]['velocities'] = []
                
            frame[resname]['positions'].append(positions[atom.index]) 
            frame[resname]['velocities'].append(velocities[atom.index])

        trajectory.append(frame)
    with open(f"{args.outdir}/output{jobid}.json", 'w') as f:
        f.write(json.dumps(trajectory))
    
def main():
    args = parse_args()
    total_sims = args.nsims
    gpus = args.ngpus
    proc = args.nprocs
    jobs = 0
    processes = []

    with tqdm(total=total_sims) as pbar:
        while jobs < total_sims:
            if(len(processes) < proc):
                print("Starting process", jobs)
                p = mp.Process(target=simulate, args=(jobs, (jobs % gpus), args))
                p.start()
                processes.append(p)
                jobs += 1
            for p in processes:
                if not p.is_alive():
                    processes.remove(p)
                    pbar.update(1)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()