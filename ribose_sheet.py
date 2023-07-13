import warnings
warnings.filterwarnings("ignore", message="importing 'simtk.openmm' is deprecated. Import 'openmm' instead.")

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openff.toolkit.topology import Molecule, Topology
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
import configparser
from rdkit import Chem
from rdkit.Chem import Draw
import parmed as pmd



def get_config():
    config = configparser.ConfigParser()
    config.read('sheet_config.ini')
    return config
    
def translate_one_axis(mol, step, axis='x'):
    if (axis == 'x'):
        mol += [step, 0, 0] * angstrom
    elif (axis == 'y'):
        mol += [0,step,0] * angstrom
    else:
        mol += [0,0,step] *angstrom
    return mol 

def translate_mol(positions, translation):
    translated_positions = positions + translation
    return translated_positions 

def rotate(mol, angle, axis='x'):
    centroid = np.mean(mol, axis=0)
    mol = translate_mol(mol, -centroid)

    if axis == 'x':
        x = np.array([[1,0,0],
                      [0,np.cos(angle), -np.sin(angle)],
                      [0,np.sin(angle), np.cos(angle)]])
        mol = np.array(mol)
        mol = mol[:,:]@x 
        mol = mol * angstrom
    elif axis == 'y':
        y = np.array([[np.cos(angle),0,np.sin(angle)],
                      [0,1,0],
                      [-np.sin(angle),0,np.cos(angle)]])
        mol = np.array(mol)
        mol = mol[:,:]@y
        mol = mol * angstrom
    else:
        z = np.array([[np.cos(angle),-np.sin(angle),0],
                      [np.sin(angle),np.cos(angle),0],
                      [0,0,1]])
        mol = np.array(mol)
        mol = mol[:,:]@z
        mol = mol * angstrom

    mol = translate_mol(mol, centroid)
    return mol

def check_overlap(pos, existing_molecules, boundaries):
    boundaries = np.asarray(boundaries) * angstrom * 10  #need *10 to convert from nm to angstrom
    for existing_pos in existing_molecules:
        dx = pos[0] - existing_pos[0]
        dy = pos[1] - existing_pos[1]
        dz = pos[2] - existing_pos[2]

        # account for periodic boundaries
        dx -= np.round(dx / boundaries[0][0]) * boundaries[0][0] 
        dy -= np.round(dy / boundaries[1][1]) * boundaries[1][1]
        dz -= np.round(dz / boundaries[2][2]) * boundaries[2][2] 

        if np.all(np.abs(dx) < 10) and np.all(np.abs(dy) < 10) and np.all(np.abs(dz) < 10):
            # The new molecule is within 1 nm in all dimensions of an existing molecule
            return True
        
    return False


def load_test_mols(filenames, resnames):
    """Loads a molecule from file.
    Args
    ====
    filenames (list) - list of molecule sdf files
    """
    mols = {}
    for filename, resname in zip(filenames, resnames):
        mol = Molecule.from_file(f'./molecules/{filename}', file_format='sdf')
        mol.generate_conformers()
        conf = to_openmm(mol.conformers[0])
        top = mol.to_topology().to_openmm()
        top = md.Topology.from_openmm(top)
        top.residue(0).name = resname
        top = top.to_openmm()
        mols[filename] = {
            "mol":mol,
            "topology": top,
            "positions": conf,
            'resname': resname
        }

    return mols

def spawn_test_mols(test_mol_names, test_mols, num_test_mols, model, sheet_x, sheet_y, config):
    test_mols_start = model.topology.getNumAtoms()
    existing_molecules = []
    boundaries = [
        Vec3(sheet_x - 0.5, 0, 0),
        Vec3(0, sheet_y - 0.5, 0),
        Vec3(0, 0, 9.5)
    ]

    for num in range(num_test_mols):
        for mol_name in test_mol_names:
            pos = test_mols[mol_name]['positions']
            top = test_mols[mol_name]['topology']

            translation = np.array([np.random.uniform(boundaries[0][0]), np.random.uniform(boundaries[1][1]), np.random.uniform(boundaries[2][2])]) * nanometer
            pos_temp = translate_mol(pos, translation)
            pos_temp = rotate(pos_temp, np.deg2rad(np.random.randint(0, 360)), axis=np.random.choice(['x', 'y', 'z']))

            while check_overlap(pos_temp, existing_molecules, boundaries):
                translation = np.array([np.random.uniform(boundaries[0][0]), np.random.uniform(boundaries[1][1]), np.random.uniform(boundaries[2][2])]) * nanometer
                pos_temp = translate_mol(pos, translation)
                pos_temp = rotate(pos_temp, np.deg2rad(np.random.randint(0, 360)), axis=np.random.choice(['x', 'y', 'z']))

            existing_molecules.append(pos_temp)
            model.add(top, pos_temp)

    return [test_mols_start, model.topology.getNumAtoms()]

def load_sheet_cell(cell_name, cell_res_names, config):
    cell_name = config.get('Sheet Setup','crystal structure')
    cell_res_names = config.get('Sheet Setup','crystal resnames').split(',')

    mol = PDBFile(f'./molecules/{cell_name}')

    cell_mols = config.get('Sheet Setup','mols in crystal').split(',')

    gaff_mols = []

    for mol_name, resname in zip(cell_mols,cell_res_names):
        gaff_mol = Molecule.from_file(f'./molecules/{mol_name}', file_format='sdf')
        gaff_mol.name = resname
        gaff_mols.append(gaff_mol)

    return mol, gaff_mols

def make_sheet(mol, model, config):
    sh = int(config.get('Sheet Setup','sheet height'))
    sw = int(config.get('Sheet Setup','sheet width'))
    res = config.get('Sheet Setup','crystal resnames').split(',')
    cell_name = config.get('Sheet Setup','crystal structure')

    sheet_mols_start = model.topology.getNumAtoms()

    pos = np.array(mol.getPositions(asNumpy=True)) * angstrom * 10
    pos[:,2] += 1 *  angstrom
    top = mol.getTopology()

    x_coords = pos[:,0]
    y_coords = pos[:,1]

    x_dimension = np.abs(np.max(x_coords)-np.min(x_coords)) + 2
    y_dimension = np.abs(np.max(y_coords)-np.min(y_coords)) + 1

    for i in range(sw):
        for j in range(sh):
            dx = i * x_dimension * angstrom
            dy = j * y_dimension * angstrom
            new_pos = pos.copy() * angstrom
            new_pos[:, 0] += dx
            new_pos[:, 1] += dy
            model.add(top, new_pos)

    sheet_pos = np.array(model.getPositions())*angstrom*10

    sheet_x_coords = sheet_pos[:,0]
    sheet_y_coords = sheet_pos[:,1]

    sheet_x_dim = np.abs(np.max(sheet_x_coords)-np.min(sheet_x_coords)) / 10 
    sheet_y_dim = np.abs(np.max(sheet_y_coords)-np.min(sheet_y_coords)) / 10 

    return[sheet_mols_start, model.topology.getNumAtoms()], sheet_x_dim, sheet_y_dim

def simulate(jobid, device_idx, config):
    print(device_idx)

    outdir  = config.get('Output Parameters','output directory')
    report = int(config.get('Output Parameters','report interval'))
    nsteps = int(config.get('Simulation Setup','number steps'))
    test_mol_names = config.get('Sheet Setup','test molecules').split(',')
    test_resnames = config.get('Sheet Setup','test resnames').split(',')
    num_test_mols = int(config.get('Sheet Setup','num of each mol'))
    cell_name = config.get('Sheet Setup','crystal structure')
    cell_res_names = config.get('Sheet Setup','crystal resnames')

    test_mols = load_test_mols(test_mol_names, test_resnames)
    unit_cell, cell_mols = load_sheet_cell(cell_name, cell_res_names, config)

    # initializing the modeler requires a topology and pos
    # we immediately empty the modeler for use later
    model = Modeller(test_mols[test_mol_names[0]]['topology'],test_mols[test_mol_names[0]]['positions'])
    model.delete(model.topology.atoms())

    #generate residue template 
    molecules = [test_mols[name]["mol"] for name in test_mols.keys()]
    molecules.extend(cell_mols)

    gaff = GAFFTemplateGenerator(molecules = molecules)

    if(config.get('Output Parameters','verbose')=='True'):
        print("Building molecules:", jobid)

    #make the sheet (height, width, make sure to pass in the guanine and cytosine confomrers (g and c) and their topologies)
    sheet_indices = []
    sheet_index, sheet_x, sheet_y= make_sheet(unit_cell, model, config)
    sheet_indices.append(sheet_index)
    
    test_mol_indices = []
    test_mol_indices.append(spawn_test_mols(test_mol_names, test_mols, num_test_mols, model, sheet_x, sheet_y, config))

    if(config.get('Output Parameters','verbose') == 'True'):
        print("Building system:", jobid)

    forcefield = ForceField('amber14-all.xml', 'tip3p.xml')
    forcefield.registerTemplateGenerator(gaff.generator)

    box_size = [
        Vec3(sheet_x+.2,0,0),
        Vec3(0,sheet_y+.2,0),
        Vec3(0,0,10)
    ] 

    model.addSolvent(forcefield=forcefield, model='tip3p', boxSize=Vec3(sheet_x, sheet_y, 9.95))
    model.topology.setPeriodicBoxVectors(box_size)

    system = forcefield.createSystem(model.topology,nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)

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
    # PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open("pre_energy_min.pdb", 'w'))

    simulation.minimizeEnergy()

    simulation.reporters.append(StateDataReporter(f"{outdir}/output{jobid}.txt", report, step=True, potentialEnergy=True, temperature=True, speed=True))
    with open (f'{outdir}/traj_{jobid}_sheet_{cell_name[:-4]}_mols_{test_resnames}_steps_{nsteps}.pdb','w') as topology_file:
        PDBFile.writeFile(simulation.topology, model.positions,topology_file)

    dcd_reporter = DCDReporter(f'{outdir}/traj_{jobid}_sheet_{cell_name[:-4]}_mols_{test_resnames}_steps_{nsteps}.dcd',report)
    simulation.reporters.append(dcd_reporter)

    simulation.step(nsteps)

def main():
    config = get_config()
    total_sims = int(config.get('Simulation Setup','number sims'))
    gpus = int(config.get('Simulation Setup','number gpus'))
    proc = int(config.get('Simulation Setup','number processes'))

    jobs = 0
    processes = []

    with tqdm(total=total_sims) as pbar:
        while jobs < total_sims:
            if(len(processes) < proc):
                print("Starting process", jobs)
                p = mp.Process(target=simulate, args=(jobs, (jobs % gpus), config))
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