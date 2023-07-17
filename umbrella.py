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
import matplotlib.colors
import mdtraj as md
import argparse
import multiprocessing as mp
from tqdm import tqdm
import json
from simtk.openmm import app
import random as random
import scipy.optimize as optim
from FastMBAR import *
import configparser

def get_config():
    config = configparser.ConfigParser()
    config.read('umbrella_config.ini')
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

def write_com(topology_list, successful_sims,target, ribose_type, config):
    outdir = config.get('Output Parameters','outdir')

    z_coordinates = []

    top_index = 0
    for i in successful_sims:

        try:
            top = md.Topology.from_openmm(topology_list[top_index])
            traj = md.load(f'traj_{i}_{ribose_type}.dcd', top=top)

            res_indices = traj.topology.select(f'resname {ribose_type}')
            
            res_traj = traj.atom_slice(res_indices)

            com = md.compute_center_of_mass(res_traj)

            z_coordinates.append(com[:, 2])

            top_index += 1

        except Exception as e:
            print(f"Error loading trajectory traj_{i}_{ribose_type}.dcd:", e)

    if z_coordinates:
        z_coordinates = np.concatenate(z_coordinates)
        np.savetxt(f'{outdir}/com_heights_{np.round(target, 3)}_{ribose_type}.csv', z_coordinates, fmt='%.5f', delimiter=',')
    else:
        print("No available simulations for this target height")


def simulate(jobid, device_idx, target, end_z, replicate, ribose_type, config):
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
        Vec3(sheet_x+0.2,0,0),
        Vec3(0,sheet_y+0.2,0),
        Vec3(0,0,end_z + 2.5)
    ]

    model.addSolvent(forcefield=forcefield, model='tip3p', boxSize=Vec3(sheet_x, sheet_y, end_z + 2.5 ))
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
    custom_force = CustomExternalForce('0.5*j*((x-x)^2+(y-y)^2+(z-target)^2)')
    system.addForce(custom_force)
    custom_force.addGlobalParameter("target", target*nanometer)  
    custom_force.addGlobalParameter("j", 5000*kilojoules_per_mole/nanometer**2) 
    custom_force.addPerParticleParameter('z0')
    custom_force.addPerParticleParameter('x0')
    custom_force.addPerParticleParameter('y0')

    for start, stop in sugar_indices:
        for i in range(start, stop):
            custom_force.addParticle(i, model.positions[i])

    stepsize = 0.001*picoseconds

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, stepsize)
    model.addExtraParticles(forcefield)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaDeviceIndex': str(device_idx), 'CudaPrecision': 'single'}

    simulation = Simulation(model.topology, system, integrator, platform, properties)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    # save pre-minimized positions as pdb

    simulation.minimizeEnergy()

    PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open(f"umbrella_first_frame_{np.round(target,3)}.pdb", 'w'))
    # simulation.reporters.append(PDBReporter(f'umbrella_{np.round(target,3)}.pdb', report))

    simulation.reporters.append(StateDataReporter(stdout, report, step=True,
        potentialEnergy=True, temperature=True, speed=True, time=True))
    
    #need to store the topologies because every sim has a slighlty different number of waters
    model_top = model.getTopology()

    file_handle = open(f"{outdir}/umbrella_traj_{replicate}_{ribose_type}.dcd", 'bw')
    dcd_file = DCDFile(file_handle, model.topology, dt=stepsize)
    for step in range(0,nsteps, report):
        simulation.step(report)
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        dcd_file.writeModel(positions)
    file_handle.close()

    return model_top

def wham(ribose_type, config):
    # https://fastmbar.readthedocs.io/en/latest/butane_PMF.html
    outdir = config.get('Output Parameters','outdir')
    heights = []
    num_conf = []

    target_list = np.loadtxt(f'{outdir}/heights_{ribose_type}.csv', delimiter=',')

    for height_index in target_list:
        height = np.loadtxt(f'{outdir}/com_heights_{np.round(height_index,3)}_{ribose_type}.csv', delimiter = ',')
        heights.append(height)
        num_conf.append(len(height))

    heights = np.concatenate(heights)
    num_conf = np.array(num_conf).astype(np.float64)
    N = len(heights)
    
    ##compute reduced energy matrix A
    A = np.zeros((len(target_list),N))
    K = 5000
    T = 300 * kelvin
    kbT = BOLTZMANN_CONSTANT_kB * 298.15 * kelvin * AVOGADRO_CONSTANT_NA
    kbT = kbT.value_in_unit(kilojoule_per_mole)

    for height_index in range(len(target_list)):
        current_height = target_list[height_index]
        diff = np.abs(heights - current_height)
        diff = np.minimum(diff, 2*np.pi -diff)
        A[height_index,:] = 0.5*K*diff**2/kbT
    
    fastmbar = FastMBAR(energy=A, num_conf=num_conf, cuda=True, verbose=True)
    
    #compute reduced energy matrix B
    L = len(target_list)
    height_PMF = np.linspace(target_list[0], target_list[-1], L, endpoint=False)
    width = (target_list[-1] - target_list [0])/L
    B = np.zeros((L,N))

    for i in range(L):
        height_center = height_PMF[i]
        height_low = height_center - 0.5*width
        height_high = height_center + 0.5*width

        indicator = ((heights > height_low) & (heights <= height_high)) | \
            ((heights + (height_high - height_low) > height_low) & (heights + (height_high - height_low) <= height_high)) | \
            ((heights - (height_high - height_low) > height_low) & (heights - (height_high - height_low) <= height_high))

        B[i,~indicator] = np.inf
    calc_PMF, _ = fastmbar.calculate_free_energies_of_perturbed_states(B)
    height_PMF -= 0.1

    return height_PMF, calc_PMF

def main():
    config = get_config()
    nsims = int(config.get('Simulation Parameters','number sims'))
    nsteps = int(config.get('Simulation Parameters','number steps'))
    report = int(config.get('Simulation Parameters','report'))
    riboses = config.get('Input Setup','test resnames').split(',')

    gpus = int(config.get('Input Setup','number gpus'))
    start_z = float(config.get('Input Setup','start z'))
    end_z = float(config.get('Input Setup','end z'))
    dz = float(config.get('Input Setup','dz'))
    jobs = 0
    target = start_z

    PMF = {}

    for ribose_type in riboses:
        target=start_z
        target_list = []

        while target < end_z:
            replicate = 1
            topology_list = []
            successful_sims = []

            while replicate <= nsims:
                print(f'This is replicate {replicate} of target height {np.round(target,3)} nm for {ribose_type}-ribose')
                try:
                    topology_list.append(simulate(jobs, jobs%gpus, target, end_z, replicate, ribose_type, config))
                    successful_sims.append(replicate)
                    target_list.append(target)
                except KeyboardInterrupt:
                    print('Keyboard Interrupt')
                    return
                except:
                    print('Particle Coordinate is NaN')
                replicate+=1

            try:
                write_com(topology_list, successful_sims, target, ribose_type, config)
            except Exception as e:
                print(e)
                print('No available simulations for this target height')

            target += dz
        target_list = list(set(target_list))
        np.savetxt(f'heights_{ribose_type}.csv',target_list)
    
        height_key = f'{ribose_type}_height_PMF'
        calc_key = f'{ribose_type}_calc_PMF'

        PMF[height_key], PMF[calc_key] = wham(ribose_type ,config)
    
    keys = list(PMF.keys())
    values = list(PMF.values())

    for i in range(0,len(keys),2):
        plt.plot(values[i],values[i+1], linewidth=1, label=f'{keys[i][0]}-Ribose')

    plt.xlabel('height above sheet (nm)')
    plt.ylabel('PMF (kJ/mol)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()