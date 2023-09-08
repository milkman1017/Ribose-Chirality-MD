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
    config.read('torsion_umbrella_config.ini')
    return config
    
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


def load_test_mols(filename, resname):
    """Loads a molecule from file.
    Args
    ====
    filenames (list) - list of molecule sdf files
    """
    mols = {}

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

def spawn_test_mols(test_mols, test_mol, model, config):
    test_mols_start = model.topology.getNumAtoms()
    existing_molecules = []
    boundaries = [
        Vec3(1.1654 - 0.5, 0, 0),      #these values are the sheet x and y dimensions
        Vec3(0, 1.3288 - 0.5, 0),      #had to hard code these in the test mol could be spawned in first and make atom selection for torsion much easier
        Vec3(0, 0, 1.1654)             #if you change the sheet, check the x and y dims, and change these values to fit
    ]

    pos = test_mols[test_mol]['positions']
    top = test_mols[test_mol]['topology']

    translation = np.array([np.random.uniform(boundaries[0][0]), np.random.uniform(boundaries[1][1]), np.random.uniform(boundaries[2][2])]) * nanometer
    pos += translation 
    pos = rotate(pos, np.deg2rad(np.random.randint(0, 360)), axis=np.random.choice(['x', 'y', 'z']))

    model.add(top, pos)

    return [test_mols_start-1, model.topology.getNumAtoms()-1]

def load_sheet_cell(cell_name, cell_res_names, config):
    cell_name = config.get('Umbrella Setup','crystal structure')
    cell_res_names = config.get('Umbrella Setup','crystal resnames').split(',')

    mol = PDBFile(f'./molecules/{cell_name}')

    cell_mols = config.get('Umbrella Setup','mols in crystal').split(',')

    gaff_mols = []

    for mol_name, resname in zip(cell_mols,cell_res_names):
        gaff_mol = Molecule.from_file(f'./molecules/{mol_name}', file_format='sdf')
        gaff_mol.name = resname
        gaff_mols.append(gaff_mol)

    return mol, gaff_mols

def make_sheet(mol, model, config):
    sh = 1
    sw = 1
    res = config.get('Umbrella Setup','crystal resnames').split(',')
    cell_name = config.get('Umbrella Setup','crystal structure')

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

    return[sheet_mols_start, model.topology.getNumAtoms()-1], sheet_x_dim, sheet_y_dim

def simulate(jobid, device_idx, theta0, replicate, test_mol, test_resname, config):

    nsteps = int(config.get('Simulation Setup','number steps'))
    report = int(config.get('Simulation Setup','report'))
    restraint_force = int(config.get('Simulation Setup','restraining force'))
    outdir = config.get('Output Parameters','outdir')
    cell_name = config.get('Umbrella Setup','crystal structure')
    cell_res_names = config.get('Umbrella Setup','crystal resnames').split(',')


    test_mol_info = load_test_mols(test_mol, test_resname)
    unit_cell, cell_mols = load_sheet_cell(cell_name, cell_res_names, config)

    # initializing the modeler requires a topology and pos
    # we immediately empty the modeler for use later

    model = Modeller(test_mol_info[test_mol]['topology'],test_mol_info[test_mol]['positions'])
    model.delete(model.topology.atoms())

    #generate residue template 
    molecules = [test_mol_info[test_mol]["mol"]]
    molecules.extend(cell_mols)

    gaff = GAFFTemplateGenerator(molecules = molecules)

    if(config.get('Output Parameters','verbose')=='True'):
        print("Building molecules:", jobid)

    sugar_indices = []
    sugar_indices.append(spawn_test_mols(test_mol_info, test_mol, model, config))

    #make the sheet (height, width, make sure to pass in the guanine and cytosine confomrers (g and c) and their topologies)
    sheet_indices = []
    sheet_index, sheet_x, sheet_y = make_sheet(unit_cell, model, config)
    sheet_indices.append(sheet_index)
    
    if(config.get('Output Parameters','verbose')=='True'):
        print("Building system:", jobid)

    forcefield = ForceField('amber14-all.xml', 'tip3p.xml')
    forcefield.registerTemplateGenerator(gaff.generator)

    model.addSolvent(forcefield=forcefield, model='tip3p', padding = 0.1*nanometer)
    # model.topology.setPeriodicBoxVectors(box_size)

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
    bias_torsion = CustomTorsionForce("0.5*K*dtheta^2; dtheta = min(diff, 2*Pi-diff); diff = abs(theta - theta0)")
    bias_torsion.addGlobalParameter("Pi", np.pi)
    bias_torsion.addGlobalParameter("K", restraint_force)
    bias_torsion.addGlobalParameter("theta0", theta0)

    torsion_atoms_idx = [17,10,1,13]
    torsion_atoms = []

    for start, stop in sugar_indices:
            for torsion_idx in torsion_atoms_idx:
                for mol_atom in range(start, stop):
                    if mol_atom % (stop-start) == torsion_idx:
                        torsion_atoms.append(mol_atom)

    bias_torsion.addTorsion(torsion_atoms[0],torsion_atoms[1],torsion_atoms[2],torsion_atoms[3])
    system.addForce(bias_torsion)

    stepsize = 0.001*picoseconds

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, stepsize)
    model.addExtraParticles(forcefield)
    platform = Platform.getPlatformByName('CUDA')
    properties = {'CudaDeviceIndex': str(device_idx), 'CudaPrecision': 'single'}

    simulation = Simulation(model.topology, system, integrator, platform, properties)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300*kelvin)
    simulation.context.setParameter("theta0", np.pi/2)
    # save pre-minimized positions as pdb

    simulation.minimizeEnergy()

    #NPT equilibration (https://github.com/openmm/openmm/issues/3782)
    equilibration_steps = 50000
    barostat = system.addForce(MonteCarloBarostat(1*atmosphere, 300*kelvin))
    simulation.context.reinitialize(True)
    print('Running NPT equil')
    for i in range(10):
        print('equil step, ', i)
        simulation.step(int(equilibration_steps/100))

    #save the equilibration results
    simulation.saveState((f'{outdir}/equilibrium.state'))
    simulation.saveCheckpoint((f'{outdir}/equilibrium.chk'))
    
    #load checkpoint and reset step and time counters
    simulation.loadCheckpoint((f'{outdir}/equilibrium.chk'))
    eq_state = simulation.context.getState(getVelocities=True, getPositions=True)
    positions = eq_state.getPositions()
    velocities = eq_state.getVelocities()

    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, stepsize)
    simulation = Simulation(model.topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)

    # PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open(f"eq_umbrella_first_frame_{np.round(theta0,3)}.pdb", 'w'))
    # # simulation.reporters.append(PDBReporter(f'umbrella_{np.round(target,3)}.pdb', report))
    # print('saved first frame')

    simulation.reporters.append(StateDataReporter(stdout, report, step=True,
        potentialEnergy=True, temperature=True, speed=True, time=True))

    #need to store the topologies because every sim has a slighlty different number of waters
    model_top = model.getTopology()

    file_handle = open(f"{outdir}/traj_{replicate}_{test_mol}.dcd", 'bw')
    dcd_file = DCDFile(file_handle, model.topology, dt=stepsize)
    for step in range(0,nsteps, report):
        simulation.step(report)
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        dcd_file.writeModel(positions)
    file_handle.close()

    return model_top

def collect_torsions(topology_list, successful_sims, theta0, current_mol, current_resname, config):
    outdir = config.get('Output Parameters','outdir')
    nsims = int(config.get('Simulation Setup','number sims'))

    torsion_atoms_idx = [17,10,1,13]
    torsions = []

    top_index = 0
    for i in successful_sims:

        top = md.Topology.from_openmm(topology_list[top_index])
        traj = md.load(f'{outdir}/traj_{i}_{current_mol}.dcd', top=top)

        theta = md.compute_dihedrals(traj, [torsion_atoms_idx])
        torsions.append(theta.flatten().tolist())
        top_index += 1

    torsions = [torsion for torsions_list in torsions for torsion in torsions_list]
    if torsions:
        np.savetxt(f'{outdir}/dihedral_{np.round(theta0,4)}_mol_{current_resname}.csv', torsions, fmt='%.5f', delimiter=',')
    else:
        print('No available simulations for this target')

def wham(test_mol, current_mol, current_resname, config):
    outdir = config.get('Output Parameters','outdir')

    centers = np.loadtxt(f'{outdir}/theta0_{current_mol}.csv', delimiter=',')
    M = len(centers)
    
    thetas = []
    num_conf = []

    for theta0 in centers:
        theta = np.loadtxt(f'dihedral_{np.round(theta0,4)}_mol_{current_resname}.csv', delimiter=',')
        thetas.append(theta)
        num_conf.append(len(theta))

    thetas = np.concatenate(thetas)
    num_conf = np.array(num_conf)
    N = len(thetas)

    A = np.zeros((M,N))
    K = int(config.get('Simulation Setup','restraining force'))
    T = 300 * kelvin
    kbT = BOLTZMANN_CONSTANT_kB * T * AVOGADRO_CONSTANT_NA
    kbT = kbT.value_in_unit(kilojoule_per_mole)

    for i, theta0 in enumerate(centers):
        diff = np.abs(thetas-theta0)
        diff = np.minimum(diff, 2*np.pi-diff)
        A[i, :] = 0.5*K*diff**2/kbT
    
    fastmbar = FastMBAR(energy = A, num_conf = num_conf, cuda=False, verbose=True)
    print('relative free energies: ', fastmbar.F)

    L = M
    theta_PMF = np.linspace(-np.pi, np.pi, L, endpoint=False)
    width = 2*np.pi/L
    B = np.zeros((L,N))

    for i in range(L):
        theta_center = theta_PMF[i]
        theta_low = theta_center - 0.5*width
        theta_high = theta_center + 0.5*width

        indicator = ((thetas > theta_low) & (thetas <= theta_high)) | \
                 ((thetas + 2*math.pi > theta_low) & (thetas + 2*math.pi <= theta_high)) | \
                 ((thetas - 2*math.pi > theta_low) & (thetas - 2*math.pi <= theta_high))

        B[i, ~indicator] = np.inf

    results = fastmbar.calculate_free_energies_of_perturbed_states(B)
    calc_PMF = results[0]

    return theta_PMF, calc_PMF

def main():
    config = get_config()
    nsims = int(config.get('Simulation Setup','number sims'))
    nsteps = int(config.get('Simulation Setup','number steps'))
    report = int(config.get('Simulation Setup','report'))
    outdir = config.get('Output Parameters','outdir')

    gpus = int(config.get('Umbrella Setup','number gpus'))
    test_mols = config.get('Umbrella Setup','test molecules').split(',')
    test_resnames = config.get('Umbrella Setup','test resnames').split(',')

    M = int(config.get('Umbrella Setup','centers')) #M centers of harmonic biasing potentials
    theta0 = np.linspace(-np.pi, np.pi, M, endpoint=False)

    cell = config.get("Umbrella Setup",'crystal structure')

    jobs = 0

    PMF = {}

    for i, current_mol in enumerate(test_mols):
        target_list = []

        #set up torsion umbrella sampling (https://fastmbar.readthedocs.io/en/latest/butane_PMF.html)
        for theta0_index in range(M):
            replicate = 1
            topology_list = []
            successful_sims = []

            while replicate <= nsims:
                print(f'Sampling replicate {replicate} for theta {theta0_index} out of {M} for mol {current_mol}')
                try:
                    topology_list.append(simulate(jobs, jobs%gpus, theta0[theta0_index], replicate, current_mol, test_resnames[i], config))
                    successful_sims.append(replicate)
                    target_list.append(theta0[theta0_index])
                except KeyboardInterrupt:
                    print('Keyboard Interrupt')
                    return
                except Exception as e:
                    print(e)
                replicate += 1
            try:
                collect_torsions(topology_list, successful_sims, theta0[theta0_index], current_mol, test_resnames[i], config)
            except Exception as e:
                print(e)
                print('No available sims for this target')

        theta0_list = list(set(target_list))
        np.savetxt(f'{outdir}/theta0_{current_mol}.csv', theta0_list, delimiter=',')

        height_key = f'{current_mol}_height_PMF'
        calc_key = f'{current_mol}_calc_PMF'

        PMF[height_key], PMF[calc_key] = wham(current_mol, current_mol, test_resnames[i], config)
    
    keys = list(PMF.keys())
    values = list(PMF.values())

    for i in range(0,len(keys),2):

        plt.plot(values[i],values[i+1], linewidth=1, label=f'{keys[i][:-15]}')

    plt.xlabel('Angle of Torsion (rads)')
    plt.ylabel('PMF (kJ/mol)')
    plt.legend()
    plt.savefig(f'{outdir}/Umbrella_graph_{test_resnames}_{cell[:-4]}.png',dpi=400)
    # plt.show()

if __name__ == "__main__":
    main()
