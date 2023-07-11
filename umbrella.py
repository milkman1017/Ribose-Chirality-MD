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

def spawn_sugar(tops, poss, model, ribose_type, config):
    sheet_starting_index = model.topology.getNumAtoms()

    if ribose_type == 'D':
        topology, positions = tops[0], poss[0]
    elif ribose_type == 'L':
        topology, positions = tops[1], poss[1]

    #randomly set the initial x, y coords for ribose
    positions = translate(positions, random.uniform(0.5, 14.5),'x')
    positions = translate(positions, random.uniform(0.5, 14.5),'y')
    positions = rotate(positions, np.deg2rad(np.random.randint(0, 360)), np.random.choice(['x', 'y', 'z']))

    model.add(topology, positions)

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

def write_com(topology_list, successful_sims,target, ribose_type, config):
    outdir = config.get('Output Parameters','outdir')

    z_coordinates = []

    top_index = 0
    for i in successful_sims:

        try:
            top = md.Topology.from_openmm(topology_list[top_index])
            traj = md.load(f'traj_{i}_{ribose_type}.dcd', top=top)
            if ribose_type == 'D':
                res_indices = traj.topology.select('resname "DRIB"')
            elif ribose_type == 'L':
                res_indices = traj.topology.select('resname "LRIB"')
            
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

    nsteps = int(config.get('Simulation Parameters','number steps'))
    report = int(config.get('Simulation Parameters','report'))
    outdir = config.get('Output Parameters','outdir')

    mols = load_mols(["aD-ribopyro.sdf", 'aL-ribopyro.sdf', 'guanine.sdf', 'cytosine.sdf'], 
                    ['DRIB', 'LRIB', 'GUA', "CYT"])

    #generate residue template 
    gaff = GAFFTemplateGenerator(molecules = [mols[name]["mol"] for name in mols.keys()])

    #move ribose to target height 
    ad_ribose_conformer = translate(mols["aD-ribopyro"]["positions"], target*10, 'z')
    al_ribose_conformer = translate(mols["aL-ribopyro"]["positions"], target*10, 'z')

    if(config.get('Output Parameters','verbose')=='True'):
        print("Building molecules:", jobid)

    #line up the guanine and cytosines so that the molecules face eachother
    c = rotate(mols["cytosine"]["positions"], np.deg2rad(300), axis = 'z') 
    c = rotate(c, np.deg2rad(180), axis='y')
    c = rotate(c, np.deg2rad(190), axis='x')
    c = translate(c,1,'z')
    c = translate(c,4,'x')
    c = translate(c,4,'y')

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

    sugar_indices.append(spawn_sugar([mols["aD-ribopyro"]["topology"], mols["aL-ribopyro"]["topology"]], [ad_ribose_conformer, al_ribose_conformer], model, ribose_type,config))
    if(config.get('Output Parameters','verbose')=='True'):
        print("Building system:", jobid)
    forcefield = ForceField('amber14-all.xml', 'tip3p.xml')
    forcefield.registerTemplateGenerator(gaff.generator)

    box_size = [
        Vec3(1.5,0,0),
        Vec3(0,1.5,0),
        Vec3(0,0,end_z + 2.5)
    ]

    model.addSolvent(forcefield=forcefield, model='tip3p', boxSize=Vec3(1.5,1.5,end_z + 2.5 ))
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

    # PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), open(f"umbrella_first_frame_{np.round(target,3)}.pdb", 'w'))
    # simulation.reporters.append(PDBReporter(f'umbrella_{np.round(target,3)}.pdb', report))

    simulation.reporters.append(StateDataReporter(stdout, report, step=True,
        potentialEnergy=True, temperature=True, speed=True, time=True))
    
    #need to store the topologies because every sim has a slighlty different number of waters
    model_top = model.getTopology()

    file_handle = open(f"{outdir}/traj_{replicate}_{ribose_type}.dcd", 'bw')
    dcd_file = DCDFile(file_handle, model.topology, dt=stepsize)
    for step in range(0,nsteps, report):
        simulation.step(report)
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        dcd_file.writeModel(positions)
    file_handle.close()

    return model_top

def wham(ribose_type, config):
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

    gpus = int(config.get('Umbrella Setup','number gpus'))
    start_z = float(config.get('Umbrella Setup','start z'))
    end_z = float(config.get('Umbrella Setup','end z'))
    dz = float(config.get('Umbrella Setup','dz'))
    jobs = 0
    riboses = ['D','L']
    target = start_z

    PMF = {}

    for i in riboses:
        ribose_type = i
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
    print(keys)
    values = list(PMF.values())

    for i in range(0,len(keys),2):
        plt.plot(values[i],values[i+1], linewidth=1, label=f'{keys[i][0]}-Ribose')

    plt.xlabel('height above sheet (nm)')
    plt.ylabel('PMF (kJ/mol)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()