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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nprocs', type=int, default=8, help='number of simultaneously processes to use')
    parser.add_argument('--outdir', type=str, default='.', help='output directory')
    parser.add_argument('--nsims', type=int, default=8, help='number of simulations to run per umbrella')
    parser.add_argument('--ngpus', type=int, default=1, help='number of gpus on the system')
    parser.add_argument('--nsteps', type=int, default=100000, help='number of steps')
    parser.add_argument('--report', type=int, default=500, help='report interval')
    parser.add_argument('--verbose', action='store_true', help='print verbose output')
    parser.add_argument('--start_z', type=float, default=1, help='starting target z coordinate for umbrella sampling')
    parser.add_argument('--end_z', type=float, default=10, help='ending z coordinate for umbrella sampling')
    parser.add_argument('--dz', type=float, default=0.5, help='how much to increase the target z coordinate for each umbrella')
    parser.add_argument('--ribose', choices=['D','L'], default='D', help='chose which ribose to simulate: D, L, or one of each')
    args = parser.parse_args()
    return args

args = parse_args()  

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

def write_com(topology_list, target, args):
    top_index = 0
    z_coordinates = []

    for replicate in range(args.nsims):
        replicate += 1
        top = md.Topology.from_openmm(topology_list[top_index])
        traj = md.load_dcd(f'traj_{replicate}_{args.ribose}.dcd', top = top)

        res_indicies = traj.topology.select('resname "DRIB"')
        res_traj = traj.atom_slice(res_indicies)

        com = md.compute_center_of_mass(res_traj)

        z_coordinates.append(com[:,2])
        top_index+=1

    z_coordinates = np.concatenate(z_coordinates) 
    np.savetxt(f'com_heights_{np.round(target,3)}_{args.ribose}.csv', z_coordinates, fmt='%.5f', delimiter=',')  

def simulate(jobid, device_idx, start_z, end_z, dz, args):
    target=start_z

    target_list = []
    while target < end_z:
        target_list.append(target)
        target += dz

    np.savetxt('heights.csv', target_list, delimiter = ',')
    target = start_z

    while target < end_z:
        replicate = 1
        topology_list = []

        while replicate <= args.nsims:
            print(f'This is replicate {replicate} of target height {np.round(target,3)} nm for {args.ribose}-ribose')
           

            mols = load_mols(["aD-ribopyro.sdf", 'aL-ribopyro.sdf', 'guanine.sdf', 'cytosine.sdf'], 
                            ['DRIB', 'LRIB', 'GUA', "CYT"])

            #generate residue template 
            gaff = GAFFTemplateGenerator(molecules = [mols[name]["mol"] for name in mols.keys()])

            #move ribose to target height 
            ad_ribose_conformer = translate(mols["aD-ribopyro"]["positions"], target*10, 'z')
            al_ribose_conformer = translate(mols["aL-ribopyro"]["positions"], target*10, 'z')

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
                Vec3(0,0,end_z + 5)
            ]

            model.addSolvent(forcefield=forcefield, model='tip3p', boxSize=Vec3(1.5,1.5,end_z + 5 ))
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
            custom_force.addGlobalParameter("j", 1000*kilojoules_per_mole/nanometer**2) 
            custom_force.addPerParticleParameter('z0')
            custom_force.addPerParticleParameter('x0')
            custom_force.addPerParticleParameter('y0')

            for start, stop in sugar_indices:
                for i in range(start, stop):
                    custom_force.addParticle(i, model.positions[i])

            stepsize = 0.002*picoseconds

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
            # simulation.reporters.append(PDBReporter(f'umbrella_{np.round(target,3)}.pdb', args.report))

            simulation.reporters.append(StateDataReporter(stdout, args.report, step=True,
                potentialEnergy=True, temperature=True, speed=True, time=True))
            
            #need to store the topologies because every sim has a slighlty different number of waters
            model_top = model.getTopology()
            topology_list.append(model_top)

            file_handle = open(f"traj_{replicate}_{args.ribose}.dcd", 'bw')
            dcd_file = DCDFile(file_handle, model.topology, dt=stepsize)
            for step in range(0,args.nsteps, args.report):
                simulation.step(args.report)
                state = simulation.context.getState(getPositions=True)
                positions = state.getPositions()
                dcd_file.writeModel(positions)
            file_handle.close()
            replicate += 1

        write_com(topology_list, target, args)
        target += dz

    return model_top, target_list, topology_list


simulate(1, 0, args.start_z, args.end_z, args.dz, args)

def wham(ribose_type):
    heights = []
    num_conf = []

    target_list = np.loadtxt('heights.csv', delimiter=',')

    for height_index in target_list:
        height = np.loadtxt(f'com_heights_{np.round(height_index,3)}_{ribose_type}.csv', delimiter = ',')
        heights.append(height)
        num_conf.append(len(height))

    heights = np.concatenate(heights)
    num_conf = np.array(num_conf).astype(np.float64)
    N = len(heights)
    
    ##compute reduced energy matrix A
    A = np.zeros((len(target_list),N))
    K = 1000
    T = 300 * kelvin
    kbT = BOLTZMANN_CONSTANT_kB * 298.15 * kelvin * AVOGADRO_CONSTANT_NA
    kbT = kbT.value_in_unit(kilojoule_per_mole)

    height_0 = np.loadtxt('heights.csv', delimiter=',')

    for height_index in range(len(target_list)):
        current_height = height_0[height_index]
        diff = np.abs(heights - current_height)
        diff = np.minimum(diff, 2*np.pi -diff)
        A[height_index,:] = 0.5*K*diff**2/kbT
    
    fastmbar = FastMBAR(energy=A, num_conf=num_conf, cuda=False, verbose=True)
    
    #compute reduced energy matrix B
    L = len(target_list)
    height_PMF = np.linspace(args.start_z, args.end_z, L, endpoint=False)
    width = (args.end_z-args.start_z)/L
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

D_height_PMF, D_calc_PMF = wham('D')
L_height_PMF, L_calc_PMF = wham('L')

plt.plot(D_height_PMF,D_calc_PMF, linewidth=1)
plt.plot(L_height_PMF,L_calc_PMF,linewidth=1)
plt.xlabel('z coordinate')
plt.ylabel('PMF')
plt.legend(['D-ribose','L-ribose'],bbox_to_anchor=(1,1),loc=2)
plt.show()

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
#     target = height_start 

#     while z < height_end:
#         while jobs < sims:
#             if len(processes) < proc: 
#                 print('starting process', jobs)
#                 p = mp.Process(target=simulate, args=(jobs, (jobs%gpus), target, args))
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