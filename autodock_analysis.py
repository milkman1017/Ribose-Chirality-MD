import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def quaternion_to_euler_angle(x, y, z, w):
    """
    Convert a quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).
    """
    # Convert quaternion to rotation matrix
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])
    
    # Extract roll, pitch, and yaw from the rotation matrix
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    return roll, pitch, yaw


def read_autodock():

    with open('aL-ribopyro-GC-cell-x500.txt', 'r') as file:
        data = file.read()

    # Use regular expressions to find all instances of the relevant information
    runs = re.findall(r'Run: (\d+)', data)
    energy_bindings = re.findall(r'Estimated Free Energy of Binding\s+=\s+(-?\d+\.\d+)', data)
    inhibition_constants = re.findall(r'Estimated Inhibition Constant, Ki\s+=\s+([\d.]+)', data)
    centers = re.findall(r'trans (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)', data)
    quat_xyzws = re.findall(r'quatxyzw (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)', data)
    torsion_values_list = re.findall(r'ntor (\d+) (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)', data)

    # Create a list of extracted data for all runs
    extracted_data = []
    for i in range(len(runs)):
        data_row = [runs[i], energy_bindings[i], inhibition_constants[i]] + list(centers[i]) + list(quat_xyzws[i]) + list(torsion_values_list[i])
        extracted_data.append(data_row)

    # Write the data to a CSV file
    with open('output.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Run Number', 'Energy of Binding', 'Inhibition Constant', 'X', 'Y', 'Z', 'Quaternion X', 'Quaternion Y', 'Quaternion Z', 'Quaternion W', 'Torsion Count', 'Torsion 1', 'Torsion 2', 'Torsion 3', 'Torsion 4'])
        csv_writer.writerows(extracted_data)

    print("Data for all runs has been extracted and saved to 'output.csv'.")

def cluster_coordinates(data):
    data['Roll'], data['Pitch'], data['Yaw'] = zip(*data.apply(
    lambda row: quaternion_to_euler_angle(row['Quaternion X'], row['Quaternion Y'], row['Quaternion Z'], row['Quaternion W']),
    axis=1
))
    
    PCA_columns = ['X','Y','Z','Torsion 1','Torsion 2','Torsion 3','Torsion 4','Roll','Pitch','Yaw']
    X = data[PCA_columns]

    # Center mean and scale to variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_

    pca_data = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(2)])
    pca_data['Molecule Name'] = data['Molecule Name']

    pca_data = pca_data.merge(data[['Molecule Name', 'Energy of Binding']], on='Molecule Name', how='left')

    colors = {'aD-ribopyro': 'Blues', 'aL-ribopyro': 'Reds'}

    # Create a figure with subplots for individual color bars
    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = {}  # Create a colormap for energy values within each molecule's color range

    for molecule, group in pca_data.groupby('Molecule Name'):
        max_energy = group['Energy of Binding'].max()
        min_energy = group['Energy of Binding'].min()
        cmap[molecule] = plt.cm.get_cmap(colors[molecule])(np.linspace(0, 1, int(len(group))))
        cmap[molecule][:, :3] = cmap[molecule][:, :3] * (1 - (group['Energy of Binding'] - min_energy) / (max_energy - min_energy))[:, np.newaxis]

        # Plot the scatter plot with the respective colormap
        scatter = ax.scatter(group['PC1'], group['PC2'], label=molecule, color=cmap[molecule], s=5)
        ax.set_title('PCA of Docked Conformations')
        ax.set_xlabel(f'PC1 ({np.round(explained_variance_ratio[0], 4) * 100}% of variance)')
        ax.set_ylabel(f'PC2 ({np.round(explained_variance_ratio[1], 4) * 100}% of variance)')

    loadings = pca.components_.T

    for i, (pc1, pc2) in enumerate(loadings):
        ax.arrow(0, 0, pc1, pc2, color='black', alpha=0.5, width=0.005, head_width=0.1, head_length=0.1)
        ax.text(pc1, pc2, PCA_columns[i], fontsize=12, ha='center', va='center', color='black')

    plt.show()

def energy_hist(data):
    fig, ax = plt.subplots()

    for mol in list(set(data['Molecule Name'])):
        subset = data[data['Molecule Name'] == mol]

        color = sns.color_palette()[list(set(data['Molecule Name'])).index(mol)]  # Get the color for the current molecule
        sns.kdeplot(data=subset, x='Energy of Binding', linewidth=1, label=mol, color=color)
        mean_energy = subset['Energy of Binding'].mean()
        
        ax.axvline(mean_energy, linestyle='--', color=color)
        
        ax.text(mean_energy, 0.95, f'Mean: {mean_energy:.2f}', color=color, rotation=90, verticalalignment='top')

    plt.legend()
    plt.show()

def std(df):
    print(df)
    std_df = df.std(numeric_only=True)
    print(std_df)

def physical_scatter(df):
    # Create separate scatter plots for each torsion angle vs. binding energy, color-coded by molecule
    plt.figure(figsize=(12, 8))

    for i, torsion_angle in enumerate(["Torsion 1", "Torsion 2", "Torsion 3", "Torsion 4"]):
        df[torsion_angle] = np.radians(df[torsion_angle])
        plt.subplot(2, 2, i + 1, projection='polar')
        sns.scatterplot(data=df, x=torsion_angle, y="Energy of Binding", hue="Molecule Name", palette="Set1")
        plt.title(f"{torsion_angle} vs. Binding Energy")
        plt.xlabel(torsion_angle)
        plt.ylabel("Energy of Binding")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))


    df['Roll'], df['Pitch'], df['Yaw'] = zip(*df.apply(
        lambda row: quaternion_to_euler_angle(row['Quaternion X'], row['Quaternion Y'], row['Quaternion Z'], row['Quaternion W']),
        axis=1
    ))

    for i, angle in enumerate(['Roll','Pitch','Yaw']):
        plt.subplot(2, 2, i+1, projection='polar')
        sns.scatterplot(data=df, x=angle, y='Energy of Binding', hue='Molecule Name', palette='Set1')
        plt.title(f'{angle} vs Binding Energy')
        plt.xlabel(angle)
        plt.ylabel('Energy of Binding')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))

    for i, com in enumerate(['X','Y','Z']):
        plt.subplot(2,2,i+1)
        sns.scatterplot(data=df, x=com, y='Energy of Binding', hue='Molecule Name', palette='Set1')
        plt.title(f'{com} vs Binding Energy')
        plt.xlabel(com)
        plt.ylabel('Energy of Binding')

    plt.tight_layout()
    plt.show()

def correlation(df):
    df['Roll'], df['Pitch'], df['Yaw'] = zip(*df.apply(
        lambda row: quaternion_to_euler_angle(row['Quaternion X'], row['Quaternion Y'], row['Quaternion Z'], row['Quaternion W']),
        axis=1
    ))
    
    analysis_columns = ['X', 'Y', 'Z', 'Torsion 1', 'Torsion 2', 'Torsion 3', 'Torsion 4', 'Roll', 'Pitch', 'Yaw', 'Molecule Name']
    X = df[analysis_columns]

    grouped = X.groupby('Molecule Name')

    for name, group in grouped:

        correlations = group.corr()
        correlations = correlations.unstack().sort_values(ascending=False)
        

        correlations = correlations[correlations != 1.0]
        correlations = correlations.drop_duplicates()
        top_correlations = correlations.head(5)
        bottom_correlations = correlations.tail(5)
        
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
        fig.suptitle(f'Molecule: {name}', fontsize=16)
        
        for i, (col1, col2) in enumerate(top_correlations.index):
            ax = axes[i // 5, i % 5]
            group.plot.scatter(x=col1, y=col2, ax=ax, s=2)
            ax.set_title(f'Correlation: {top_correlations[i]:.2f}')
        
        for i, (col1, col2) in enumerate(bottom_correlations.index):
            ax = axes[1, i]
            group.plot.scatter(x=col1, y=col2, ax=ax, s=2)
            ax.set_title(f'Correlation: {bottom_correlations[i]:.2f}')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # plt.savefig(f'{name}_scatter_plot.png')
        plt.show()

def main():
    # read_autodock()

    data = pd.read_csv('autodock_results.csv')
    # cluster_coordinates(data)

    # energy_hist(data)

    # std(data)

    physical_scatter(data)

    # correlation(data)
    
if __name__ == "__main__":
    main()