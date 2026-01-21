"""
Basic Higgs -> gamma gamma Analysis
Uses real ATLAS Open Data from CERN servers
"""

import uproot
import awkward as ak
import vector
import matplotlib.pyplot as plt
import numpy as np
import os


def load_data(filename):
    """Load diphoton data from ATLAS servers"""
    
    base_url = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/GamGam/Data/"
    url = base_url + filename
    
    print("Loading data: " + filename + " ...")
    
    # Open ROOT file and get the tree
    file = uproot.open(url + ":mini")
    
    # Get photon variables
    data = file.arrays(["photon_pt", "photon_eta", "photon_phi", "photon_E", "photon_n"])
    
    print("Total events: " + str(len(data)))
    return data


def select_events(data):
    """Keep only events with at least 2 photons"""
    
    mask = data["photon_n"] >= 2
    filtered = data[mask]
    
    n_passed = len(filtered)
    n_total = len(data)
    percent = 100 * n_passed / n_total
    
    print("Diphoton events: " + str(n_passed) + " (" + str(round(percent, 1)) + "%)")
    return filtered


def calculate_mass(events):
    """Calculate invariant mass of two photons"""
    
    # Create 4-vectors for each photon
    photons = vector.zip({
        "pt": events["photon_pt"],
        "eta": events["photon_eta"],
        "phi": events["photon_phi"],
        "E": events["photon_E"]
    })
    
    # Get first and second photon
    photon1 = photons[:, 0]
    photon2 = photons[:, 1]
    
    # Add 4-vectors and get mass
    # This uses: m^2 = (E1+E2)^2 - (p1+p2)^2
    diphoton = photon1 + photon2
    mass = diphoton.mass / 1000.0  # Convert MeV to GeV
    
    # Convert to numpy array
    mass_array = ak.to_numpy(mass)
    return mass_array


def make_plot(mass, save_path=None):
    """Create invariant mass histogram"""
    
    # Create figure
    fig = plt.figure(figsize=(10, 7))
    
    # Make histogram
    counts, bins, patches = plt.hist(
        mass, 
        bins=60, 
        range=(100, 160),
        color='skyblue',
        edgecolor='steelblue',
        label='ATLAS Data'
    )
    
    # Mark where Higgs should be
    plt.axvline(x=125, color='red', linestyle='--', linewidth=2, label='Higgs (125 GeV)')
    
    # Highlight signal region
    plt.axvspan(120, 130, alpha=0.2, color='red', label='Signal Region')
    
    # Labels
    plt.title("ATLAS Open Data: Diphoton Invariant Mass", fontsize=14)
    plt.xlabel("Invariant Mass [GeV]", fontsize=12)
    plt.ylabel("Events / 1 GeV", fontsize=12)
    
    # Add arrow pointing to Higgs region
    max_count = max(counts)
    plt.annotate(
        'Higgs Signal?\n(~125 GeV)',
        xy=(125, max_count * 0.6),
        xytext=(145, max_count * 0.85),
        fontsize=11,
        arrowprops=dict(arrowstyle='->', color='darkred', lw=2)
    )
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(100, 160)
    
    # Save if path given
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path, dpi=150)
        print("Plot saved: " + save_path)
    
    plt.show()


def main():
    """Main function"""
    
    print("=" * 50)
    print("ATLAS Higgs -> gamma gamma Analysis")
    print("=" * 50)
    
    # Use only period A data (smaller file for testing)
    filename = "data_A.GamGam.root"
    
    # Step 1: Load data
    data = load_data(filename)
    
    # Step 2: Select diphoton events
    diphoton_events = select_events(data)
    
    # Step 3: Calculate invariant mass
    print("Calculating invariant mass...")
    mass = calculate_mass(diphoton_events)
    
    # Step 4: Print statistics
    mass_in_range = mass[(mass > 100) & (mass < 160)]
    print("")
    print("Statistics (100-160 GeV):")
    print("  Events: " + str(len(mass_in_range)))
    print("  Mean mass: " + str(round(np.mean(mass_in_range), 2)) + " GeV")
    print("  Std dev: " + str(round(np.std(mass_in_range), 2)) + " GeV")
    
    # Step 5: Make plot
    script_folder = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_folder, "..", "results", "diphoton_mass_basic.png")
    
    print("")
    print("Creating histogram...")
    make_plot(mass, save_path)
    
    print("")
    print("=" * 50)
    print("Analysis complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
