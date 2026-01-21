"""
- Loads all data periods (A, B, C, D)
- Fits background with polynomial
- Extracts signal by subtracting background
"""

import uproot
import awkward as ak
import vector
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os

# Fit functions
def gaussian(x, amplitude, mean, sigma):
    """Gaussian function for Higgs signal"""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2)


def polynomial(x, p0, p1, p2, p3):
    """3rd degree polynomial for background"""
    return p0 + p1*x + p2*x**2 + p3*x**3


def signal_plus_background(x, amp, mean, sigma, p0, p1, p2, p3):
    """Full model: Gaussian signal + polynomial background"""
    signal = gaussian(x, amp, mean, sigma)
    background = polynomial(x, p0, p1, p2, p3)
    return signal + background


# Loading the data

def load_all_periods():
    """Load and combine data from all periods (A, B, C, D)"""
    
    base_url = "https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/GamGam/Data/"
    periods = ["A", "B", "C", "D"]
    
    all_masses = []
    total_events = 0
    
    for period in periods:
        filename = "data_" + period + ".GamGam.root"
        url = base_url + filename
        
        print("Loading: data_" + period + " ...", end=" ")
        
        try:
            # get data
            file = uproot.open(url + ":mini")
            data = file.arrays(["photon_pt", "photon_eta", "photon_phi", "photon_E", "photon_n"])
            
            # Select events with 2+ photons
            mask = data["photon_n"] >= 2
            filtered = data[mask]
            
            # Build 4-vectors
            photons = vector.zip({
                "pt": filtered["photon_pt"],
                "eta": filtered["photon_eta"],
                "phi": filtered["photon_phi"],
                "E": filtered["photon_E"]
            })
            
            # Calculate mass
            mass = (photons[:, 0] + photons[:, 1]).mass / 1000.0
            all_masses.extend(ak.to_numpy(mass))
            
            n_events = len(filtered)
            total_events = total_events + n_events
            print("OK - " + str(n_events) + " events")
            
        except Exception as e:
            print("ERROR: " + str(e))
    
    print("")
    print("Total combined events: " + str(total_events))
    return np.array(all_masses)


# fitting functions

def fit_background(bin_centers, counts, errors):
    """
    Fit background using sideband method:
    - Mask signal region (120-130 GeV)
    - Fit polynomial to sidebands only
    """
    
    # Mask signal region
    mask = (bin_centers < 120) | (bin_centers > 130)
    
    x_sideband = bin_centers[mask]
    y_sideband = counts[mask]
    err_sideband = errors[mask]
    
    # Fix zero errors
    err_sideband = np.where(err_sideband == 0, 1, err_sideband)
    
    # Do the fit
    try:
        initial_params = [10000, -100, 1, -0.01]
        
        params, covariance = curve_fit(
            polynomial, 
            x_sideband, 
            y_sideband,
            sigma=err_sideband,
            p0=initial_params,
            maxfev=10000
        )
        
        # Calculate chi-square
        y_fit = polynomial(x_sideband, *params)
        chi2 = np.sum(((y_sideband - y_fit) / err_sideband)**2)
        ndof = len(x_sideband) - len(params)
        
        print("Background fit quality: chi2/ndof = " + str(round(chi2/ndof, 2)))
        return params
        
    except Exception as e:
        print("Fit error: " + str(e))
        return None


def fit_full_model(bin_centers, counts, errors):
    """Fit signal + background model to all data"""
    
    # Fix zero errors
    errors = np.where(errors == 0, 1, errors)
    
    # Initial parameters: [amplitude, mean, sigma, p0, p1, p2, p3]
    initial = [500, 125, 2, 10000, -100, 1, -0.01]
    
    # Parameter bounds
    lower = [0, 120, 0.5, -1e6, -1e4, -100, -1]
    upper = [5000, 130, 5, 1e6, 1e4, 100, 1]
    
    try:
        params, covariance = curve_fit(
            signal_plus_background,
            bin_centers,
            counts,
            sigma=errors,
            p0=initial,
            bounds=(lower, upper),
            maxfev=20000
        )
        
        # Get errors from covariance matrix
        errors = np.sqrt(np.diag(covariance))
        return params, errors
        
    except Exception as e:
        print("Full fit error: " + str(e))
        return None, None


# plot

def create_plots(mass, save_path=None):
    """Create 3-panel analysis figure"""
    
    # make histogram
    n_bins = 60
    counts, bin_edges = np.histogram(mass, bins=n_bins, range=(100, 160))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    errors = np.sqrt(counts)
    
    # Do fits
    print("")
    print("Fitting background (sideband method)...")
    bg_params = fit_background(bin_centers, counts, errors)
    
    print("Fitting full model (signal + background)...")
    full_params, full_errors = fit_full_model(bin_centers, counts, errors)
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ATLAS Open Data: Higgs to Diphoton Analysis", fontsize=14, fontweight='bold')
    
    #Panel 1: Data + BG Fit
    ax1 = axes[0]
    
    ax1.errorbar(bin_centers, counts, yerr=errors, fmt='o', markersize=3,
                color='black', label='ATLAS Data', capsize=2)
    
    if bg_params is not None:
        x_smooth = np.linspace(100, 160, 200)
        bg_curve = polynomial(x_smooth, *bg_params)
        ax1.plot(x_smooth, bg_curve, 'r-', linewidth=2, label='Background Fit')
    
    ax1.axvspan(120, 130, alpha=0.2, color='green', label='Signal Region')
    ax1.set_xlabel("Mass [GeV]")
    ax1.set_ylabel("Events / " + str(round(bin_width, 1)) + " GeV")
    ax1.set_title("(a) Data and Background Model")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(100, 160)
    
    # Panel 2: Signal extraction
    ax2 = axes[1]
    
    if bg_params is not None:
        background = polynomial(bin_centers, *bg_params)
        signal = counts - background
        
        ax2.errorbar(bin_centers, signal, yerr=errors, fmt='o', markersize=3,
                    color='blue', label='Data - Background', capsize=2)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=125, color='red', linestyle='--', linewidth=2, label='Higgs (125 GeV)')
    
    ax2.set_xlabel("Mass [GeV]")
    ax2.set_ylabel("Excess Events")
    ax2.set_title("(b) Background Subtracted")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(100, 160)
    
    # Panel 3: full fit
    ax3 = axes[2]
    
    ax3.errorbar(bin_centers, counts, yerr=errors, fmt='o', markersize=3,
                color='black', label='ATLAS Data', capsize=2)
    
    if full_params is not None:
        x_smooth = np.linspace(100, 160, 200)
        
        # Full model
        full_curve = signal_plus_background(x_smooth, *full_params)
        ax3.plot(x_smooth, full_curve, 'b-', linewidth=2, label='Signal + Background')
        
        # Bg only
        bg_only = polynomial(x_smooth, *full_params[3:])
        ax3.plot(x_smooth, bg_only, 'r--', linewidth=2, label='Background Only')
        
        # Signal only (filled)
        sig_only = gaussian(x_smooth, *full_params[:3])
        ax3.fill_between(x_smooth, bg_only, bg_only + sig_only, 
                        alpha=0.3, color='green', label='Higgs Signal')
    
    ax3.set_xlabel("Mass [GeV]")
    ax3.set_ylabel("Events / " + str(round(bin_width, 1)) + " GeV")
    ax3.set_title("(c) Signal + Background Fit")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_xlim(100, 160)
    
    plt.tight_layout()
    
    # save
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path, dpi=150, facecolor='white')
        print("")
        print("Plot saved: " + save_path)
    
    plt.show()
    
    return full_params, full_errors


def main():
    """Main function"""
    
    print("=" * 60)
    print("ATLAS Higgs -> gamma gamma ADVANCED ANALYSIS")
    print("Background Fitting + Signal Extraction")
    print("=" * 60)
    print("")
    
    # Loading all data
    print("LOADING ALL DATA PERIODS...")
    print("")
    mass = load_all_periods()
    
    # Filter to analysis range
    mass_filtered = mass[(mass > 100) & (mass < 160)]
    print("Events in 100-160 GeV range: " + str(len(mass_filtered)))
    
    # Creating plots and getting fit results
    script_folder = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_folder, "..", "results", "higgs_advanced_analysis.png")
    
    params, errors = create_plots(mass_filtered, save_path)
    
    # Print final results
    print("")
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    if params is not None:
        print("")
        print("Results:")
        print("  Measured Higgs Mass: " + str(round(params[1], 2)) + " +/- " + str(round(errors[1], 2)) + " GeV")
        print("  PDG Value:           125.25 +/- 0.17 GeV")
        print("  Difference:          " + str(round(abs(params[1] - 125.25), 2)) + " GeV")
        print("")
        print("  Signal Width:        " + str(round(params[2], 2)) + " +/- " + str(round(errors[2], 2)) + " GeV")
        print("")
        print("Conclusion: Higgs signal observed at ~125 GeV!")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
