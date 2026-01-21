"""
Higgs -> gamma gamma Analysis - SIMULATION MODE
Works offline with simulated data (no internet needed)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


# fit functions

def gaussian(x, amplitude, mean, sigma):
    """Gaussian function for signal"""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2)


def polynomial(x, p0, p1, p2, p3):
    """Polynomial function for background"""
    return p0 + p1*x + p2*x**2 + p3*x**3


def signal_plus_background(x, amp, mean, sigma, p0, p1, p2, p3):
    """Combined model"""
    return gaussian(x, amp, mean, sigma) + polynomial(x, p0, p1, p2, p3)


# data simulation

def generate_data(n_background=50000, n_signal=500, seed=42):
    """
    Generate realistic diphoton mass data
    
    Background: Falling exponential (QCD processes)
    Signal: Gaussian peak at 125 GeV (Higgs boson)
    """
    
    np.random.seed(seed)
    
    # Background: falling exponential shifted to 100-160 range
    tau = 40  # decay constant
    background = np.random.exponential(scale=tau, size=n_background * 3)
    background = background + 100
    background = background[(background > 100) & (background < 160)]
    background = background[:n_background]
    
    # Signal: Gaussian at Higgs mass
    higgs_mass = 125.25  # GeV (PDG value)
    detector_resolution = 1.7  # GeV
    signal = np.random.normal(loc=higgs_mass, scale=detector_resolution, size=n_signal)
    
    # Combine
    all_data = np.concatenate([background, signal])
    np.random.shuffle(all_data)
    
    print("Simulated Data:")
    print("  Background events: " + str(len(background)))
    print("  Signal events:     " + str(n_signal))
    print("  Signal/Background: " + str(round(100*n_signal/len(background), 1)) + "%")
    print("  Total events:      " + str(len(all_data)))
    
    return all_data


# analysis and plotting

def analyze(mass_data, save_path=None):
    """Analyze data and create plots"""
    
    #histogram
    n_bins = 60
    counts, bin_edges = np.histogram(mass_data, bins=n_bins, range=(100, 160))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    errors = np.sqrt(counts)
    errors = np.where(errors == 0, 1, errors)
    
    # Fiting full model
    print("")
    print("Fitting signal + background model...")
    
    initial = [100, 125, 2, 5000, -50, 0.5, -0.001]
    lower = [0, 120, 0.5, -1e6, -1e4, -100, -1]
    upper = [1000, 130, 5, 1e6, 1e4, 100, 1]
    
    try:
        params, covariance = curve_fit(
            signal_plus_background,
            bin_centers,
            counts,
            sigma=errors,
            p0=initial,
            bounds=(lower, upper),
            maxfev=10000
        )
        param_errors = np.sqrt(np.diag(covariance))
        fit_ok = True
        
        print("Fit successful!")
        print("  Higgs Mass:  " + str(round(params[1], 2)) + " +/- " + str(round(param_errors[1], 2)) + " GeV")
        print("  Width:       " + str(round(params[2], 2)) + " +/- " + str(round(param_errors[2], 2)) + " GeV")
        
    except Exception as e:
        print("Fit error: " + str(e))
        fit_ok = False
        params = None
        param_errors = None
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Higgs to Diphoton Simulation Analysis", fontsize=14, fontweight='bold')
    
    # Panel 1: Data + Fit
    ax1 = axes[0]
    
    ax1.errorbar(bin_centers, counts, yerr=errors, fmt='o', markersize=3,
                color='black', label='Simulated Data', capsize=2)
    
    if fit_ok:
        x_smooth = np.linspace(100, 160, 200)
        full_fit = signal_plus_background(x_smooth, *params)
        ax1.plot(x_smooth, full_fit, 'b-', linewidth=2, label='Signal + Background')
        
        bg_only = polynomial(x_smooth, *params[3:])
        ax1.plot(x_smooth, bg_only, 'r--', linewidth=2, label='Background')
    
    ax1.axvline(x=125.25, color='green', linestyle=':', alpha=0.7, label='PDG: 125.25 GeV')
    ax1.set_xlabel("Mass [GeV]")
    ax1.set_ylabel("Events / " + str(round(bin_width, 1)) + " GeV")
    ax1.set_title("(a) Data and Fit")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(100, 160)
    
    # Panel 2: Signal Extract
    ax2 = axes[1]
    
    if fit_ok:
        bg_at_bins = polynomial(bin_centers, *params[3:])
        signal_extracted = counts - bg_at_bins
        
        ax2.errorbar(bin_centers, signal_extracted, yerr=errors, fmt='o', 
                    markersize=3, color='blue', label='Data - Background', capsize=2)
        
        x_smooth = np.linspace(100, 160, 200)
        signal_fit = gaussian(x_smooth, *params[:3])
        ax2.plot(x_smooth, signal_fit, 'r-', linewidth=2, 
                label='Gaussian: mean=' + str(round(params[1], 1)) + ' GeV')
        
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax2.axvline(x=125.25, color='green', linestyle=':', alpha=0.7)
    ax2.set_xlabel("Mass [GeV]")
    ax2.set_ylabel("Excess Events")
    ax2.set_title("(b) Background Subtracted")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(100, 160)
    
    #Panel 3: Results Summa
    ax3 = axes[2]
    ax3.axis('off')
    
    if fit_ok:
        # Calculate signal events
        signal_events = int(params[0] * params[2] * np.sqrt(2*np.pi) / bin_width)
        
        results = """
========================================
         ANALYSIS RESULTS
========================================

HIGGS PARAMETERS
----------------------------------------

  Measured Mass:  {:.2f} +/- {:.2f} GeV
  
  PDG Value:      125.25 +/- 0.17 GeV
  
  Difference:     {:.2f} GeV

  Signal Width:   {:.2f} +/- {:.2f} GeV

========================================

STATISTICS
----------------------------------------

  Total Events:   {:,}
  
  Signal Events:  ~{:,}

========================================

CONCLUSION: Higgs Signal Observed!

  A clear excess is visible at 125 GeV,
  consistent with the Higgs boson.
  
========================================
        """.format(
            params[1], param_errors[1],
            abs(params[1] - 125.25),
            params[2], param_errors[2],
            len(mass_data),
            signal_events
        )
    else:
        results = "Fit failed. Need more data."
    
    ax3.text(0.05, 0.95, results, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    if save_path is not None:
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path, dpi=150, facecolor='white')
        print("")
        print("Plot saved: " + save_path)
    
    plt.show()
    
    return params, param_errors


def main():
    """Main function"""
    
    print("=" * 50)
    print("Higgs -> gamma gamma SIMULATION ANALYSIS")
    print("=" * 50)
    print("")
    print("This mode works offline with simulated data.")
    print("For real ATLAS data: use basic_analysis.py")
    print("")
    
    # Generate simulated data
    mass_data = generate_data(
        n_background=50000,
        n_signal=500,
        seed=2012  # Discovery year :)
    )
    
    # Analyze
    script_folder = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_folder, "..", "results", "higgs_simulation.png")
    
    analyze(mass_data, save_path)
    
    print("")
    print("=" * 50)
    print("Simulation analysis complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
