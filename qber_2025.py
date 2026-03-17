import numpy as np
import matplotlib.pyplot as plt

""" Compute the expected SKR on a terrestrial FSO link based on the model from
Shukla and Kundu, "Performance of DVQKD Protocol Over Gamma Gamma Turbulence 
Channel" COMSNETS 2025"""

def H2(x):
    """Binary entropy function."""
    x = np.clip(x, 1e-15, 1 - 1e-15) # Prevent log(0) errors
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def calculate_theoretical_skr(d_b, Cn2=1e-14):
    """
    Calculates the SKR in Mbps using the exact analytical lower bound 
    derived in Eq. 18 of the paper.
     - d_b is the distance in m
     - Cn2 is the air turbolence, 1e-14 is average turbolence
    """
    # System parameters explicitly from the paper's Section IV
    wavelength = 1550e-9
    kappa = 0.43e-3     # Attenuation in dB/m
    Y0 = 1e-7           # Dark count rate
    eta = 0.2           # Detector efficiency
    mu = 0.5            # Signal intensity (photons/pulse)
    f = 1.22            # Error correction factor
    T1 = 1e-9           # Pulse duration (1 ns)
    
    rep_rate = 1 / T1   # 1 GHz repetition rate
    e_d = 0.00          # Baseline optical error 
                        # This is missing from the paper, so 
                        # Assumed 0 based on QBER graphs
                        # could be in the order 0.01-0.03
    theta = 0.1e-3      # Beam divergence angle (radians) - 0.1 mrad
    d_t = 0.05          # Transmitter lens diameter (m) - 5 cm
    a_r = 0.20          # Receiver aperture diameter (m) - 20 cm
    # Wavenumber
    k = 2 * np.pi / wavelength
    
    # Rytov variance (Eq. 6 plane wave assumption)
    sigma_I2 = 1.23 * Cn2 * (k**(7/6)) * (d_b**(11/6))
    
    # Large and small scale scattering parameters (Eq. 5)
    # Note: Handled extreme cases for d_b = 0
    if d_b == 0:
        alpha, beta = np.inf, np.inf
        turbulence_factor = 1.0
    else:
        alpha = (np.exp(0.49 * sigma_I2 / (1 + 1.11 * sigma_I2**(12/5))**(7/6)) - 1)**(-1)
        beta =  (np.exp(0.51 * sigma_I2 / (1 + 0.69 * sigma_I2**(12/5))**(5/6)) - 1)**(-1)
        turbulence_factor = (alpha * beta) / ((alpha - 1) * (beta - 1))
    
    # --- NTANOS GEOMETRIC LOSS CALCULATION ---
    # Calculate the expanded spot diameter of the beam at distance d_b
    d_s = d_t + 2 * d_b * np.tan(theta / 2)
    
    # Calculate fraction of beam captured by the receiver aperture
    if a_r >= d_s:
        tau_geo = 1.0  # Receiver captures the whole beam
    else:
        tau_geo = (a_r / d_s)**2 # Area ratio
    
    # Transmittance
    tau_alpha = 10**(-0.1 * kappa * d_b)
    tau_total = tau_alpha * tau_geo          # Combined channel transmittance
    tau_r = eta * tau_total                  # Total expected yield    
    # Average QBERs (Eq. 14 and 16)
    e1_avg = e_d + (Y0 * turbulence_factor) / (2 * eta * tau_alpha)
    emu_avg = e_d + (Y0 * turbulence_factor) / (2 * eta * mu * tau_alpha)
    
    # Average Gains (Derived for Eq. 18)
    Q1_avg = tau_r * mu * np.exp(-mu)
    Qmu_avg = tau_r * mu
    
    # Analytical lower bound for Secret Key Rate per pulse (Eq. 17 & 18)
    rd = 0.5 * (Q1_avg * (1 - H2(e1_avg)) - f * Qmu_avg * H2(emu_avg))
    
    # Convert to Mbps
    skr_mbps = (rd * rep_rate) / 1e6
    return max(0, skr_mbps)

def replicate_paper_figure_3():
    """Plots the theoretical SKR for weak turbulence, to replicate the results
    in the original paper."""
    # The paper plots up to 1500m for Figures 3, 4, and 5
    distances = np.linspace(0.1, 30000, 100) 
    cn2_value = 1e-14 # Moderate turbulence (this ranges from 1e-13 to 1e-15,
                      # but authors show its impact is very low on the SKR
    
    skr_results = [calculate_theoretical_skr(d, cn2_value) for d in distances]
    
    plt.figure(figsize=(8, 6))
    plt.plot(distances, skr_results, color='blue', linestyle='-', linewidth=2, 
             label='LoS Theo.')

    # Using linear scale as shown in the paper's axes for SKR
    #plt.ylim(10, 30) # Adjust based on exact plot boundaries
    #plt.xlim(0, 30000)
    
    plt.xlabel('$d_b$ (in m)', fontsize=12)
    plt.ylabel('SKR (Mbps)', fontsize=12)
    plt.title('Secrecy key rate for DVQKD $C_n^2 = 1e-15$', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    replicate_paper_figure_3()