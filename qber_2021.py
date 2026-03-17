import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters (As per Table 1 and Section IV) ---
# Naming conventions follow the paper's symbols exactly
eta_det = 0.10            # Quantum efficiency of detectors
V = 0.98                  # Visibility
e_det = (1 - V) / 2       # Intrinsic detector error (0.01)
f_e = 1.2                 # Error correction efficiency f(E_mu)
q = 1/3                   # Protocol efficiency
mu = 0.439                # Signal state mean photon number
nu = 0.11                 # Decoy state mean photon number

# Background Noise
# Total solar radiance counts + dark counts = 51000 cps
# y0 is probability per gate (1ns)
y0 = 51000 * 1e-9         

# Hardware/Fixed Losses (dB)
L_Bob = 2.65
L_filter = 3.0
L_housing = 1.5
L_misalign = 0.5
L_fixed_total = L_Bob + L_filter + L_housing + L_misalign # 7.65 dB

# Geometrical Parameters (Section IV-B)
D_t = 0.05                # Transmitter aperture diameter (m)
a_r = 0.18                # Receiver aperture diameter (m)
theta = 0.182e-3          # Full beam divergence angle (rad)

def h2(x):
    """Binary Shannon entropy H(x)."""
    if x <= 0 or x >= 1: return 0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def calculate_skr(L_atm_add, dist=0.3):
    """
    Calculates SKR/frep using the exact conventions of the paper.
    
    Args:
        L_atm_add: Additional atmospheric attenuation (dB) 
        dist: Distance in km (default 0.3 km for Fig 2)
    """
    # --- Equation (2): Geometrical Loss ---
    # This takes into account the fact that the beam enlarges
    # and some photons never reach the receiver
    d_s = D_t + 2 * (dist * 1000) * np.tan(theta / 2)
        
    if a_r >= d_s: # if the beam is still lower than the receiver
        a_geo_db = 0
    else:
        # Paper uses log10(a_r/d_s). 
        # Since a_r < d_s, this log is negative.
        a_geo_db = 20 * np.log10(a_r / d_s)
    
    # --- Equation (3): 
    # loss due to clean air. As the authors say, equation simplifies to a simple 
    # multiplication 
    a_clear_db = 0.192 * dist # Negative value representing gain/loss
    
    # --- Total Transmittance η (eta) ---
    # We combine all the loss components in dB 
    total_attenuation_db = a_geo_db -  a_clear_db - L_fixed_total - L_atm_add
    
    
    eta_ch = 10**(total_attenuation_db / 10)
    eta = eta_ch * eta_det
    
    # What follows comes from a citation in the paper, Ma et al. 
    # --- Yields and Gains (Decoy State BB84) ---
    Q_mu = y0 + 1 - np.exp(-mu * eta) # eq. 10 Ma et al
    Q_nu = y0 + 1 - np.exp(-nu * eta) 
    
    # Single-photon yield Y1 (Ma et al. bound)
    Y1 = (mu / (mu * nu - nu**2)) * (
        Q_nu * np.exp(nu) - Q_mu * np.exp(mu) * (nu / mu)**2 - y0 * (1 - (nu / mu)**2)
    )
    Q1 = Y1 * mu * np.exp(-mu)
    
    # --- Error Rates ---
    # E_mu (QBER) and e1 (Single-photon error rate)
    E_mu = (0.5 * y0 + e_det * (1 - np.exp(-mu * eta))) / Q_mu # (eq. 11 Ma et al)
    e1 = (0.5 * y0 + e_det * Y1 * eta) / Y1 if Y1 > 0 else 0.5 
    
    # --- Equation (1): Secure Key Rate ---
    # R >= q * { Q1 * [1 - H(e1)] - Q_mu * f(E_mu) * H(E_mu) }
    R = q * (Q1 * (1 - h2(e1)) - Q_mu * f_e * h2(E_mu))
    
    if R < 1e-10:
        return 0
    return R 

# --- Execution and Plotting ---

def generate_plots():
    # Fig 2 reproduction: 300m link
    L_atm_axis = np.linspace(0, 14, 100)
    rates = [calculate_skr(L) for L in L_atm_axis]
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(L_atm_axis, rates, color='blue', linewidth=2, label='Simulation')
    
    # Aesthetics to match the paper style
    plt.title("Secure Key Rate per Pulse vs Additional Attenuation")
    plt.xlabel("Additional Atmospheric Attenuation (dB)")
    plt.ylabel("SKR / $f_{rep}$")
    plt.xlim(0, 14)
    plt.ylim(1e-9, 1e-3)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    
    # Reference thresholds from Section IV-C
    plt.axvline(11.8, color='red', linestyle=':', label='Threshold (11.8 dB)')
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    generate_plots()