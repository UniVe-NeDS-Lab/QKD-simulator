#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, kv
from scipy.integrate import quad
from scipy.optimize import root_scalar

# =====================================================================
# QKD SYSTEM DEFINITION (Strict Paper Parameters)
# =====================================================================

# V    # visibility (km) https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7124736
# rain (mm/hour)
# refraction parameter  (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6844864)

# weather conditions parameters

weather_profiles_small = {
    'GOOD': {
        'V': 20.0,    # clear visibility
        'rain': 0.0,  # no rain
        'Cn2': 1e-17  # zero turbulence
    },
    'AVG': {
        'V': 10.0,    # light haze
        'rain': 10.0, # light rain
        'Cn2': 1e-15  # some turbulence
    },
    'BAD': {
        'V': 4.0,     # heavy haze
        'rain': 20.0, # heavy rain
        'Cn2': 1e-14  # turbulence
    },
    'EXTREME': {      
        'V': 0.6,     # thick fog
        'rain': 50.0, # extreme rain
        'Cn2': 1e-13  # heavy turbulence
    }
}

weather_profile = {
    'L0_OPTIMAL': {
        'V': 50.0,    # Exceptionally clear sky
        'rain': 0.0,  # No precipitation
        'Cn2': 1e-17  # Practically zero turbulence
    },
    'L1_EXCELLENT': {
        'V': 30.0,    # Very clear sky
        'rain': 0.0,  # No precipitation
        'Cn2': 1e-17  # Practically zero turbulence
    },
    'L2_VERY_GOOD': {
        'V': 20.0,    # Very clear
        'rain': 0.0,  # No precipitation
        'Cn2': 5e-17  # Very weak turbulence
    },
    'L3_GOOD': {
        'V': 10.0,    # Clear
        'rain': 0.0,  # No precipitation
        'Cn2': 1e-16  # Weak turbulence
    },
    'L4_FAIR': {
        'V': 8.0,     # Light haze
        'rain': 2.0,  # Light drizzle
        'Cn2': 5e-16  # Weak-to-moderate turbulence
    },
    'L5_MODERATE': {
        'V': 5.0,     # Haze
        'rain': 5.0,  # Moderate rain
        'Cn2': 1e-15  # Moderate turbulence
    },
    'L6_POOR': {
        'V': 3.0,     # Heavy haze
        'rain': 10.0, # Steady rain
        'Cn2': 5e-15  # Moderate-to-strong turbulence
    },
    'L7_BAD': {
        'V': 1.5,     # Thin fog
        'rain': 20.0, # Heavy rain
        'Cn2': 1e-14  # Strong turbulence
    },
    'L8_SEVERE': {
        'V': 0.8,     # Light fog
        'rain': 30.0, # Violent rain showers
        'Cn2': 5e-14  # Very strong turbulence
    },
    'L9_EXTREME': {
        'V': 0.4,     # Moderate fog
        'rain': 50.0, # Severe storm
        'Cn2': 1e-13  # Heavy turbulence
    }
}

class NtanosFSO_QKD:
    def __init__(self):
        # Explicitly stated
        self.wavelength = 1550e-9        
        self.d_t = 0.05                  
        self.a_r = 0.18                  
        self.theta = 182e-6              
        self.alpha_clear = 0.192         
        
        self.mu = 0.439                  
        self.nu = 0.11                   
        self.f_ec = 1.2                  
        self.e_opt = 0.01                
        self.rho_ap = 0.008              
        self.e0 = 0.5                    
        
        # Guessed / Assumed baseline hardware losses
        self.L_sys = 17.65 
        self.q = 0.5                     
        self.t_gate = 1e-9
        self.Y0 = 51000 * self.t_gate 

    def H2(self, x):
        x = np.clip(x, 1e-15, 1 - 1e-15)
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    def get_A_geo(self, L):
        d_s = self.d_t + 2 * L * np.tan(self.theta / 2)
        return np.where(self.a_r > d_s, 0.0, 20 * np.log10(d_s / self.a_r))

    def calculate_skr(self, L, A_add):
        total_loss_dB = self.get_A_geo(L) + (self.alpha_clear * (L/1000.0)) + self.L_sys + A_add
        eta = 10 ** (-total_loss_dB / 10)
        
        Q_mu_raw = self.Y0 + 1 - np.exp(-eta * self.mu)
        Q_mu = Q_mu_raw * (1 + self.rho_ap)
        E_mu = (self.e0 * self.Y0 + self.e_opt * (1 - np.exp(-eta * self.mu)) + 0.5 * self.rho_ap * Q_mu_raw) / Q_mu
        
        Q_nu_raw = self.Y0 + 1 - np.exp(-eta * self.nu)
        Q_nu = Q_nu_raw * (1 + self.rho_ap)
        E_nu = (self.e0 * self.Y0 + self.e_opt * (1 - np.exp(-eta * self.nu)) + 0.5 * self.rho_ap * Q_nu_raw) / Q_nu
        
        term1 = Q_nu * np.exp(self.nu)
        term2 = Q_mu * np.exp(self.mu) * (self.nu**2 / self.mu**2)
        term3 = (self.mu**2 - self.nu**2) / self.mu**2 * self.Y0
        
        Y1_L = (self.mu / (self.mu * self.nu - self.nu**2)) * (term1 - term2 - term3)
        Y1_L = np.maximum(Y1_L, 1e-15) 
        Q1_L = Y1_L * self.mu * np.exp(-self.mu)
        
        e1_U = (E_nu * Q_nu * np.exp(self.nu) - self.e0 * self.Y0) / (self.nu * Y1_L)
        e1_U = np.clip(e1_U, 1e-15, 0.5)
        
        R = self.q * (Q1_L * (1 - self.H2(e1_U)) - Q_mu * self.f_ec * self.H2(E_mu))
        return np.maximum(R, 1e-15)

# =====================================================================
# ORIGINAL SCALAR WEATHER FUNCTIONS
# =====================================================================

def get_scattering_loss(visibility_km, lambda_nm=1550):
    if visibility_km > 50:
        q = 1.6
    elif visibility_km > 6:
        q = 1.3
    elif visibility_km > 1:
        q = 0.16 * visibility_km + 0.34
    elif visibility_km > 0.5:
        q = visibility_km - 0.5
    else:
        q = 0.0
    return (17.0 / visibility_km) * (lambda_nm / 550.0)**(-q)

def get_rain_loss(rain_rate_mm_hr):
    if rain_rate_mm_hr <= 0:
        return 0.0
    return 1.1394 * (rain_rate_mm_hr ** 0.7057)


def get_turbulence_margin(L_m, Cn2, p_outage=0.01, wavelength=1550e-9, a_r=0.18):
    if Cn2 <= 1e-17 or L_m < 10:
        return 0.0
        
    k_wave = 2 * np.pi / wavelength
    sigma_r2 = 1.23 * Cn2 * (k_wave**(7/6)) * (L_m**(11/6))
    d = np.sqrt((k_wave * a_r**2) / (4 * L_m))
    
    term_a = 0.49 * sigma_r2 / (1 + 0.18 * d**2 + 0.56 * sigma_r2**(12/5))**(7/6)
    a = (np.exp(term_a) - 1)**-1
    term_b = 0.51 * sigma_r2 * (1 + 0.69 * sigma_r2**(12/5))**(-5/6) / (1 + 0.9 * d**2 + 0.62 * sigma_r2**(12/5))**(5/6)
    b = (np.exp(term_b) - 1)**-1
    
    # FIX: Intercept weak turbulence/high aperture averaging before gamma() overflows.
    # If a or b > 150, the variance is practically zero. The margin is 0 dB.
    if a > 150 or b > 150:
        return 0.0
    
    def pdf(I):
        if I <= 0: return 0.0
        val = 2 * (a * b)**((a + b) / 2) / (gamma(a) * gamma(b)) * I**((a + b) / 2 - 1) * kv(a - b, 2 * np.sqrt(a * b * I))
        return np.nan_to_num(val, nan=0.0, posinf=0.0)

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = root_scalar(lambda I_th: quad(pdf, 0, I_th, limit=200)[0] - p_outage, 
                              bracket=[1e-20, 10], method='brentq')
        return -10 * np.log10(sol.root)
    except:
        # FIX: Only returns 100 dB loss if the integration fails due to true, extreme fading.
        return 100.0

# =====================================================================
# MAIN WRAPPER FUNCTION
# =====================================================================

def get_skr(L, A_add=None, V=50.0, rain=0.0, Cn2=1e-17, p_outage=0.01):
    """ L is meters """
    model = NtanosFSO_QKD()
    
    if A_add is not None:
        return model.calculate_skr(L, A_add)
        
    else:
        L_km = L / 1000.0
        total_A_add = (get_scattering_loss(V) * L_km + 
                       get_rain_loss(rain) * L_km + 
                       get_turbulence_margin(L, Cn2, p_outage))
        return model.calculate_skr(L, total_A_add)


def get_max_distance(V, rain, Cn2):
    """ Return the maximum distance at which we have a reasonable SKR 
    given certain weather conditions """
    L = 100
    while True:    
        skr = get_skr(L, V=V, rain=rain, Cn2=Cn2)
        if skr < 10**(-9): # we assume GHz generation of photons, so 
                           # if SKR is < 1/s the link is unusable
            return L-100
        L += 100

# =====================================================================
# PLOTTING FUNCTIONS
# =====================================================================

def plot_figure_2():
    """Sweeps A_add from 0 to 15 dB at L = 300m to recreate Figure 2."""
    A_add_sweep = np.linspace(0, 15, 200)
    
    # Calculate SKR for the sweep using the override mode
    skr_results = get_skr(L=300.0, A_add=A_add_sweep)
    
    # Plot formatting
    plt.figure(figsize=(8, 6))
    plt.semilogy(A_add_sweep, skr_results, 'k-', linewidth=2, label="Calculated SKR")
    plt.title('Figure 2 Recreation (Strict Paper Parameters, L=300m)')
    plt.xlabel('Additional Atmospheric Attenuation $A_{add}$ (dB)')
    plt.ylabel('Normalized SKR (bits/pulse)')
    
    # Limit to paper's visual bounds
    plt.ylim(1e-8, 1e-2)
    plt.xlim(0, 12)
    
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.show()

def plot_skr_vs_distance():
    """Sweeps L from 100m to 3000m for three different weather conditions."""
    L_sweep = np.linspace(100, 5000, 50) 
    
    skr_best = []
    skr_mod = []
    skr_bad = []
    
    for L in L_sweep:
        skr_best.append(get_skr(L, **weather_profiles['GOOD']))
        skr_mod.append(get_skr(L, **weather_profiles['AVG']))
        skr_bad.append(get_skr(L, **weather_profiles['BAD']))
    
    # Plotting
    plt.figure(figsize=(9, 6))
    
    plt.semilogy(L_sweep, skr_best, 'g-', linewidth=2, label="Best Case (Clear, Cn2=1e-17)")
    plt.semilogy(L_sweep, skr_mod,  'b--', linewidth=2, label="Moderate (Haze, Cn2=1e-15)")
    plt.semilogy(L_sweep, skr_bad,  'r-.', linewidth=2, label="Worst Case (Fog+Rain, Cn2=1e-14)")
    
    plt.title('Normalized SKR vs Link Distance Under Different Weather Conditions')
    plt.xlabel('Link Distance $L$ (meters)')
    plt.ylabel('Normalized SKR (bits/pulse)')
    
    plt.ylim(1e-8, 1e-2)
    plt.xlim(100, 5000)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Generating Figure 2 recreation...")
    plot_figure_2()
    
    print("Generating SKR vs Distance plot...")
    plot_skr_vs_distance()