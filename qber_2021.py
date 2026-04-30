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

weather_profiles = {
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
        'rain': 0.2,  # Almost no precipitation
        'Cn2': 1e-16  # Weak turbulence
    },
    'L4_MODERATE': {
        'V': 8.0,     # Light haze
        'rain': 1.2,  # Light precipitation
        'Cn2': 5e-16  # Weak-to-moderate turbulence
    },
    'L5_POOR': {
        'V': 5.0,     # Haze
        'rain': 2.0,  # Light rain
        'Cn2': 1e-15  # Moderate turbulence
    },
    'L6_BAD': {
        'V': 3.0,     # Heavy haze
        'rain': 5.0,  # Steady rain
        'Cn2': 5e-15  # Moderate-to-strong turbulence
    },
    'L7_SEVERE': {
        'V': 1,       # Fog
        'rain': 15.0, # Heavy rain
        'Cn2': 1e-14  # Strong turbulence
    },
    'L8_EXTREME': {
        'V': 0.4,     # Thick fog
        'rain': 30.0, # Violent rain showers
        'Cn2': 1e-13  # Heavy turbulence
    }
}

cluster_colors = [
        '#006400', '#228B22', '#90EE90',  # Sfumature verde
        '#FFD700', '#FFA500', '#FF8C00',   # Sfumature arancio/giallo
        '#FF6B6B', '#DC143C', '#8B0000'   # Sfumature rosso
    ]



class NtanosFSO_QKD:
    def __init__(self):
        # Explicitly stated # table 1
        self.wavelength = 1550e-9        
        self.d_t = 0.05                  
        self.a_r = 0.18                  
        self.theta = 182e-6              
        self.alpha_clear = 0.192         
        
        self.mu = 0.439     # mean photon number this is adjusted in order
                            # not to have too many photons, and it is increased
                            # to 0.65 in Stathis26 (defined in text)
                            # it should be optimized as in Ma III.a based
                            # on function f() [that is NOT the generation 
                            # frequency], implementing (12)
                            # From Ma et al we understand it should be as close
                            # to 1 as possible, as only single fotons transmit 
                            # valid information
        self.nu = 0.11      # mean photon number for decoy states
        self.f_ec = 1.22    # error correction, table 1 from Ntanos
        self.e_det = 0.01   # detector error Table 1 from Ma et al. 
                            # (1-Visibility)/2 in Ntanos26, but visibility is not
                            # given. 
        self.rho_ap = 0.008 # afterpulse  probability factor (Ntanos)
        self.e0 = 0.5       # error rate background. Same in both papers
        
        # hardware losses
        # Ntanos table 1
        self.L_sys = 17.65 # 2.65 + 3.0 + 1.5 + 0.5 + 10.0 (10% efficiency) 
        self.q = 0.5       # 0.5 in Ntanos21, 0.9 in Stathis26 actually, there are
                           # "efficient BB84" versions that reduce randomicity of the 
                           # base choice, and thus can receive more than 50% of the qbits
        
        # Y0 is the rate of dark counts, photons that were detected during the
        # vacuum state, i.e. no signal at all. Ntanos21 mention that this 
        # should be 50k per sec. As all the other quantities are per pulse, 
        # we need to report it to pulses. This is the only quantity that 
        # depends on the assumed rate and should be changed if rate
        # is changed. 
        # note that authors mention that this quantity includes the 
        # solar radiance from eq. 11
        # Spathis et al use 6000, but they operate at night
        self.t_gate = 1e-9
        self.Y0 = 50000 * self.t_gate 

    def H2(self, x):
        x = np.clip(x, 1e-15, 1 - 1e-15)
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

    def get_A_geo(self, L):
        # eq. 2
        d_s = self.d_t + 2 * L * np.tan(self.theta / 2)
        return np.where(self.a_r > d_s, 0.0, 20 * np.log10(d_s / self.a_r))

    def get_A_clear(self, L):
        # eq. 3 in Ntanos21, is not really clear. It reports a A_clear in 
        # dB, but it uses an 'a' coefficient that refers dB/km. In Stathis26 it 
        # is more clearly said that the atmospheric absorption is alpha*km
        return self.alpha_clear * L/1000
                       
    def calculate_skr(self, L, A_add):
        # eq. 3 can be written as
        total_loss_dB = self.get_A_geo(L) + self.get_A_clear(L) + self.L_sys + A_add
        eta = 10 ** (-total_loss_dB / 10)
        
        # So the Ntanos21 paper
        # uses a theoretical model from Ma2025: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.72.012326
        # the paper starts from the transmittance obtained with the previous
        # path loss calculation, and implements a lower bound for SRK when
        # a three-state decoy protocol is applied. 
        
        # eq 10
        Q_mu_raw = self.Y0 + 1 - np.exp(-eta * self.mu)
        Q_mu = Q_mu_raw * (1 + self.rho_ap) # in Nntanos21 the After-pulse 
                                            # probability factor rho_ap is mentioned
                                            # it is not presente in Stathis, 
                                            # simply because Q_mu is measured and
                                            # not estimated. However it does not 
                                            # say how it changes the equations.
                                            # What I'v found is eq. 7 in Papapanos20
                                            # https://arxiv.org/pdf/2010.03358
           
        # this formulation is cosmetically different from eq 8 in Papapanos
        # but it is numerically the same (here Y0 = Pdc in the paper, but
        # rescaling all to Y0 as in the paper, it is OK).
        E_mu = (self.e0 * self.Y0 + self.e_det * (1 - np.exp(-eta * self.mu)) + self.e0 * self.rho_ap * Q_mu_raw) / Q_mu
        
        
        # this is needed to compute the BER at the decoy state, E_nu. It is 
        # functionally equivalent to E_mu but it uses another intensity (nu != mu)
        # as by definition of decoy state
        Q_nu_raw = self.Y0 + 1 - np.exp(-eta * self.nu)
        Q_nu = Q_nu_raw * (1 + self.rho_ap)
        E_nu = (self.e0 * self.Y0 + self.e_det * (1 - np.exp(-eta * self.nu)) + self.e0 * self.rho_ap * Q_nu_raw) / Q_nu
        
        ## the next four lines provide the bount on Y1 as in eq 34 Ma et al
        term1 = Q_nu * np.exp(self.nu)
        term2 = Q_mu * np.exp(self.mu) * (self.nu**2 / self.mu**2)
        term3 = (self.mu**2 - self.nu**2) / self.mu**2 * self.Y0
        Y1 = (self.mu / (self.mu * self.nu - self.nu**2)) * (term1 - term2 - term3)
        #Y1 = np.maximum(Y1, 1e-15)
        ##
        
        # eq. 8 Ma et al, for i=1. Becomes eq. 35 by pluggin eq. 34 in.
        Q1 = Y1 * self.mu * np.exp(-self.mu)
        
        # eq 37 in Ma et al.
        e1 = (E_nu * Q_nu * np.exp(self.nu) - self.e0 * self.Y0) / (self.nu * Y1)
        e1 = np.clip(e1, 1e-15, 0.5) 
        
        # Ntano21 eq 1 or Stathis26 eq. 21 
        R = self.q * (Q1 * (1 - self.H2(e1)) - Q_mu * self.f_ec * self.H2(E_mu))
        return np.maximum(R, 1e-15)

# =====================================================================
# ORIGINAL SCALAR WEATHER FUNCTIONS
# =====================================================================

def get_scattering_loss(visibility_km, lambda_nm=1550):
    # the Kim/Kruse model (TBC with reference)
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
    # eq. 10 Ntanos2021
    if rain_rate_mm_hr <= 0:
        return 0.0
    return 1.1394 * (rain_rate_mm_hr ** 0.7057)


def get_turbulence_margin(L_m, Cn2, p_outage=0.01, wavelength=1550e-9, a_r=0.18):
    if Cn2 <= 1e-17 or L_m < 10:
        return 0.0
        
    # scintillation Ntanos 2026 eq 10
    k_wave = 2 * np.pi / wavelength
    # Rytov Variance defined in text after e1. 6
    sigma_r2 = 1.23 * Cn2 * (k_wave**(7/6)) * (L_m**(11/6))
    
    # defined in text below eq. 6
    d = np.sqrt((k_wave * a_r**2) / (4 * L_m))
    
    # eq. 5 and 6 
    term_a = 0.49 * sigma_r2 / (1 + 0.18 * d**2 + 0.56 * sigma_r2**(12/5))**(7/6)
    a = (np.exp(term_a) - 1)**-1
    term_b = 0.51 * sigma_r2 * (1 + 0.69 * sigma_r2**(12/5))**(-5/6) / (1 + 0.9 * d**2 + 0.62 * sigma_r2**(12/5))**(5/6)
    b = (np.exp(term_b) - 1)**-1
    
    # FIX: Intercept weak turbulence/high aperture averaging before gamma() overflows.
    # If a or b > 150, the variance is practically zero. The margin is 0 dB.
    if a > 150 or b > 150:
        return 0.0
    
    # def eq. 4
    def pdf(I):
        if I <= 0: return 0.0
        val = 2 * (a * b)**((a + b) / 2) / (gamma(a) * gamma(b)) * I**((a + b) / 2 - 1) * kv(a - b, 2 * np.sqrt(a * b * I))
        return np.nan_to_num(val, nan=0.0, posinf=0.0)

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # implements the integral of eq 7 and then the inversion of 
            # eq. 8, finally convert to dB
            sol = root_scalar(lambda I_th: quad(pdf, -np.inf, I_th)[0] - p_outage, 
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
    from model import default_values
    print(V, rain, Cn2)
    # This is the minimum SKR needed for a single path, swe don't consider
    # multipath so a link that does not provide at least the load needed
    # for one flow between the endpoints is useless.  
    min_skr = 2*default_values['key_size']*default_values['traffic_demand']*1e6/\
        (default_values['rekey_interval']*8e9*default_values['gen_rate'])
    print(min_skr)
    L = 10
    while True:    
        skr = get_skr(L, V=V, rain=rain, Cn2=Cn2)
        if skr < min_skr: 
            return L-10
        L += 10

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
    
    plt.figure(figsize=(9, 6))
    i = 0 
    for profile in weather_profiles: #['L0_OPTIMAL','L2_VERY_GOOD', 'L4_FAIR', 'L6_POOR', 'L9_EXTREME']:
        skr = []
        for L in L_sweep:
            skr.append(get_skr(L, **weather_profiles[profile]))
        plt.semilogy(L_sweep, skr, linewidth=2, label=profile[3:], color=cluster_colors[i])
        i+=1
       
    plt.title('Normalized SKR vs Link Distance Under Different Weather Conditions')
    plt.xlabel('Link Distance $L$ (meters)')
    plt.ylabel('Normalized SKR (bits/pulse)')
    
    plt.ylim(1e-8, 1e-2)
    plt.xlim(100, 5000)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_max_dist():
    max_dists = []
    labels = [] 
    for profile in weather_profiles:
        d = get_max_distance(**weather_profiles[profile])
        max_dists.append(d)
        labels.append(profile[3:])
        
    n_bars = len(max_dists)
    group_size = 3
    x_positions = []
    current_x = 0
    
    for i in range(n_bars):
        if i > 0 and i % group_size == 0:
            current_x += 1.5  # Aumentato leggermente il gap tra i cluster
        x_positions.append(current_x)
        current_x += 1.0

    # --- Palette di colori (Sfumature per ogni cluster) ---
    # GOOD: Forest Green -> Lime Green -> Pale Green
    # AVG:  Dark Orange -> Orange -> Yellow
    # BAD:  Dark Red -> Crimson -> Pinkish Red

    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(x_positions, max_dists, color=cluster_colors, edgecolor='black', alpha=0.9, width=0.8)
    
    # Formattazione asse X
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    
    # Etichette sopra le barre
    ax.bar_label(bars, padding=5, fmt='%.1f', fontweight='bold')
    

    # Pulizia estetica
    ax.set_ylabel("Distance [m]", fontsize=12)
    ax.set_title("Max Transmission Distance per Weather Profile", fontsize=15, pad=20)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Rimuove i bordi superflui per un look moderno
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_max_dist_old():
    max_dists = []
    labels = [] 
    for profile in weather_profiles:
        d = get_max_distance(**weather_profiles[profile])
        max_dists.append(d)
        labels.append(profile[3:])
        
    fig, ax = plt.subplots()
    for x in max_dists:
        print(f'"{x}" {x}, ', end='')
    bars = ax.bar(labels, max_dists, edgecolor='black', alpha=0.8)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.bar_label(bars, padding=3, fmt='%.1f')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def get_weather_mapping():
    return { get_max_distance(**weather_profiles[profile]): profile for profile in weather_profiles}


if __name__ == "__main__":
    #print("Generating Figure 2 recreation...")
    #plot_figure_2()

    print("Generating SKR vs Distance plot...")
    plot_skr_vs_distance()
    
    plot_max_dist()