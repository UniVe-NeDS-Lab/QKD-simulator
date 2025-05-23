#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:12:18 2023

@author: leonardo
"""

import math
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

@dataclass
class FSOQKD():
    tx_power: float = 0.0300
    visibility: float = 6
    eta_q: float = 0.1   # quantum efficiency
    wave_lambda: float = 1550e-9    # wavelength
    ts_diameter: float = 0.05       # telescope diameter in meters (source)
    td_diameter: float = 0.18       # telescope diameter n meters (destination)
    # optical efficiencies at S and D TODO: why are they different?
    eta_s: float = 0.9
    eta_d: float = 0.9
    mu_v: float = 4e-6
    sigma_v: float = math.sqrt(1.44e-14)
    mu_h: float = 4e-6
    sigma_h: float = math.sqrt(1.44e-14)

     
    def get_rate(self, d, tx_rate):
        """ distance in m """   
        self.responsivity = self.eta_q*(self.wave_lambda*1e6)/1.23985 # FIXME make this number a parameter
        self.s_gain = (math.pi*self.ts_diameter/self.wave_lambda)**2
        self.d_gain = (math.pi*self.td_diameter/self.wave_lambda)**2
        #pointing error angles for elevation and azimuth (gaussian)
        self.teta_v = np.random.normal(self.mu_v, self.sigma_v) 
        self.teta_h = np.random.normal(self.mu_h, self.sigma_h)
        self.teta = math.sqrt(self.teta_v**2+self.teta_h**2) # TODO muovi nel corpo della funzione
        self.l_teta = math.exp(-self.d_gain*self.teta**2)
        self.lambda_nm=self.wave_lambda*10**9
        self.elementary_q: float = 1.602e-19

        if (self.visibility < 6):
            self.qstar = 0.585*self.visibility**(1/3);
        if (self.visibility >= 6 and self.visibility<50 ):
            self.qstar=1.3
        else:
            self.qstar=1.5
        # scattering coefficient
        self.alpha = (3.91/self.visibility)*(self.lambda_nm/550)**(-self.qstar)
        self.beta=0 #absorption coefficient (negligible)
  
        distkm = d/1000;
        LA = math.exp(-distkm*(self.alpha+self.beta))
        #Eq. 2 (instantaneous)
        K1 = self.eta_q * self.tx_power * self.s_gain * self.eta_s * self.eta_d\
            * (LA/d**2) * (self.wave_lambda/(4*math.pi))**2
        P_rec = K1 * self.d_gain * self.l_teta
        P_rec2 = 0; # Not used (I left it here just if we need some additional statistic)
        P_rec = P_rec*(10**(1.5/10)+10**(3/10) )
        #Confirmed for filtering (3dB) and housing (1.5dB) from refs. 10 and 11 respectively
        totAtt = P_rec/self.tx_power; #It's valid... between 0 and 1 (0---->1)
        # The quantity above can be called transmittance, instead of 
        # attenuation... but we leave the previous word
        # Remind that total_attenuation considers: G_S*G_D*L_teta*eta_q*eta_S*eta_D*(LA/D1^2)*(lambda/(4*pi))^2 from above
        if (totAtt>1):
            #To be activated to know at which distance there is the amplification effect
            print('Warning: detected amplification phenomenon at distance: ',
                  d,' (m), please check link distance and/or telescope'
                  'diameters for the gains');
            #In order to avoid errors due to wrong transmission power setting
            totAtt=0.999;#like no atmospheric attenuation
        #I0dB=-20
        #Ranges from -30dB to -10dB
        I0 = P_rec/(math.pi*(self.td_diameter/2)**2); #Irradiance (Watts over receiver surface)
        R = self.responsivity
        
        Iback = 0.2;#set from 07
        N2back = (2*self.elementary_q*Iback*tx_rate)/(R*I0**2);
        #Thermal noise
        k1 = 1.380649e-23;#Boltzmann constant
        T = 300;#Temperature in °K
        RL = 50;#Ohm, Load resistance on the receiver unit
        N2term = (4*k1*T*tx_rate)/(R*RL*I0**2);
        #Quantum noise
        N2q = (2*self.elementary_q*tx_rate)/(R*I0);
        #Total noise
        N2 = N2back+N2term+N2q;
        SNR0 = P_rec/N2;
        SNR = 10 * math.log10(SNR0);
        BER = 0.5 * math.erfc((1/(2*math.sqrt(2))) * math.sqrt(SNR0));
    
        #BER evaluation (OOK-RZ)
        #BER=0.5*math.erfc( 0.5*math.sqrt(SNR0));
    
        #BER evaluation (PPM)
        #BER=0.5*math.erfc( (log(4)/(2*math.sqrt(2)))*math.sqrt(SNR0));
    
        #BER evaluation PAM-N
        #N=4
        #BER=0.5*math.erfc(math.sqrt(log(2*N)*SNR0/(2*math.sqrt(2)*(N-1))));
    
        alpha_dB = -10 * math.log10(totAtt) # in dB (only atmospheric)
        keyrate0 = tx_rate
        # Assumption: 1 photon for each pulse, so the starting key-rate 
        # is equal to tx_rate
        r = 1 # Assumed ot be valid for BB84 (it is the security proof, ref.05)
        pdark = 5e-10;#ref. 04 (to be redefined better)
        lambdaQKD = 10/(alpha_dB * r * math.log(10))
        #print("alpha:",alpha_dB);
        trasmittance = 10**(-alpha_dB/10)
        #alpha contains already the reference to distance (it is totAtt again, not in dB)
        # bobsPerf = trasmittance
        number_of_detectors = 1 #only one detector
        #Probability of correctly decoding a photon and its polarization
        Psifting = trasmittance/(2*number_of_detectors) 
        #BobsError=1-Psifting#Error probability at the receiver side (introduced by BB84)
        pBobDetect = 1*trasmittance*Psifting
        #*eta_D;#eq.18 terms eta_D already considered before
        keyrate0 = tx_rate*pBobDetect # eq.18 dense
        SK = 0
        #Probability of sensing something (distance is already taken into account in lambdaQKD)
        ps=math.exp((-1/lambdaQKD));
        #when equal to dark count transmission performance decrease iper-exponentially
        if ((ps-pdark)>0):
            SK=keyrate0;
        return P_rec, SNR, BER, SK, alpha_dB,\
               P_rec2, lambdaQKD, ps, totAtt



if __name__ == '__main__':
    x = []
    y_old = []
    y_new = []
    df = pd.DataFrame()
    tx_rate = 6_000_000_000
    new_f = FSOQKD()
    for distance in np.arange(10, 1000, 10):
        x.append(distance)
        new_func_res = new_f.get_rate(distance, tx_rate)
        y_new.append(new_func_res[3]/1000_000)
    plt.plot(x, y_new)
    plt.ylabel('bit-rate [mb/s]')
    plt.xlabel('distance [m]')
    plt.yscale('log')
    plt.show()
    
