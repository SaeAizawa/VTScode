# Python routine to solve VTS model (Hedin+ 1983) written by Sae Aizawa (LPP)
# email : sae.aizawa@lpp.polytechnique.fr
# To Moa Persson
# On 5th March, 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Parameters:
    def __init__(self):
        self.PT = np.zeros(50)
        self.PD = np.zeros((50, 6))
        self.PP = np.zeros((50, 6))
        self.PTM = np.zeros(8)
        self.PDM = np.zeros((8, 6))
        self.SW = np.zeros(15)
        self.gsurf = 887.4
        self.re = 6052.7
        self.rgas = 831.4
        self.zlb = 150.

        # TEMPERATURE
        self.PT1 = np.array([1.14575E0, 5.73938E-1, -1.05623E-1, -1.53896E-1, -7.11596E-3,
                             -1.82894E-1, 4.41052E-3, 1.22197E-1, 3.20351E-4, -9.28368E-3,
                             -2.32161E-5, 0.0, 0.0, 1.00000E-3, 6.00000E-4,
                             7.93416E-1, 1.30399E-1, 8.82217E-2, -4.98274E-1, -2.05990E-2,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             1.12000E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             7.23933E-1, 9.74046E-1, 1.00000E0, 7.24953E-2, 0.0])

        # CO2 DENSITY
        self.PA1 = np.array([7.76049E-1, 2.93750E0, -1.47210E-1, -1.57306E-1, -6.69599E-2,
                             -8.18055E-1, -1.06697E-2, 3.00201E-1, 7.96075E-4, 3.24607E-1,
                             9.23450E-5, 0.0, 0.0, -2.37584E-3, -1.34238E-4,
                             1.00000E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             1.13277E0, -1.87582E0, -3.38029E-1, -9.31956E-1, 9.04382E-2,
                             1.67238E0, 7.32745E-3, 8.28310E-1, 1.69341E-3, -6.84008E-1,
                             -1.00458E-4, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 2.01296E-2, 0.0])

        # O DENSITY
        self.PB1 = np.array([1.07134E0, 7.13595E-1, -3.61877E-2, 2.82843E-1, 4.85210E-3,
                             -1.95412E-1, -1.76002E-3, -3.34167E-1, -9.68110E-4, 3.87223E-1,
                             3.88084E-5, 0.0, 0.0, 2.84044E-3, 1.20219E-3,
                             3.00713E-2, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             3.90127E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             1.12140E0, 1.30508E0, 0.0, 0.0, 9.80888E-4])

        # CO DENSITY & TLB
        self.PC1 = np.array([9.84586E-1, 1.92394E0, -8.16346E-2, 1.85261E-1, -4.62701E-3,
                             -4.08268E-1, -1.68582E-3, -2.05573E-1, -1.15921E-3, 4.93592E-1,
                             -2.59753E-5, 0.0, 0.0, 1.39529E-3, 5.53068E-4,
                             1.00000E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             1.00000E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0])

        # HE DENSITY
        self.PD1 = np.array([9.61871E-1, -1.42734E0, 5.93447E-1, 9.36320E-2, 1.39517E-1,
                             8.39837E-1, -3.60608E-3, -3.57368E-1, -1.38972E-3, -1.96184E-2,
                             8.86656E-5, 0.0, 0.0, 2.15513E-3, -7.68169E-4,
                             3.03416E-2, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             1.00000E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0])

        # N DENSITY
        self.PE1 = np.array([1.07218E0, 1.07061E0, -1.92920E-1, 1.72877E-1, -4.19517E-2,
                             -2.37737E-1, -3.55792E-4, -2.46647E-1, -8.06493E-4, 4.72518E-1,
                             8.04218E-6, 0.0, 0.0, 4.85444E-3, 1.24067E-3,
                             1.00000E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             1.00000E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0])

        # N2 DENSITY
        self.PF1 = np.array([8.11373E-1, 1.15335E0, 1.26869E-1, 2.25488E-1, 4.20755E-2,
                             -2.21562E-1, 6.93173E-3, -4.49676E-1, -2.56643E-4, 5.91909E-1,
                             1.22099E-5, 0.0, 0.0, -1.42414E-3, -6.52404E-4,
                             1.00000E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             1.00000E0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 1.88000E0, 0.0])

        # LOWER BOUNDARY
        self.PM = np.array([[1.99171E2, 1.87600E2, 1.77000E2, 9.21400E-2, 1.25000E2,
                             1.50000E2, 1.40000E2, 1.00000E-1, 5.00530E8, 0.0,
                             1.30000E2, 4.00000E1, 4.34577E1, 1.00000E2, 0.0,
                             1.63000E0, 1.04912E9, 1.00000E-1, 1.12862E2, 0.0,
                             0.0, 1.20000E2, 5.80000E0, 1.00000E0, 2.95223E8,
                             0.0, 1.26638E2, 0.0, 0.0, 0.0,
                             0.0, 0.0, 5.64985E6, 2.00000E-5, 1.23157E2, 1.00000E1,
                             0.0, 0.0, 0.0, 0.0, 9.30114E6,
                             0.0, 1.12862E2, 0.0, 0.0, 1.24000E2,
                             1.45000E0, 0.0, 1.39749E8, 3.50000E-2, 1.29748E2,
                             0.0, 0.0, 0.0, 0.0, 0.0]]).reshape((7, 8)).T

        self.SW = np.array([1., 1., 1., 1., 1.,
                            1., 1., 1., 1., 1.,
                            1., 1., 1., 1., 1.])


        self.PT = self.PT1
        self.PP[:, 0] = self.PA1
        self.PP[:, 1] = self.PB1
        self.PP[:, 2] =	self.PC1
        self.PP[:, 3] =	self.PD1
        self.PP[:, 4] =	self.PE1
        self.PP[:, 5] =	self.PF1
        self.PD = self.PP.T
        self.PTM = self.PM[:, 0]
        self.PDM = self.PM[:, 1:7]
        return

def globv(lat, tloc, f107a, f107, data, param):

    dgtr = 1.74533e-2
    xl = -1000
    tll = 1000
    hr = 0.2618
    t = np.zeros((6))
    plg = np.zeros((6, 6))
    
    if xl != lat or tll != tloc:
        # Calculate Legendre Polynomials
        s = np.cos(lat * dgtr)
        c = s * np.cos(hr * (tloc - 12.))
        s2 = s * s
        c2 = c * c
        c4 = c2 * c2
        plg[1, 0] = c
        plg[2, 0] = 0.5 * (3.* c2 - 1.)
        plg[3, 0] = 0.5 * (5.* c2 - 3.) * c
        plg[4, 0] = (35.* c4 - 30.* c2 + 3) / 8.
        plg[5, 0] = (63.* c4 - 70.* c2 + 15.) * c / 8.
        plg[1, 1] = s
        plg[2, 2] = 3.* s2
        plg[3, 3] = 15.* s2 * s
        plg[4, 4] = 105.* s2 * s2
        plg[5, 5] = 945.* s2 * s2 * s
        xl = lat

    if (tll != tloc):
        stloc = np.sin(hr * tloc)
        s2tloc = np.sin(2. * hr * tloc)
        s3tloc = np.sin(3. * hr * tloc)
        s4tloc = np.sin(4. * hr * tloc)
        s5tloc = np.sin(5. * hr * tloc)

    tll = tloc
    t[0] = data[13] * (f107a - 200.) + data[14] * (f107 - f107a)

    if (data[1] <= 0.) :
        f = 1.
    else:
        f = 1. + t[0]/(data[1] - data[3] + data[5] - data[7] + data[9])
    

    t[1] = (data[1] * plg[1,0] + data[2] * plg[1,1] * stloc) * f
    t[2] = (data[3] * plg[2,0] + data[4] * plg[2,2] * s2tloc) * f
    t[3] = (data[5] * plg[3,0] + data[6] * plg[3,3] * s3tloc) * f
    t[4] = (data[7] * plg[4,0] + data[8] * plg[4,4] * s4tloc) * f
    t[5] = (data[9] * plg[5,0] + data[10] * plg[5,5] * s5tloc) * f
    g = 0.

    for i in  range (0, 6):
        g = g + param.SW[i] * t[i]
    
    globv = g
        
    return globv

def denss(alt, dlb, tinf, xm, alpha, \
          zlb, s2, t0, za, ta, zm, Am, sp, params):


    # Calculate ZETA - equation (A7)
    ZETA = (alt - zlb) * (params.re + zlb) / (params.re + alt)

    # Check if TA is within bounds  
    TAF = (ta - t0) / (tinf - t0)
    if (TAF <= 0.):
        ta = t0 + 0.0001 * (tinf - t0)
    elif (TAF >= 1.) :
        ta = tinf - 0.0001 * (tinf - t0)


    zz = np.maximum(alt, za)
    ZG2 = (zz - zlb) * (params.re + zlb) / (params.re + zz)
    ZGA = (za - zlb) * (params.re + zlb) / (params.re + za)
    ZG  = ZETA

    # Calculate TLB
    tlb = tinf - (tinf - ta) * np.exp(s2 * ZGA)

    # Calculate T2
    T2 = tinf - (tinf - tlb) * np.exp(-s2 * ZG2)
    TT = T2

    # Check altitude
    if (alt < za):
        # Calculate temperature below ZA
        S1 = -s2 * (tinf - ta) / (ta - t0)
        ZG1 = ZG - ZGA
        if (S1 * ZG1 >= 10.):
            T1 = t0
        else:
            T1 = t0 - (t0 - ta) * np.exp(-S1 * ZG1)
        
        TT = T1

    tz = TT
    TZL = tlb

    if (Am != 0.):
        # Calculate density above ZA
        GLB = params.gsurf / (1. + zlb / params.re)**2
        GAMMA2 = xm * GLB / (s2 * params.rgas * tinf)
        DENS2 = dlb * (tlb / T2)**(GAMMA2) * np.exp(-s2 * GAMMA2 * ZG2)
        DENSS = DENS2

        # Check altitude again
        if (alt < za):
            # Calculate density below ZA
            GAMMA1 = xm * GLB / (S1 * params.rgas * t0)
            DENS1 = DENS2 * (ta / TT)**(GAMMA1) * np.exp(-S1 * GAMMA1 * ZG1)
            DENSS = DENS1
        
        # Check for non-zero XM
        if (xm != 0.) :
            # Calculate density modification factor
            ZGM = (zm - zlb) * (params.re + zlb) / (params.re + zm)
            EXZ = np.exp(-(sp * (ZG - ZGM)))
            DIT = 4. * Am * EXZ / (1. + EXZ)**2 / tinf
            tz = TT / (1. + TT * DIT)
            EXM = np.exp(sp* ZGM)
            DITL = 4. * Am * EXM / (1. + EXM)**2 / tinf
            TZL = tlb / (1. + tlb * DITL)
            # Modify density
            GAMMAP = xm * GLB / (sp * params.rgas * tinf)
            DENSM = np.exp(GAMMAP * 4. * Am * (EXZ / (1. + EXZ) - EXM / (1. + EXM)))
            DENSS = DENSM * DENSS

    # final density calculation
    DENSS = DENSS * (TZL / tz)**(1. + alpha)
    
    return DENSS, tz, tlb

def dnet(dd, dm, zhm, xmm, xm, params):
    # Turbopause correction
    a = zhm/(xmm - xm)
    adm = 0.

    if(dm > 0.):
        adm = np.log(dm)
    add = 0.
    if(dd > 0.):
        add = np.log(dd)
    ylog = a * (add - adm)
    if(ylog < -10):
        dnet = dm
    elif(ylog > 10.):
        dnet = dd
    else:
        dnet = dm * (1.+ np.exp(ylog))**(1/a)

    return dnet

def ccor(alt, r, h1, zh):
    e = (zh - alt)/h1
    if e > 170.:
        ccor = r
    elif e < -170.:
        ccor = 1.
    else:
        ex = np.exp(e)
        ccor = (1. + r*ex)/(1. + ex)
        
    return ccor

def turbo(dd, dm, zb, xm, xmm, tz, params):
    inv = (1. + zb/params.re)**2
    gzb = params.gsurf/inv
    dif = xm - xmm
    zh = zb + params.rgas * tz/gzb/dif * np.log(dd/dm)
    
    return zh
    
def vts(alt, xlat, tloc, f107a, f107, mas, params):
    Density = np.zeros((7))
    Temperature = np.zeros((2))
    T_infv = globv(xlat, tloc, f107a, f107, params.PT, params)
    T_inf = params.PTM[0] * (1. + T_infv) * params.PT[0]
    Temperature[0] = T_inf
    Z_a = params.PTM[4] * params.PT[30]
    T_0 = params.PTM[2] * params.PT[46]
    Sig_1 = params.PTM[3] * params.PT[45]
    Sig_e = params.PT[48]


    # Set T_a between T_0 and T_inf
    fr = params.PT[15] * (1. + T_infv)
    fr = 1./(1. + np.exp(-2.0 * (fr - 5.0e-1)))
    T_a = T_inf + fr * (T_0 - T_inf)
    Z_m = params.PTM[6] * params.PT[47]
    Am = params.PTM[7] * params.PD.T[30, 0] * \
        (1. + globv(xlat, tloc, f107a, f107, params.PD.T[30:, 0], params))

    xmr = params.PDM[1, 1] * params.PD.T[15, 1] * \
        (1. + globv(xlat, tloc, f107a, f107, params.PD.T[16:, 1], params))
    if xmr < 1.e-3 :
        xmr = 1.e-3

    H1 = params.PDM[6, 1] * params.PD.T[46, 1]
    Zh0 = params.PDM[5, 1] * params.PD.T[45, 1]
    ymr = xmr * ccor(params.PDM[5, 0], params.PDM[4, 1], H1, Zh0)

    # Calculate mean mass
    xmm = (44.0 + 44.0 * ymr + \
           28.0 * params.PDM[1, 5])\
           /(1. + 2.0 * ymr + params.PDM[1, 5])
    zhm = params.PDM[3, 0]/params.PD.T[30, 1]

    # CO2 density
    CO2 = globv(xlat, tloc, f107a, f107, params.PD.T[:, 0], params)
    DB44 = params.PDM[0, 0] * np.exp(CO2) * params.PD.T[0, 0]
    Density[1], Temperature[1], tlb = denss(alt, DB44, T_inf, 44., 0., \
                                            params.PTM[5], Sig_1, T_0, Z_a,\
                                            T_a, Z_m, Am, Sig_e, params)
    ZH44 = params.PDM[2, 0] * params.PD.T[15, 0]

    # Get mixing density at zlb
    xmd = 44. - xmm
    B44, tz, tlb = denss(ZH44, DB44, T_inf, xmd, -1.0, \
                         params.PTM[5], Sig_1, T_0, Z_a,\
                         T_a, Z_m, Am, Sig_e, params)
    DM44, tz, tlb = denss(alt, B44, T_inf, xmm, 0.0, \
                          params.PTM[5], Sig_1, T_0, Z_a,\
                          T_a, Z_m, Am, Sig_e, params)
    Density[1] = dnet(Density[1], DM44, zhm, xmm, 44.0, params)

    # O Density

    O = globv(xlat, tloc, f107a, f107, params.PD.T[:, 1], params)
    DB16 = params.PDM[0, 1] * np.exp(O) * params.PD.T[0, 1]
    Density[2], Temperature[1], tlb = denss(alt, DB16, T_inf, 16., 0., \
                                            params.PTM[5], Sig_1, T_0, Z_a,\
                                            T_a, Z_m, Am, Sig_e, params)
    DM16 = DM44 * xmr
    Density[2] = dnet(Density[2], DM16, zhm, xmm, 16, params)
    Density[2] = Density[2] * ccor(alt, params.PDM[4, 1], H1, Zh0)

    # Get O Turbopause Estimate
    DD16, tz, tlb  = denss(ZH44, DB16, T_inf, 16., 0., \
                           params.PTM[5], Sig_1, T_0, Z_a,\
                           T_a, Z_m, Am, Sig_e, params)
    DMZ44, tz, tlb = denss(ZH44, B44, T_inf, xmm, 0., \
                           params.PTM[5], Sig_1, T_0, Z_a,\
                           T_a, Z_m, Am, Sig_e, params)

    ZH16 = turbo(DD16, DMZ44*xmr, ZH44, 16., xmm, tz, params)

    # CO Density
    CO = globv(xlat, tloc, f107a, f107, params.PD.T[:, 2], params)
    DB29 = params.PDM[0, 2] * np.exp(CO) * params.PD.T[0, 2]
    Density[3], Temperature[1], tlb = denss(alt, DB29, T_inf, 28., 0., \
                                            params.PTM[5], Sig_1, T_0, Z_a,\
                                            T_a, Z_m, Am, Sig_e, params)
    DM29 = DM44 * xmr
    Density[3] = dnet(Density[3], DM29, zhm, xmm, 28., params)
    Density[3] = Density[3] * ccor(alt, params.PDM[4, 1], H1, Zh0)

    # He Density
    HE = globv(xlat, tloc, f107a, f107, params.PD.T[:, 3], params)
    DB04 = params.PDM[0, 3] * np.exp(HE) * params.PD.T[0, 3]
    Density[4], Temperature[1], tlb = denss(alt, DB04, T_inf, 4., -0.6, \
                                            params.PTM[5], Sig_1, T_0, Z_a,\
                                            T_a, Z_m, Am, Sig_e, params)
    DM04 = DM44 * params.PDM[1, 3]*params.PD.T[15, 3]

    Density[4] = dnet(Density[4], DM04, zhm, xmm, 4., params)

    # N Density
    N = globv(xlat, tloc, f107a, f107, params.PD.T[:, 4], params)
    DB14 = params.PDM[0, 4] * np.exp(N) * params.PD.T[0, 4]
    Density[5], Temperature[1], tlb = denss(alt, DB14, T_inf, 14., 0., \
                                            params.PTM[5], Sig_1, T_0, Z_a,\
                                            T_a, Z_m, Am, Sig_e, params)
    ZH14 = ZH16
    xmd = 14. - xmm
    B14, tz, tlb = denss(ZH14, DB14, T_inf, xmd, -1., \
                params.PTM[5], Sig_1, T_0, Z_a,\
                T_a, Z_m, Am, Sig_e, params)
    DM14,tz, tlb = denss(alt, B14, T_inf, xmm, 0., \
                 params.PTM[5], Sig_1, T_0, Z_a,\
                 T_a, Z_m, Am, Sig_e, params)
    Density[5] = dnet(Density[5], DM14, zhm, xmm, 14., params)
    Density[5] = Density[5] * ccor(alt, params.PDM[4, 4], params.PDM[6, 4], params.PDM[5, 4])
    
    # N2 Density
    N2 = globv(xlat, tloc, f107a, f107, params.PD.T[:, 5], params)
    DB28 = params.PDM[0, 5] * np.exp(N2) * params.PD.T[0, 5]
    Density[6], Temperature[1], tlb = denss(alt, DB28, T_inf, 28., 0., \
                                            params.PTM[5], Sig_1, T_0, Z_a,\
                                            T_a, Z_m, Am, Sig_e, params)
    DM28 = DM44 * params.PDM[1, 5]
    Density[6] = dnet(Density[6], DM28, zhm, xmm, 28., params)

    # Total mss density
    Density[0] = 1.66e-24 * \
        (44. * Density[1] + \
         16. * Density[2] + \
         28. * Density[3] + \
         4. * Density[4] + \
         14. * Density[5] + \
         28. * Density[6])

    for i in range (0, 7):
        Density[i] = Density[i] * params.PDM[7, 0]
                                 
    return Density, Temperature

def load_inputfile(fname):
    df = pd.read_csv(fname, sep="\t", skiprows=2, decimal=',', header=None)
    index = ['ALT', 'LAT', 'LOCT']
    df.columns = index
    return df

def check_data(fname):
    df = pd.read_csv(fname, sep="\t", skiprows=2,header=None)
    index = ['Alt', 'Lat', 'Loct', \
             'Total', 'CO2', 'O', 'CO', 'HE', 'N', 'N2', \
             'T_exo', 'T_alt']
    df.columns = index

    fig, ax = plt.subplots()
    ax.plot(df['Alt'], df['CO2'], label=index[4])
    ax.plot(df['Alt'], df['O'], label=index[5])
    ax.plot(df['Alt'], df['CO'], label=index[6])
    ax.plot(df['Alt'], df['HE'], label=index[7])
    ax.plot(df['Alt'], df['N'], label=index[8])
    ax.plot(df['Alt'], df['N2'], label=index[9])
    ax.set_yscale('log')
    ax.legend()
    ax.grid(c='lightgray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Altitude [km]')
    ax.set_ylabel('Number Density [/cc]')
    plt.show()

    return

def main():
    # Parameters given in the model
    params = Parameters()

    # Input parameters
    # alt: Altitude (Km)
    # xlat: Latitude (DEG)
    # tloc: Local time (HRS)
    # f107a: average F10.7
    # f107: F10,7
    # mas: 48 for all the species (fixed)


    # For one set of input parameters and output
    alt = 130.0
    xlat = 0.0
    tloc = 15.0
    f107a = 200.0
    f107 = 200.0
    mas = 48

    
    Density, Temperature = vts(alt, xlat, tloc, f107a, f107, mas, params)
    print('Density -- Total, CO2, O, CO, He, N, N2')
    print(Density)
    print('--------------------------------')
    print('Temperature -- Exospheric, At Altitude')
    print(Temperature)
    print('BEFORE LOOP')
    
    
    '''
    # For loading input parameters from a file
    path = '/Users/saeaizawa/Desktop/MoaVTS/rewrite_20240223/'
    fname = path + '20110518_altlatlst.dat'
    df = load_inputfile(fname)
    ofname = path + 'output.txt'

    f107a = 200.
    f107 = 200.
    mas = 48.
    
    with open (ofname, 'w') as f:
        f.write(f'# Input file name: {fname}\n')
        f.write('# Alt[km], Lat[deg], Loct[HRS], Total Mass Density[gm/cc], C02[/cc], O, CO, HE, N, N2, \
        Exospheric temperature, Temperature ad given altitude \n')

        for i in range (0, len(df['ALT'])):
            Density, Temperature = vts(df['ALT'][i], df['LAT'][i], df['LOCT'][i], \
                                       f107a, f107, mas, params)
            f.write(f"{df['ALT'][i]}\t{df['LAT'][i]}\t{df['LOCT'][i]}\t"
                    f"{Density[0]}\t{Density[1]}\t{Density[2]}\t"
                    f"{Density[3]}\t{Density[4]}\t{Density[5]}\t"
                    f"{Density[6]}\t{Temperature[0]}\t{Temperature[1]}\n")

    # Check your output by visualizing the data
    check_data(ofname)
    
    '''
    '''
    for i in range (0, 31):
        altit = 100.0 + i*5.0
        Density, Temperature = vts(altit, 0, 15, 200, 200, mas, params)

    '''
    
    return

if __name__ == '__main__':
    main()
