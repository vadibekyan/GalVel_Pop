import numpy as np
import pandas as pd


def UVW(ra=None, dec=None, pmra=None, pmdec=None, vrad=None, plx = None, warnings = True):
    """
    This function determines the UVW galactic space velocity components with respect to the local standard of rest (LSR).
    ra (right ascension) and dec (declination): in degrees
    plx (parallax): in milli-arcseconds (mas)
    pmra (proper motion in right ascension direction), pmdec (proper motion in declination direction): in milli-arcseconds/yr (mas/yr)
    vrad (radial velocity): in km/s

    U - positive toward the Galactic anti-center
    V - positive in the direction of Galactic rotation
    W - positive toward the North Galactic Pole
    """

    if warnings == True:
        parameters = [ra, dec, pmra, pmdec, vrad, plx]
        parameters_str = ['ra', 'dec', 'pmra', 'pmdec', 'vrad', 'plx']
  
        for i, param in enumerate(parameters):  
            if param is None or param != param:
                print (f"Please check the value of {parameters_str[i]}")

    C = 4.74047     #Constant - The equivalent of 1 AU/yr in km/s

    # transformation matrix to calculate Galactic coordinates from the equatorial ones
    # taken from https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html

    matix_1 = np.array([[0.0548755604162154, +0.4941094278755837, -0.8676661490190047],
                  [0.8734370902348850, -0.4448296299600112, -0.1980763734312015],
                  [0.4838350155487132, 0.7469822444972189, +0.4559837761750669]])

    # the values of RA and DEC need to be transformed from deg to rad
    matix_2 = np.array([[-np.sin(np.deg2rad(ra)), np.cos(np.deg2rad(ra)), 0],
                  [-np.cos(np.deg2rad(ra))*np.sin(np.deg2rad(dec)), -np.sin(np.deg2rad(ra))*np.sin(np.deg2rad(dec)), np.cos(np.deg2rad(dec))],
                  [np.cos(np.deg2rad(ra))*np.cos(np.deg2rad(dec)), np.sin(np.deg2rad(ra))*np.cos(np.deg2rad(dec)), np.sin(np.deg2rad(dec))]])

    matix_3 = np.array([C*pmra/plx, C*pmdec/plx, vrad])


    matrix_21 =  np.dot(matix_2, matix_1) # multiplies the matix_1 and matix_2 matrices
    UVW = np.dot(matix_3, matrix_21) # multiplies the matix_3 and matix_21 matrices


    # (U,V,W)_Sun relative to LSR taken from  Schönrich et al. (2010)
    # The U_Sun_LSR is negative to make the U component positive towaards the direction of the Galactic anti-center
    Sun_lsr = np.array([-11.1, 12.24, 7.25]) 


    UVW_lsr = np.add(Sun_lsr, UVW)

    #print (f"U = {UVW_lsr[0]:.1f} km/s; V = {UVW_lsr[1]:.1f} km/s; W = {UVW_lsr[2]:.1f} km/s")
    return     UVW_lsr


def gal_population(U = None, V = None, W= None, char_vel = 'Mean'):
    """
    char_vel takes of of the three values: "Bensby", "Robin", or "Mean"
    If Bensby, the characteristic velocity dispersions, the asymmetric drift, and fraction of star in the stellar populations
    is taked from Bensby et al. (2005). 
    If "Robin", then from Robin et al. (2005). In case of "Mean", the mean values of the two works are taken.
    """

    # The characteristic velocity dispersions and the asymmetric drift for think disk (D), thick disk (TD), and halo (H) respectively
    #Bensby et al. (2005)
    sigmaU_b = np.array([35, 67, 160])
    sigmaV_b = np.array([20, 38, 90])
    sigmaW_b = np.array([16, 35, 90])
    Vasym_b = np.array([-15, -46, -220])
    #Robin et al. (2005)
    sigmaU_r = np.array([43, 67, 131])
    sigmaV_r = np.array([28, 51, 106])
    sigmaW_r = np.array([17, 42, 85])
    Vasym_r = np.array([-9, -48, -220])
    # Mean of Robin et al and Bensby et al.
    sigmaU_mean = np.add(sigmaU_b, sigmaU_r)/2
    sigmaV_mean = np.add(sigmaV_b, sigmaV_r)/2
    sigmaW_mean = np.add(sigmaW_b, sigmaW_r)/2
    Vasym_mean = np.add(Vasym_b, Vasym_r)/2


    #(overall) relative likelihoods of belonging to think disk (D), thick disk (TD), and halo (H) respectively in the solar neighborhood
    frac_b = np.array([0.9, 0.0985, 0.0015]) #Bensby et al. (2005)
    frac_r = np.array([0.924, 0.07, 0.006]) #Robin et al. (2005)
    frac_mean = np.add(frac_b, frac_r)/2 #mean

    if char_vel == "Bensby":
        sigmaU = sigmaU_b
        sigmaV = sigmaV_b
        sigmaW = sigmaW_b
        Vasym = Vasym_b
        frac = frac_b
    elif char_vel == "Robin":
        sigmaU = sigmaU_r
        sigmaV = sigmaV_r
        sigmaW = sigmaW_r
        Vasym = Vasym_r
        frac = frac_r
    elif char_vel == "Mean":
        sigmaU = sigmaU_mean
        sigmaV = sigmaV_mean
        sigmaW = sigmaW_mean
        Vasym = Vasym_mean
        frac = frac_mean

    #normalization factors for D, TD, and H
    k = np.array([1./(((2*np.pi)**1.5)*sigmaU[0]*sigmaV[0]*sigmaW[0]), 
                  1./(((2*np.pi)**1.5)*sigmaU[1]*sigmaV[1]*sigmaW[1]),
                  1./(((2*np.pi)**1.5)*sigmaU[2]*sigmaV[2]*sigmaW[2])])


    #"Absolute" probabilities of belonging to D, TD, and H
    abs_p = np.zeros(3)

    for i in range(3):
        abs_p[i] = frac[i]*k[i]*np.exp(-0.5*(U/sigmaU[i])**2 - 0.5*((V-Vasym[i])/sigmaV[i])**2 - 0.5*(W/sigmaW[i])**2)

    #Relative probabilities of belonging to D, TD, and H
    Prob = dict(zip(['Thin_disk', 'Thick_disk', 'Halo'], [0,0,0]))

    for i, (key, value) in enumerate(Prob.items()):
        Prob[key] = np.round(100*abs_p[i]/np.sum(abs_p),1)

    #print ({k: v for k, v in reversed(sorted(Prob.items(), key=lambda item: item[1]))})
    return list(Prob.values())



def UVW_prob_err(ra=None, dec=None, pmra=None, pmdec=None, vrad=None, plx = None, ra_err = None, dec_err = None, pmra_err = None, pmdec_err = None, vrad_err = None, plx_err = None, display = True, N = 1000):
    """
    N is the number of MonteCarlo realizations
    """

    parameters = [ra, dec, pmra, pmdec, vrad, plx]
    parameters_str = ['ra', 'dec', 'pmra', 'pmdec', 'vrad', 'plx']
  
    for i, param in enumerate(parameters):  
        if param is None or param != param:
            print (f"***WARNING*** Please check the value of {parameters_str[i]}")

    UVW_prob = pd.DataFrame(index=range(0,N), 
                            data = np.zeros((N,12)), 
                            columns=['U', 'V', 'W', 'Pthin_m', 'Pthick_m', 'Phalo_m', 'Pthin_b', 'Pthick_b', 'Phalo_b', 'Pthin_r', 'Pthick_r', 'Phalo_r'])


    for i in range(N):

        # This is to be sure that the uncertainties are not None or "Nan". For "nan", "nan" != "nan"
        if ra_err is not None and ra_err == ra_err:
            ra_tmp = np.random.normal(ra, ra_err, 1)[0]
        else:
            ra_tmp = ra

        if dec_err is not None and dec_err == dec_err:
            dec_tmp = np.random.normal(dec, dec_err, 1)[0]
        else:
            dec_tmp = dec

        if pmra_err is not None and pmra_err == pmra_err:
            pmra_tmp = np.random.normal(pmra, pmra_err, 1)[0]
        else:
            pmra_tmp = pmra
        
        if pmdec_err is not None and pmdec_err == pmdec_err:
            pmdec_tmp = np.random.normal(pmdec, pmdec_err, 1)[0]
        else:
            pmdec_tmp = pmdec
        
        if vrad_err is not None and vrad_err == vrad_err:
            vrad_tmp = np.random.normal(vrad, vrad_err, 1)[0]
        else:
            vrad_tmp = vrad

        if plx_err is not None and plx_err == plx_err:
            plx_tmp = np.random.normal(plx, plx_err, 1)[0]
        else:
            plx_tmp = plx

        U, V, W = UVW(ra=ra_tmp, dec=dec_tmp, pmra=pmra_tmp, pmdec=pmdec_tmp, vrad=vrad_tmp, plx=plx_tmp, warnings = False)

        Pthin_m, Pthick_m, Phalo_m = gal_population(U = U, V = V, W= W, char_vel = 'Mean')
        Pthin_b, Pthick_b, Phalo_b = gal_population(U = U, V = V, W= W, char_vel = 'Bensby')
        Pthin_r, Pthick_r, Phalo_r = gal_population(U = U, V = V, W= W, char_vel = 'Robin')


        UVW_prob['U'][i] = U
        UVW_prob['V'][i] = V
        UVW_prob['W'][i] = W
        UVW_prob['Pthin_m'][i] = Pthin_m
        UVW_prob['Pthick_m'][i] = Pthick_m
        UVW_prob['Phalo_m'][i] = Phalo_m
        UVW_prob['Pthin_b'][i] = Pthin_b
        UVW_prob['Pthick_b'][i] = Pthick_b
        UVW_prob['Phalo_b'][i] = Phalo_b
        UVW_prob['Pthin_r'][i] = Pthin_r
        UVW_prob['Pthick_r'][i] = Pthick_r
        UVW_prob['Phalo_r'][i] = Phalo_r

    UVW_prob_mean_std = UVW_prob.describe().loc[['mean','std']]
    #format the output
    UVW_prob_mean_std = UVW_prob_mean_std.applymap(lambda x: f"{x:.1f}")

    if display == True:
        print (UVW_prob_mean_std)

    return \
    UVW_prob_mean_std['U'][0], UVW_prob_mean_std['U'][1],               \
    UVW_prob_mean_std['V'][0], UVW_prob_mean_std['V'][1],               \
    UVW_prob_mean_std['W'][0], UVW_prob_mean_std['W'][1],               \
    UVW_prob_mean_std['Pthin_m'][0], UVW_prob_mean_std['Pthin_m'][1],   \
    UVW_prob_mean_std['Pthick_m'][0], UVW_prob_mean_std['Pthick_m'][1], \
    UVW_prob_mean_std['Phalo_m'][0], UVW_prob_mean_std['Phalo_m'][1],   \
    UVW_prob_mean_std['Pthin_b'][0], UVW_prob_mean_std['Pthin_b'][1],   \
    UVW_prob_mean_std['Pthick_b'][0], UVW_prob_mean_std['Pthick_b'][1], \
    UVW_prob_mean_std['Phalo_b'][0], UVW_prob_mean_std['Phalo_b'][1],   \
    UVW_prob_mean_std['Pthin_r'][0], UVW_prob_mean_std['Pthin_r'][1],   \
    UVW_prob_mean_std['Pthick_r'][0], UVW_prob_mean_std['Pthick_r'][1], \
    UVW_prob_mean_std['Phalo_r'][0], UVW_prob_mean_std['Phalo_r'][1]

     

def UVW_prob_err_sample(display = False, N = 1000):
    """
    This function determines the UVW velocities and the probabilities of belonging to different populations.
    The input parameters are taken from "gal_vel_pop_param.rdb" table.
    The table should have the following columns:
    "star    ra	dec	pmra	pmdec	vrad	plx	ra_err	dec_err	pmra_err	pmdec_err	vrad_err	plx_err"

    INPUT KEYWORD:
        (1) "display': if True, displays the calculated parameters for each star.
        (2) "N": The number of MonteCarlo realizations.
        
    """

    input_table = pd.read_csv('gal_vel_param.rdb', sep = '\t')


    output_table = pd.DataFrame(index=range(0,len(input_table.star)), 
                            data = np.zeros((len(input_table.star),24)), 
                            columns=['U', 'U_err', 'V', 'V_err', 'W', 'W_err', 
                                    'Pthin_m', 'Pthin_m_err', 'Pthick_m', 'Pthick_m_err', 'Phalo_m', 'Phalo_m_err',
                                    'Pthin_b', 'Pthin_b_err', 'Pthick_b', 'Pthick_b_err', 'Phalo_b', 'Phalo_b_err',
                                    'Pthin_r', 'Pthin_r_err', 'Pthick_r', 'Pthick_r_err', 'Phalo_r', 'Phalo_r_err'])


    for i, star in enumerate(input_table.star):

        print (f'{i+1} of {len(input_table.star)}: {star}')

        output_table['U'][i], output_table['U_err'][i], output_table['V'][i], output_table['V_err'][i], output_table['W'][i], output_table['W_err'][i], output_table['Pthin_m'][i], output_table['Pthin_m_err'][i], output_table['Pthick_m'][i], output_table['Pthick_m_err'][i], output_table['Phalo_m'][i], output_table['Phalo_m_err'][i], output_table['Pthin_b'][i], output_table['Pthin_b_err'][i], output_table['Pthick_b'][i], output_table['Pthick_b_err'][i], output_table['Phalo_b'][i], output_table['Phalo_b_err'][i], output_table['Pthin_r'][i], output_table['Pthin_r_err'][i], output_table['Pthick_r'][i], output_table['Pthick_r_err'][i], output_table['Phalo_r'][i], output_table['Phalo_r_err'][i] = \
                    UVW_prob_err(ra = input_table['ra'][i], dec = input_table['dec'][i], 
                                pmra = input_table['pmra'][i], 
                                pmdec = input_table['pmdec'][i],
                                vrad = input_table['vrad'][i],
                                plx = input_table['plx'][i],
                                ra_err = input_table['ra_err'][i],
                                dec_err = input_table['dec_err'][i],
                                pmra_err = input_table['pmra_err'][i],
                                pmdec_err = input_table['pmdec_err'][i],
                                vrad_err = input_table['vrad_err'][i],
                                plx_err = input_table['plx_err'][i],
                                display = display, N = N)

    print ("\nDone")

    output_table.to_csv('gal_vel_pop_results.rdb', index = False, sep = '\t')

