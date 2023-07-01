#!/usr/bin/env python3

import numpy as np
import pandas as pd


def UVW(ra=None, dec=None, pmra=None, pmdec=None, vrad=None, plx=None, warnings=True):
    """

    This function determines the UVW galactic space velocity components with respect to the local standard of rest (LSR), 
    using transformations from equatorial to galactic coordinates. 

    The output is an array [U, V, W] representing the galactic space velocity components in km/s. 
    The U component is positive toward the Galactic anti-center, the V component is positive 
    in the direction of Galactic rotation, and the W component is positive toward the North Galactic Pole.

    The function includes a commented line that prints the calculated velocities. 
    If you want to see the velocities printed, uncomment that line by removing the # character at the beginning of the line.

    
    Parameters:
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        pmra (float): Proper motion in the right ascension direction in milli-arcseconds/yr (mas/yr).
        pmdec (float): Proper motion in the declination direction in milli-arcseconds/yr (mas/yr).
        vrad (float): Radial velocity in km/s.
        plx (float): Parallax in milli-arcseconds (mas).
        warnings (bool): Flag to display warnings for missing or invalid input values. Default is True.
        
    Returns:
        numpy.ndarray: Array containing the UVW galactic space velocity components in km/s.
        
    Note:
        U - Positive toward the Galactic anti-center.
        V - Positive in the direction of Galactic rotation.
        W - Positive toward the North Galactic Pole.
    """

    if warnings:
        parameters = [ra, dec, pmra, pmdec, vrad, plx]
        parameters_str = ['ra', 'dec', 'pmra', 'pmdec', 'vrad', 'plx']
  
        for i, param in enumerate(parameters):
            # Check if the parameter is None or NaN
            if param is None or param != param:
                print(f"Please check the value of {parameters_str[i]}")

    C = 4.74047  # Constant - The equivalent of 1 AU/yr in km/s

    # Transformation matrix to calculate Galactic coordinates from the equatorial ones
    # Taken from https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html
    matrix_1 = np.array([[0.0548755604162154, +0.4941094278755837, -0.8676661490190047],
                  [0.8734370902348850, -0.4448296299600112, -0.1980763734312015],
                  [0.4838350155487132, 0.7469822444972189, +0.4559837761750669]])

    # The values of RA and DEC need to be transformed from deg to rad
    matrix_2 = np.array([[-np.sin(np.deg2rad(ra)), np.cos(np.deg2rad(ra)), 0],
                        [-np.cos(np.deg2rad(ra)) * np.sin(np.deg2rad(dec)),
                         -np.sin(np.deg2rad(ra)) * np.sin(np.deg2rad(dec)), np.cos(np.deg2rad(dec))],
                        [np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec)),
                         np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec)), np.sin(np.deg2rad(dec))]])

    matrix_3 = np.array([C * pmra / plx, C * pmdec / plx, vrad])


    matrix_21 =  np.dot(matrix_2, matrix_1) # multiplies the matix_1 and matix_2 matrices
    UVW = np.dot(matrix_3, matrix_21)  # Multiply the matrix_3 and matrix_21 matrices

    # (U,V,W)_Sun relative to LSR taken from Schönrich et al. (2010)
    # The U_Sun_LSR is negative to make the U component positive towards the direction of the Galactic anti-center
    Sun_lsr = np.array([-11.1, 12.24, 7.25])

    UVW_lsr = np.add(Sun_lsr, UVW)

    # Uncomment the line below to print the calculated velocities
    # print(f"U = {UVW_lsr[0]:.1f} km/s; V = {UVW_lsr[1]:.1f} km/s; W = {UVW_lsr[2]:.1f} km/s")

    return UVW_lsr


def gal_population(U=None, V=None, W=None, char_vel='Mean'):
    """

    The code calculates the relative probabilities of belonging to different galactic 
    populations based on the given U, V, and W galactic space velocity components. 
    It uses characteristic velocity dispersion values, asymmetric drift values, and 
    fraction of stars in each population from two different works (Bensby et al. and Robin et al.) 
    or their mean values. The probabilities are calculated using Gaussian probability 
    distributions and normalization factors. The function returns a list of the 
    relative probabilities for the Thin disk, Thick disk, and Halo populations.


    This function calculates the relative probabilities of belonging to different galactic populations (Thin disk, Thick disk, and Halo)
    based on the given values of U, V, and W galactic space velocity components.
    
    Parameters:
        U (float): U galactic space velocity component in km/s.
        V (float): V galactic space velocity component in km/s.
        W (float): W galactic space velocity component in km/s.
        char_vel (str): Characteristic velocity source. Can be 'Bensby', 'Robin', or 'Mean'. Default is 'Mean'.
        
    Returns:
        list: List containing the relative probabilities of belonging to Thin disk, Thick disk, and Halo populations.
    """

    # The characteristic velocity dispersions and asymmetric drift for the thin disk (D), thick disk (TD), and halo (H) populations
    sigmaU_b = np.array([35, 67, 160])  # Bensby et al. (2005)
    sigmaV_b = np.array([20, 38, 90])
    sigmaW_b = np.array([16, 35, 90])
    Vasym_b = np.array([-15, -46, -220])
    
    sigmaU_r = np.array([43, 67, 131])  # Robin et al. (2005)
    sigmaV_r = np.array([28, 51, 106])
    sigmaW_r = np.array([17, 42, 85])
    Vasym_r = np.array([-9, -48, -220])
    
    sigmaU_mean = np.add(sigmaU_b, sigmaU_r) / 2  # Mean of Bensby et al. and Robin et al.
    sigmaV_mean = np.add(sigmaV_b, sigmaV_r) / 2
    sigmaW_mean = np.add(sigmaW_b, sigmaW_r) / 2
    Vasym_mean = np.add(Vasym_b, Vasym_r) / 2

    # Overall relative likelihoods of belonging to thin disk (D), thick disk (TD), and halo (H) populations in the solar neighborhood
    frac_b = np.array([0.9, 0.0985, 0.0015])  # Bensby et al. (2005)
    frac_r = np.array([0.924, 0.07, 0.006])  # Robin et al. (2005)
    frac_mean = np.add(frac_b, frac_r) / 2  # Mean

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

    # Normalization factors for the thin disk (D), thick disk (TD), and halo (H)
    k = np.array([1. / (((2 * np.pi) ** 1.5) * sigmaU[0] * sigmaV[0] * sigmaW[0]),
                  1. / (((2 * np.pi) ** 1.5) * sigmaU[1] * sigmaV[1] * sigmaW[1]),
                  1. / (((2 * np.pi) ** 1.5) * sigmaU[2] * sigmaV[2] * sigmaW[2])])

    # "Absolute" probabilities of belonging to the thin disk (D), thick disk (TD), and halo (H)
    abs_p = np.zeros(3)

    for i in range(3):
        abs_p[i] = frac[i] * k[i] * np.exp(-0.5 * (U / sigmaU[i]) ** 2 - 0.5 * ((V - Vasym[i]) / sigmaV[i]) ** 2 - 0.5 * (
                W / sigmaW[i]) ** 2)

    # Relative probabilities of belonging to the thin disk, thick disk, and halo populations
    Prob = dict(zip(['Thin_disk', 'Thick_disk', 'Halo'], [0, 0, 0]))

    for i, (key, value) in enumerate(Prob.items()):
        Prob[key] = np.round(100 * abs_p[i] / np.sum(abs_p), 1)

    return list(Prob.values())




def UVW_prob_err(ra=None, dec=None, pmra=None, pmdec=None, vrad=None, plx=None, ra_err=None, dec_err=None, pmra_err=None, pmdec_err=None, vrad_err=None, plx_err=None, display=True, N=1000):
    """

    The code calculates the mean and standard deviation of the U, V, and W galactic 
    space velocity components and the relative probabilities of belonging to different 
    galactic populations (Thin disk, Thick disk, and Halo) using Monte Carlo simulations. 
    It generates random samples for the input parameters based on their uncertainties 
    and then computes the corresponding values for each realization. The results are 
    stored in a DataFrame and summarized by calculating the mean and standard deviation. 
    The output is formatted and displayed if display is set to True. Finally, the mean 
    and standard deviation values are returned as a tuple.


    Parameters:
        ra (float): Right ascension in degrees.
        dec (float): Declination in degrees.
        pmra (float): Proper motion in right ascension in mas/yr.
        pmdec (float): Proper motion in declination in mas/yr.
        vrad (float): Radial velocity in km/s.
        plx (float): Parallax in mas.
        ra_err (float): Uncertainty in right ascension in degrees.
        dec_err (float): Uncertainty in declination in degrees.
        pmra_err (float): Uncertainty in proper motion in right ascension in mas/yr.
        pmdec_err (float): Uncertainty in proper motion in declination in mas/yr.
        vrad_err (float): Uncertainty in radial velocity in km/s.
        plx_err (float): Uncertainty in parallax in mas.
        display (bool): Whether to display the mean and standard deviation of the results. Default is True.
        N (int): Number of Monte Carlo realizations. Default is 1000.

    Returns:
        tuple: Tuple containing the mean and standard deviation of U, V, and W galactic space velocity components,
               as well as the mean and standard deviation of the relative probabilities for the Thin disk, Thick disk, and Halo populations.
    """

    parameters = [ra, dec, pmra, pmdec, vrad, plx]
    parameters_str = ['ra', 'dec', 'pmra', 'pmdec', 'vrad', 'plx']

    for i, param in enumerate(parameters):
        if param is None or param != param:
            print(f"***WARNING*** Please check the value of {parameters_str[i]}")

    # Create a DataFrame to store the results
    UVW_prob = pd.DataFrame(index=range(0, N), data=np.zeros((N, 12)), columns=['U', 'V', 'W', 'Pthin_m', 'Pthick_m', 'Phalo_m', 'Pthin_b', 'Pthick_b', 'Phalo_b', 'Pthin_r', 'Pthick_r', 'Phalo_r'])

    for i in range(N):
        # Generate random samples for the parameters based on their uncertainties
        ra_tmp = np.random.normal(ra, ra_err, 1)[0] if ra_err is not None and ra_err == ra_err else ra
        dec_tmp = np.random.normal(dec, dec_err, 1)[0] if dec_err is not None and dec_err == dec_err else dec
        pmra_tmp = np.random.normal(pmra, pmra_err, 1)[0] if pmra_err is not None and pmra_err == pmra_err else pmra
        pmdec_tmp = np.random.normal(pmdec, pmdec_err, 1)[0] if pmdec_err is not None and pmdec_err == pmdec_err else pmdec
        vrad_tmp = np.random.normal(vrad, vrad_err, 1)[0] if vrad_err is not None and vrad_err == vrad_err else vrad
        plx_tmp = np.random.normal(plx, plx_err, 1)[0] if plx_err is not None and plx_err == plx_err else plx

        # Calculate the U, V, and W galactic space velocity components
        U, V, W = UVW(ra=ra_tmp, dec=dec_tmp, pmra=pmra_tmp, pmdec=pmdec_tmp, vrad=vrad_tmp, plx=plx_tmp, warnings=False)

        # Calculate the relative probabilities for different galactic populations using different characteristic velocity sets
        Pthin_m, Pthick_m, Phalo_m = gal_population(U=U, V=V, W=W, char_vel='Mean')
        Pthin_b, Pthick_b, Phalo_b = gal_population(U=U, V=V, W=W, char_vel='Bensby')
        Pthin_r, Pthick_r, Phalo_r = gal_population(U=U, V=V, W=W, char_vel='Robin')

        # Store the results in the DataFrame
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

    # Calculate the mean and standard deviation of the results
    UVW_prob_mean_std = UVW_prob.describe().loc[['mean', 'std']]
    # Format the output
    UVW_prob_mean_std = UVW_prob_mean_std.applymap(lambda x: f"{x:.1f}")

    if display:
        print(UVW_prob_mean_std)

    # Return the mean and standard deviation of U, V, W, and the relative probabilities
    return (
        UVW_prob_mean_std['U'][0],
        UVW_prob_mean_std['U'][1],
        UVW_prob_mean_std['V'][0],
        UVW_prob_mean_std['V'][1],
        UVW_prob_mean_std['W'][0],
        UVW_prob_mean_std['W'][1],
        UVW_prob_mean_std['Pthin_m'][0],
        UVW_prob_mean_std['Pthin_m'][1],
        UVW_prob_mean_std['Pthick_m'][0],
        UVW_prob_mean_std['Pthick_m'][1],
        UVW_prob_mean_std['Phalo_m'][0],
        UVW_prob_mean_std['Phalo_m'][1],
        UVW_prob_mean_std['Pthin_b'][0],
        UVW_prob_mean_std['Pthin_b'][1],
        UVW_prob_mean_std['Pthick_b'][0],
        UVW_prob_mean_std['Pthick_b'][1],
        UVW_prob_mean_std['Phalo_b'][0],
        UVW_prob_mean_std['Phalo_b'][1],
        UVW_prob_mean_std['Pthin_r'][0],
        UVW_prob_mean_std['Pthin_r'][1],
        UVW_prob_mean_std['Pthick_r'][0],
        UVW_prob_mean_std['Pthick_r'][1],
        UVW_prob_mean_std['Phalo_r'][0],
        UVW_prob_mean_std['Phalo_r'][1]
    )
