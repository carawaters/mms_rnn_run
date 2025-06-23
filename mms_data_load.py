"""
    Loads MMS burst data needed for typical analysis, incorporating some
    corrections. Calculates currents and J.E quantities. Includes functions
    for averaging data and finding perpendicular and parallel components.

    Author: Cara Waters
    06/11/2024

"""


import pyspedas
import pytplot
import pandas as pd
import numpy as np
import os


# Define colours
red = (213, 94, 0)
green = (0, 158, 115)
blue = (86, 180, 233)

def mms_ave(var1: str, var2: str, var2ave: str) -> None:
    """
    Averages var2 onto var1 and stores the result in var2ave.

    Args:
        var1 (str): Name of the tplot variable to average onto
        var2 (str): Name of the tplot variable to average
        var2ave (str): Name of the tplot variable to store the average in

    Raises:
        Exception: If var2 is empty
    """
    
    t1, d1 = pytplot.get_data(var1)
    t2, d2 = pytplot.get_data(var2)

    dt = t1[1:] - t1[:-1]
    dt_half = dt/2
    steps = t1[:-1] + dt_half
    # Gives threshold times for limits of taking mean
    np.append(steps, steps[-1] + dt_half[-1])
    means = np.zeros_like(d1)
    
    for i in range(len(steps)):
        if len(np.shape(d2)) == 1:
            if i == 0:
                index = np.where(t2 < steps[i])
            elif i == len(steps)-1:
                index = np.where((t2 >= steps[i-1]) & (t2 <= steps[i]))
            else:
                index = np.where((t2 >= steps[i-1]) & (t2 < steps[i]))
            val = np.nanmean(d2[index])
            if val == np.nan:
                means[i] = means[i-1]
            else:
                means[i] = val
        if len(np.shape(d2)) == 0:
            raise Exception('var2 must not be empty')
        else: # Accounts for multiple dimensions
            if i == 0:
                index = np.where(t2 <= steps[i])
            elif i == len(steps)-1:
                index = np.where(t2 >= steps[i])
            else:
                index = np.where((t2 >= steps[i-1]) & (t2 <= steps[i]))
            val = np.nanmean(d2[index,:], axis=1)
            if np.sum(np.isnan(val)) > 0:
                means[i,:] = means[i-1,:]
            else:
                means[i,:] = val

    pytplot.store_data(var2ave, data={'x':t1, 'y':means})

    return None

def vperppara_xyz(vel: str, magf: str, vperp_mag: str = None, vpara: str = None,
                  vperp_xyz: str = None):
    """Finds perpendicular and parallel components of vector relative to another

    Args:
        vel (str): tplot variable name to find components of
        magf (str): tplot variable name to find components relative to
        vperp_mag (str, optional): tplot variable name for output perpendicular 
                                    component magnitude. Defaults to None.
        vpara (str, optional): tplot variable name for output parallel
                                    component. Defaults to None.
        vperp_xyz (str, optional): tplot variable name for output perpendicular
                                    components. Defaults to None.

    Returns:
        None
    """

    pyspedas.tinterpol(magf, vel, newname='b_interp')

    t1, vec1 = pytplot.get_data(vel)
    _, vec2 = pytplot.get_data('b_interp')

    vpara_t = np.sum(vec1*vec2, axis=1)/np.sqrt(np.sum(vec2*vec2, axis=1))
    vperp_t = np.sqrt(np.sum(vec1*vec1, axis=1) - vpara_t**2)

    if vperp_mag != None:
        pytplot.store_data(vperp_mag, data={'x':t1, 'y':vperp_t})
    if vpara != None:
        pytplot.store_data(vpara, data={'x':t1, 'y':vpara_t})
    
    tv, v = pytplot.get_data(vel)
    _, b = pytplot.get_data('b_interp')
    pyspedas.tdotp(vel, 'b_interp', newname='vdotb')
    _, vdotb = pytplot.get_data('vdotb')

    vperpx = v[:, 0] - vdotb * b[:, 0] / (b[:, 0]**2 + b[:, 1]**2 + b[:, 2]**2)
    vperpy = v[:, 1] - vdotb * b[:, 1] / (b[:, 0]**2 + b[:, 1]**2 + b[:, 2]**2)
    vperpz = v[:, 2] - vdotb * b[:, 2] / (b[:, 0]**2 + b[:, 1]**2 + b[:, 2]**2)

    vperp = np.column_stack((vperpx, vperpy, vperpz))

    if vperp_xyz != None:
        pytplot.store_data(vperp_xyz, data={'x':tv, 'y':vperp})

    return None

def mms_load_brst(probe_id: int, t_start: str, t_end: str) -> None:
    """
    Load MMS burst data for a given spacecraft number and time range. Also
    applies penetrating radiation corrections to ions and centres FPI data.

    Args:
        probe_id (int): MMS spacecraft number
        t_start (str): "YYYY-MM-DDTHH:MM:SS"
        t_end (str): "YYYY-MM-DDTHH:MM:SS"

    Returns:
        None
    """

    probe_id_str = str(probe_id)
    sc_id = "mms" + probe_id_str

    pyspedas.mms.fpi(trange = [t_start, t_end], probe = probe_id_str,
                     datatype = ['dis-moms', 'des-moms'],
                     data_rate = 'brst', level = 'l2', time_clip = True)
    pyspedas.mms.fgm(trange = [t_start, t_end], probe = probe_id_str,
                     data_rate = 'brst', time_clip = True)

    # --------------------------------------------------------------------------
    # Account for background signatures in DIS
    # Methods described in MMS CMAD Section 4.5.7.5
    # https://lasp.colorado.edu/mms/sdc/public/MMS%20CMAD%20v9Sept2024.pdf
    # --------------------------------------------------------------------------

    # Number density
    t, ni = pytplot.get_data(sc_id + '_dis_numberdensity_brst')
    t, nibg = pytplot.get_data(sc_id + '_dis_numberdensity_bg_brst')
    ni_fix = ni - nibg
    pytplot.store_data(sc_id + '_dis_numberdensity_brst', data = {'x': t,
                                                                'y': ni_fix})

    # Bulk velocity
    t, vi = pytplot.get_data(sc_id + '_dis_bulkv_gse_brst')
    vi_fix = np.einsum('t, ti->ti', ni/ni_fix, vi)
    pytplot.store_data(sc_id + '_dis_bulkv_gse_brst', data = {'x': t,
                                                            'y': vi_fix})

    # Pressure tensor
    t, pi = pytplot.get_data(sc_id + '_dis_prestensor_gse_brst')
    t, pibgtr = pytplot.get_data(sc_id + '_dis_pres_bg_brst')
    pibg = np.einsum('t,ij->tij', pibgtr, np.identity(3))
    pi_fix = (pi * 1e-9 + 1.6726e-27 * np.einsum('t,tij->tij', ni * 1e6,
                np.einsum('ti,tj->tij', vi * 1e3, vi * 1e3)) - pibg * 1e-9
                - 1.6726e-27 * np.einsum('t,tij->tij', ni_fix * 1e6,
                np.einsum('ti,tj->tij', vi_fix * 1e3, vi_fix * 1e3))) # in Pa
    pi_fix_npa = pi_fix * 1e9 # in nPa
    pytplot.store_data(sc_id + '_dis_prestensor_gse_brst', data = {'x': t,
                                                            'y': pi_fix_npa})

    # Temperature tensor
    tij = np.einsum('t,tij->tij', 1/(ni * 1e6), pi_fix) / (1.3807e-23 * 11604)

    # Get parallel and perpendicular temperatures
    pyspedas.tinterpol(sc_id + '_fgm_b_gse_brst_l2_bvec',
                       sc_id + '_dis_numberdensity_brst',
                       newname = sc_id + '_dis_bvec_interp', method = 'linear')
    tb, b = pytplot.get_data(sc_id + '_dis_bvec_interp')
    b = b[:, 0:3] # get just vector components
    bhat = b / np.linalg.norm(b, axis=1)[:, None]

    tpar = np.einsum('tj,tij,ti->t', bhat, tij, bhat)
    tperp = (np.trace(tij, axis1 = 1, axis2 = 2) - tpar) / 2

    pytplot.store_data(sc_id + '_dis_temppara_brst', data = {'x': t, 'y': tpar})
    pytplot.store_data(sc_id + '_dis_tempperp_brst', data = {'x': t, 'y': tperp})

    # Copy to renamed variables for ease of use
    pytplot.tplot_copy(sc_id + '_dis_numberdensity_brst', sc_id + '_Ni')
    pytplot.tplot_copy(sc_id + '_des_numberdensity_brst', sc_id + '_Ne')
    pytplot.tplot_copy(sc_id + '_dis_bulkv_gse_brst', sc_id + '_vi')
    pytplot.tplot_copy(sc_id + '_des_bulkv_gse_brst', sc_id + '_ve')
    pytplot.tplot_copy(sc_id + '_dis_temppara_brst', sc_id + '_Ti_para')
    pytplot.tplot_copy(sc_id + '_dis_tempperp_brst', sc_id + '_Ti_perp')
    pytplot.tplot_copy(sc_id + '_des_temppara_brst', sc_id + '_Te_para')
    pytplot.tplot_copy(sc_id + '_des_tempperp_brst', sc_id + '_Te_perp')

    # Store an array of variable names for each
    var_dis = [sc_id + '_Ni', sc_id + '_vi', sc_id + '_Ti_para',
               sc_id + '_Ti_perp']
    var_des = [sc_id + '_Ne', sc_id + '_ve', sc_id + '_Te_para',
               sc_id + '_Te_perp']
    
    # --------------------------------------------------------------------------
    # Shift FPI data to center the timestamps
    # --------------------------------------------------------------------------

    # Ions
    for var in var_dis:
        t, d = pytplot.get_data(var)
        pytplot.store_data(var, data = {'x': t + 0.075, 'y': d})
    
    t, d, v = pytplot.get_data(sc_id + '_dis_energyspectr_omni_brst')
    pytplot.store_data(sc_id + '_dis_energyspectr_omni_brst',
                       data = {'x': t + 0.075, 'y': d, 'v': v})
    
    # Electrons
    for var in var_des:
        t, d = pytplot.get_data(var)
        pytplot.store_data(var, data = {'x': t + 0.075, 'y': d})

    var_des_spectr = [sc_id+'_des_energyspectr_omni_brst',
                      sc_id+'_des_energyspectr_par_brst',
                      sc_id+'_des_energyspectr_perp_brst',
                      sc_id+'_des_energyspectr_anti_brst']
    
    for var in var_des_spectr:
        t, d, v = pytplot.get_data(var)
        pytplot.store_data(var, data = {'x': t + 0.075, 'y': d, 'v': v})
    
    # --------------------------------------------------------------------------
    # Field data clean up variables
    # --------------------------------------------------------------------------

    pyspedas.mms.edp(trange = [t_start, t_end], probe = probe_id_str,
                        data_rate = 'brst', level = 'l2', time_clip = True)
    pyspedas.mms.edp(trange = [t_start, t_end], probe = probe_id_str,
                        data_rate = 'fast', level = 'l2', time_clip = True)
    pytplot.tplot_copy(sc_id + '_edp_dce_gse_brst_l2', sc_id + '_E')
    pytplot.tplot_copy(sc_id + '_edp_dce_gse_fast_l2', sc_id + '_E_fast')

    pytplot.tplot_copy(sc_id + '_fgm_b_gse_brst_l2_bvec', sc_id + '_B')
    t, d = pytplot.get_data(sc_id + '_B')
    # Calculate magnitude just to be sure
    pytplot.store_data(sc_id + '_Bmag', data = {'x': t, 'y':
                            np.sqrt(d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2)})
    
    return None

def mms_currents(probe_id: int):
    """
    Calculates currents and J.E quantities for a given spacecraft

    Args:
        probe_id (int): _description_

    Raises:
        ValueError: If no data loaded for spacecraft

    Returns:
        None
    """

    probe_id_str = str(probe_id)
    sc_id = "mms" + probe_id_str

    # Assuming data already loaded

    try:
        pyspedas.get_data(sc_id + '_Ni')
    except:
        raise ValueError("No data loaded for spacecraft " + probe_id_str +
                          "Have you run mms_load_brst() yet?")
    
    # --------------------------------------------------------------------------
    # Calculate currents
    # Current at electron scale
    # Total current
    # Ion current
    # Electron current
    # --------------------------------------------------------------------------

    # Ion velocity to electron timescale
    pyspedas.tinterpol(sc_id + '_vi', sc_id + '_ve', suffix='_des',
                       method='linear')
    
    _, vii = pytplot.get_data(sc_id + '_vi_des')
    t, vee = pytplot.get_data(sc_id + '_ve')
    _, nee = pytplot.get_data(sc_id + '_Ne')

    current_e = (np.reshape(np.repeat(nee, 3), (len(nee), 3))
                    * (vii - vee) * 1.6e-10 * 1e6) # in uA
    pytplot.store_data(sc_id + '_current_fpi', data = {'x': t, 'y': current_e})
    t, d = pytplot.get_data(sc_id + '_current_fpi')
    pytplot.store_data(sc_id + '_jmag', data = {'x': t, 'y':
                                np.sqrt(d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2)})
    
    ioncurrent = (np.reshape(np.repeat(nee, 3), (len(nee), 3))
                  * vii * 1.6e-10 * 1e6) # uA
    pytplot.store_data(sc_id + '_current_fpi_ions',
                       data = {'x': t, 'y': ioncurrent})
    t, d = pytplot.get_data(sc_id + '_current_fpi_ions')
    pytplot.store_data(sc_id + '_jmag_ions', data = {'x': t, 'y':
                                np.sqrt(d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2)})
    
    eleccurrent = (np.reshape(np.repeat(nee, 3), (len(nee), 3))
                    * -1 * vee * 1.6e-10 * 1e6) # uA
    pytplot.store_data(sc_id + '_current_fpi_elec',
                       data = {'x': t, 'y': eleccurrent})
    t, d = pytplot.get_data(sc_id + '_current_fpi_elec')
    pytplot.store_data(sc_id + '_jmag_elec', data = {'x': t, 'y':
                                np.sqrt(d[:, 0]**2 + d[:, 1]**2 + d[:, 2]**2)})
    
    # --------------------------------------------------------------------------
    # Calculate j.E quantities
    # --------------------------------------------------------------------------
    
    # Average E and B onto ve to use with this
    mms_ave(sc_id + '_ve', sc_id + '_E_fast', sc_id + '_E_ave_ve')
    mms_ave(sc_id + '_ve', sc_id + '_B', sc_id + '_B_ave_ve')

    # Average B onto ve and vi
    mms_ave(sc_id + '_ve', sc_id + '_B', sc_id + '_B_des')
    mms_ave(sc_id + '_vi', sc_id + '_B', sc_id + '_B_dis')

    # Calculate -vexB at electron scale and find convection E field
    tv, v = pytplot.get_data(sc_id + '_ve')
    tb, b = pytplot.get_data(sc_id + '_B_ave_ve')

    cross_prod = -np.cross(v, b)
    v_cross_b = 1e-3 * cross_prod # remember this is actually -vexB

    # This is the vxB E field - the convection electric field
    pytplot.store_data(sc_id + '_vexB_efield', data = {'x': tv, 'y': v_cross_b})

    # Calculate E + ve x B

    te, e_des = pytplot.get_data(sc_id + '_E_ave_ve') # measured E field
    tvb, vexb = pytplot.get_data(sc_id + '_vexB_efield') # -ve x B

    e_frozen = e_des - vexb # E + ve x B - non-zero if not frozen in

    pytplot.store_data(sc_id + '_e_frozen', data = {'x': te, 'y': e_frozen})

    # Calculate J.E' and J.E

    pyspedas.tdotp(sc_id + '_current_fpi', sc_id + '_e_frozen',
                   sc_id + '_jdoteprime') # in nW/m3
    pyspedas.tdotp(sc_id + '_current_fpi', sc_id + '_E_ave_ve',
                   sc_id + '_jdote') # nW/m3

    # Calculate Ji.E' and Ji.E

    pyspedas.tdotp(sc_id + '_current_fpi_ions', sc_id + '_e_frozen',
                    sc_id + '_jidoteprime') # in nW/m3
    pyspedas.tdotp(sc_id + '_current_fpi_ions', sc_id + '_E_ave_ve',
                    sc_id + '_jidote') # nW/m3
    
    # Calculate Je.E' and Je.E

    pyspedas.tdotp(sc_id + '_current_fpi_elec', sc_id + '_e_frozen',
                    sc_id + '_jedoteprime') # in nW/m3
    pyspedas.tdotp(sc_id + '_current_fpi_elec', sc_id + '_E_ave_ve',
                    sc_id + '_jedote') # nW/m3
    
    # Find parallel and perpendicular components of J.E'

    vperppara_xyz(sc_id + '_current_fpi', sc_id + '_B_ave_ve', 
                  vpara = sc_id + '_current_fpi_para',
                  vperp_xyz = sc_id + '_current_fpi_perp_xyz')
    vperppara_xyz(sc_id + '_current_fpi_ions', sc_id + '_B_ave_ve',
                  vpara = sc_id + '_current_fpi_ions_para',
                  vperp_xyz = sc_id + '_current_fpi_ions_perp_xyz')
    vperppara_xyz(sc_id + '_current_fpi_elec', sc_id + '_B_ave_ve',
                  vpara = sc_id + '_current_fpi_elec_para',
                  vperp_xyz = sc_id + '_current_fpi_elec_perp_xyz')
    vperppara_xyz(sc_id + '_e_frozen', sc_id + '_B_ave_ve',
                  vpara = sc_id + '_e_frozen_para',
                  vperp_xyz = sc_id + '_e_frozen_perp_xyz')
    
    pyspedas.tdotp(sc_id + '_current_fpi_perp_xyz',
                   sc_id + '_e_frozen_perp_xyz',
                   sc_id + '_jdoteprime_perp') # in nW/m3
    
    t, current_fpi_para = pytplot.get_data(sc_id + '_current_fpi_para')
    t, e_frozen_para = pytplot.get_data(sc_id + '_e_frozen_para')

    jdoteprime_para = current_fpi_para * e_frozen_para # in nW/m3

    pytplot.store_data(sc_id + '_jdoteprime_para', data = {'x': te,
                                                        'y': jdoteprime_para})
    
    pyspedas.tdotp(sc_id + '_current_fpi_ions_perp_xyz',
                   sc_id + '_e_frozen_perp_xyz',
                   sc_id + '_jdoteprime_ions_perp')
    
    t, current_fpi_para = pytplot.get_data(sc_id+'_current_fpi_ions_para')
    t, e_frozen_para = pytplot.get_data(sc_id+'_e_frozen_para')

    jdoteprime_para = current_fpi_para * e_frozen_para # in nW/m3

    pytplot.store_data(sc_id + '_jdoteprime_ions_para', data = {'x': te,
                                                        'y': jdoteprime_para})
    
    pyspedas.tdotp(sc_id + '_current_fpi_elec_perp_xyz',
                    sc_id + '_e_frozen_perp_xyz',
                    sc_id + '_jdoteprime_elec_perp')
    
    t, current_fpi_para = pytplot.get_data(sc_id+'_current_fpi_elec_para')
    t, e_frozen_para = pytplot.get_data(sc_id+'_e_frozen_para')

    jdoteprime_para = current_fpi_para * e_frozen_para # in nW/m3

    pytplot.store_data(sc_id + '_jdoteprime_elec_para', data = {'x': te,
                                                        'y': jdoteprime_para})
    
    # Find parallel and perpendicular components of J.E

    vperppara_xyz(sc_id+'_E_ave_ve', sc_id+'_B_ave_ve',
                  vpara=sc_id+'_E_ave_ve_para',
                  vperp_xyz=sc_id+'_E_ave_ve_perp_xyz')

    pyspedas.tdotp(sc_id+'_current_fpi_perp_xyz', sc_id+'_E_ave_ve_perp_xyz',
                   sc_id+'_jdote_perp') # in nW/m3
    
    t, current_fpi_para = pytplot.get_data(sc_id+'_current_fpi_para')
    t, e_frozen_para = pytplot.get_data(sc_id+'_E_ave_ve_para')

    jdote_para = current_fpi_para * e_frozen_para # in nW/m3

    pytplot.store_data(sc_id + '_jdote_para', data = {'x': te, 'y': jdote_para})

    pyspedas.tdotp(sc_id+'_current_fpi_ions_perp_xyz',
                    sc_id+'_E_ave_ve_perp_xyz',
                    sc_id+'_jidote_ions_perp') # in nW/m3
    
    t, current_fpi_para = pytplot.get_data(sc_id+'_current_fpi_ions_para')
    t, e_frozen_para = pytplot.get_data(sc_id+'_E_ave_ve_para')

    jdote_para = current_fpi_para * e_frozen_para # in nW/m3

    pytplot.store_data(sc_id + '_jidote_ions_para', data = {'x': te,
                                                            'y': jdote_para})
    
    pyspedas.tdotp(sc_id+'_current_fpi_elec_perp_xyz',
                    sc_id+'_E_ave_ve_perp_xyz',
                    sc_id+'_jedote_elec_perp') # in nW/m3
    
    t, current_fpi_para = pytplot.get_data(sc_id+'_current_fpi_elec_para')
    t, e_frozen_para = pytplot.get_data(sc_id+'_E_ave_ve_para')

    jdote_para = current_fpi_para * e_frozen_para # in nW/m3

    pytplot.store_data(sc_id + '_jedote_elec_para', data = {'x': te,
                                                            'y': jdote_para})
    
    return None

def mms_save_vars(probe_id: int, savefiledirec: str) -> None:
    """
    Saves all variables for a given spacecraft to a file in the specified
    directory.

    Args:
        probe_id (int): MMS spacecraft number
        savefiledirec (str): save directory

    Returns:
        None
    """

    probe_id_str = str(probe_id)
    sc_id = "mms" + probe_id_str

    vars = [sc_id+'_Bmag',
            sc_id+'_B',
            sc_id+'_B_des',
            sc_id+'_Ni',
            sc_id+'_Ne',
            sc_id+'_vi',
            sc_id+'_vi_des',
            sc_id+'_ve',
            sc_id+'_current_fpi',
            sc_id+'_Te_perp',
            sc_id+'_Te_para',
            sc_id+'_Ti_perp',
            sc_id+'_Ti_para',
            sc_id+'_E',
            sc_id+'_E_fast',
            sc_id+'_dis_heatq_gse_brst',
            sc_id+'_des_heatq_gse_brst',
            sc_id+'_vexB_efield',
            sc_id+'_e_frozen',
            sc_id+'_jdoteprime',
            sc_id+'_jdote',
            sc_id+'_current_fpi_para',
            sc_id+'_current_fpi_perp_xyz',
            sc_id+'_current_fpi_ions_para',
            sc_id+'_current_fpi_ions_perp_xyz',
            sc_id+'_current_fpi_elec_para',
            sc_id+'_current_fpi_elec_perp_xyz',
            sc_id+'_e_frozen_para',
            sc_id+'_e_frozen_perp_xyz',
            sc_id+'_jdoteprime_perp',
            sc_id+'_jdoteprime_para',
            sc_id+'_jdoteprime_ions_perp',
            sc_id+'_jdoteprime_ions_para',
            sc_id+'_jdoteprime_elec_perp',
            sc_id+'_jdoteprime_elec_para',
            sc_id+'_E_ave_ve_para',
            sc_id+'_E_ave_ve_perp_xyz',
            sc_id+'_jdote_perp',
            sc_id+'_jdote_para',
            sc_id+'_jdote_perp',
            sc_id+'_jidote_ions_para',
            sc_id+'_jidote_ions_perp',
            sc_id+'_jedote_elec_perp',
            sc_id+'_jedote_elec_para',
            sc_id+'_dis_prestensor_gse_brst',
            sc_id+'_des_prestensor_gse_brst',
            sc_id+'_dis_energyspectr_omni_brst',
            sc_id+'_des_energyspectr_omni_brst',
            sc_id+'_des_energyspectr_par_brst',
            sc_id+'_des_energyspectr_perp_brst',
            sc_id+'_des_energyspectr_anti_brst'
            ]
    
    if not os.path.exists(savefiledirec):
        os.makedirs(savefiledirec)
    
    pytplot.tplot_save(vars,
                       filename = savefiledirec+'/mms'+probe_id_str+'_vars')

    return None