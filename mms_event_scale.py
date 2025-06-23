"""
    Normalises MMS data for an event to make it comparable between events.

    Author: Cara Waters
    07/11/2024

"""


import numpy as np
import pandas as pd
import pyspedas
import pytplot
import os


# Define colours
red = (213, 94, 0)
green = (0, 158, 115)
blue = (86, 180, 233)

def mms_tail_scale(probe_id: int, n0: float, B0: float,
                   savefiledirec: str) -> None:
    """
    Takes the lobe field and current sheet density and normalises data for an
    event to make it comparable between events.

    Args:
        probe_id (int): MMS spacecraft ID
        n0 (float): current sheet density (cm-3)
        B0 (float): lobe field (reversing field) (nT)
        savefiledirec (str): save file directory

    Returns:
        None
    """

    sc_id = 'mms' + str(probe_id)

    # --------------------------------------------------------------------------
    # Number densities
    # --------------------------------------------------------------------------

    ti, ni = pytplot.get_data(sc_id + '_Ni')
    te, ne = pytplot.get_data(sc_id + '_Ne')

    # Normalise number densities
    ni_norm = ni / n0
    ne_norm = ne / n0

    pytplot.store_data(sc_id + '_Ni_norm', data={'x': ti, 'y': ni_norm})
    pytplot.store_data(sc_id + '_Ne_norm', data={'x': te, 'y': ne_norm})

    # --------------------------------------------------------------------------
    # Magnetic field
    # --------------------------------------------------------------------------

    pyspedas.cotrans(sc_id + '_B', sc_id + '_B',
                     coord_in='gse', coord_out='gsm')

    tb, B = pytplot.get_data(sc_id + '_B')
    tbmag, Bmag = pytplot.get_data(sc_id + '_Bmag')

    # Normalise magnetic field
    B_norm = B / B0
    Bmag_norm = Bmag / B0

    pytplot.store_data(sc_id + '_B_norm', data={'x': tb, 'y': B_norm})
    pytplot.store_data(sc_id + '_Bmag_norm', data={'x': tbmag, 'y': Bmag_norm})

    # --------------------------------------------------------------------------
    # Velocity
    # --------------------------------------------------------------------------

    v_factor = B0 * 1e-9 / np.sqrt(n0 * 1e6 * 1.6726219e-27 * 4 * np.pi * 1e-7)
    v_factor = v_factor / 1e3 # in km/s

    pyspedas.cotrans(sc_id + '_vi', sc_id + '_vi',
                     coord_in='gse', coord_out='gsm')
    pyspedas.cotrans(sc_id + '_ve', sc_id + '_ve',
                     coord_in='gse', coord_out='gsm')

    ti, vi = pytplot.get_data(sc_id + '_vi')
    te, ve = pytplot.get_data(sc_id + '_ve')

    # Normalise velocity
    vi_norm = vi / v_factor
    ve_norm = ve / v_factor

    pytplot.store_data(sc_id + '_vi_norm', data={'x': ti, 'y': vi_norm})
    pytplot.store_data(sc_id + '_ve_norm', data={'x': te, 'y': ve_norm})

    # --------------------------------------------------------------------------
    # Electric field
    # --------------------------------------------------------------------------

    e_factor = 1 / (B0 * 1e-9 * v_factor * 1e3)
    e_factor = e_factor / 1e3 # in mV/m

    pyspedas.cotrans(sc_id + '_E', sc_id + '_E',
                     coord_in='gse', coord_out='gsm')

    te, E = pytplot.get_data(sc_id + '_E')
    
    # Normalise electric field
    E_norm = E * e_factor

    pytplot.store_data(sc_id + '_E_norm', data={'x': te, 'y': E_norm})

    # --------------------------------------------------------------------------
    # Save data
    # --------------------------------------------------------------------------

    vars = [sc_id + '_Ni_norm', sc_id + '_Ne_norm', sc_id + '_B_norm',
            sc_id + '_Bmag_norm', sc_id + '_vi_norm', sc_id + '_ve_norm',
            sc_id + '_E_norm']
    
    if not os.path.exists(savefiledirec):
        os.makedirs(savefiledirec)
    
    pytplot.tplot_save(vars,
                    filename = savefiledirec+'/mms'+str(probe_id)+'_vars_norm')

    return None