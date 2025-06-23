import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspedas
import pytplot

from mms_data_load import mms_load_brst, mms_currents, mms_save_vars
from mms_event_scale import mms_tail_scale
from mms_rnn import mms_rnn_data, mms_rnn_label

events = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'O', 'P', 'Q',
          'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

B0 = [21.27865148, 26.53054016, 16.14916568, 19.05825971, 7.805345825,
      16.38363369, 17.54366084, 22.2842775, 15.41015406, 14.32471067,
      17.15643793, 15.75670785, 16.11432974, 29.09204865, 26.14315926,
      10.23943403, 32.36794703, 13.37737541, 14.44350423, 13.92892911,
      20.14111812, 9.251656107, 18.31873051]

n0 = [0.13838144, 1.0288908, 0.4389502, 0.19944784, 0.19892825, 0.17318012,
      0.20273407, 0.16811092, 0.11836687, 0.38465217, 1.3245863, 0.21070065,
      0.15921502, 2.759707, 0.41606516, 0.20430537, 0.71044284, 0.12632449,
      0.2835195, 0.2835195, 0.18773806, 0.12356131, 0.28314748]

tstart = ['2017-05-28T03:58:30', '2017-07-03T05:27:08', '2017-07-06T15:35:10',
          '2017-07-06T15:46:35', '2017-07-11T22:33:30', '2017-07-17T07:48:45',
          '2017-07-26T00:04:25', '2017-07-26T07:00:00', '2017-07-26T07:28:30',
          '2017-08-06T05:13:30', '2017-08-23T17:53:30', '2018-08-27T11:41:00',
          '2018-08-27T12:15:30', '2018-09-10T17:14:45', '2018-09-11T00:00:45',
          '2019-07-25T21:41:10', '2019-08-31T12:03:00', '2019-09-06T04:37:30',
          '2020-08-02T16:58:20', '2020-08-02T17:10:00', '2020-08-03T01:05:40',
          '2020-08-05T14:18:30', '2020-08-29T09:55:00']

tend = ['2017-05-28T03:59:15', '2017-07-03T05:27:30', '2017-07-06T15:36:00',
        '2017-07-06T15:46:52', '2017-07-11T22:34:15', '2017-07-17T07:49:30',
        '2017-07-26T00:04:35', '2017-07-26T07:00:45', '2017-07-26T07:29:15',
        '2017-08-06T05:14:00', '2017-08-23T17:54:15', '2018-08-27T11:41:45',
        '2018-08-27T12:16:00', '2018-09-10T17:15:30', '2018-09-11T00:01:15',
        '2019-07-25T21:41:55', '2019-08-31T12:03:30', '2019-09-06T04:41:00',
        '2020-08-02T16:59:30', '2020-08-02T17:11:20', '2020-08-03T01:06:20',
        '2020-08-05T14:21:00', '2020-08-29T09:58:00']

nums = np.array([0, 2, 3, 4, 5, 12, 13, 15, 16, 20])

events = [events[i] for i in nums]
B0 = [B0[i] for i in nums]
n0 = [n0[i] for i in nums]
tstart = [tstart[i] for i in nums]
tend = [tend[i] for i in nums]

for i, event in enumerate(events):
    probe = 4 # change for each spacecraft

    mms_load_brst(probe_id=probe, t_start=tstart[i], t_end=tend[i])
    mms_currents(probe_id=probe)
    mms_save_vars(probe_id=probe,
                  savefiledirec='data/' + event + '_mms' + str(probe))
    
    pytplot.tplot_restore('data/' + event + '_mms' +
                          str(probe) + '/mms' + str(probe) + '_vars')
    
    mms_tail_scale(probe_id=probe, n0=n0[i], B0=B0[i],
                   savefiledirec='data/' + event + '_mms' + str(probe))#
    
    pytplot.tplot_restore('data/' + event + '_mms' + str(probe) + '/mms' +
                          str(probe) + '_vars_norm')
    
    mms_rnn_data(probe, 'data/' + event + '_mms' + str(probe))
    mms_rnn_label(probe, 'data/' + event + '_mms' + str(probe))