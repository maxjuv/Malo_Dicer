from configuration import *
from select_mice_cata_Malo import get_mice
from power_spectrum_Malo import get_clear_spectral_score_one_mouse_one_condition_day_night
import tensorpac
import scipy.signal

def export_tdw_socring_for_moji():

    dcr_mice = get_mice(group = 'DCR-HCRT')
    control_mice = get_mice(group = 'Control')
    mice = dcr_mice+control_mice
    dirname = precompute_dir + '/tdw_score/'
    output_path = work_dir + '/scoring_with_tdw_moji/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for mouse in mice :
        ds_tdw = xr.open_dataset(dirname + 'tdw_score_{}.nc'.format(mouse))
        score = list(ds_tdw['new_score'].values)

        np.savetxt(output_path +mouse+".txt",score ,fmt="%s")


def one_mouse_one_day_theta_gamma(mouse, session, state, period):

    ds = xr.open_dataset(precompute_dir + '/raw/raw_{}.nc'.format(mouse))
    raw = ds['signal'].values.astype('float32')
    sr = ds['sampling_rate'].values
    #
    # times = ds.coords['times_second'].values
    times = np.arange(raw.size)/(sr*3600)

    windows = 4   #### in s
    n_epochs = int(raw.size//(windows*sr))
    epochs = np.arange(n_epochs)
    sample_per_epoch = int(windows*sr)
    mylist = []
    index = []
    for i in range(n_epochs):
        mylist.append(4*i*sr)
    real_sample_by_epoch= np.diff(np.array(mylist, dtype='int'))
    ind = 0
    for fr in real_sample_by_epoch:
        fr = int(fr)
        ind += fr
        if fr == 799:
            index.append(ind)
    index = np.array(index, dtype ='int')+1
    ref = 69120000
    if raw.size + index.size != ref:
        index = index[:-(raw.size + index.size-ref)]
    raw = np.insert(raw, index, raw[index])
    point_per_epochs = 800
    stacked_sigs = raw.reshape((-1, point_per_epochs)).astype('float32')


    score_no_transition, mask = get_clear_spectral_score_one_mouse_one_condition_day_night(mouse, session, state, period)

    session_data = stacked_sigs[mask]
    selected_data = session_data[score_no_transition]

    freqs_welch, welch = scipy.signal.welch(stacked_sigs, fs = sr, nperseg = int(3.99*sr) )
    fig,ax = plt.subplots()
    ax.plot(freqs_welch, np.mean(welch,axis = 0))
    fig,ax = plt.subplots()
    # p = tensorpac.Pac(idpac=(6, 0, 0), f_pha='hres', f_amp='hres')
    p = tensorpac.Pac(idpac=(2, 0, 0), f_pha=(6, 12, 1, .2), f_amp=(30, 100, 2, 1))

    # Filter the data and extract pac
    xpac = p.filterfit(float(sr), selected_data)
    # xpac = p.filterfit(float(sr), selected_data)

    # plot your Phase-Amplitude Coupling :
    p.comodulogram(xpac.mean(-1), cmap='Spectral_r', plotas='contour', ncontours=5,
                   title=r'10hz phase$\Leftrightarrow$100Hz amplitude coupling',
                   fz_title=14, fz_labels=13)
    p.show()



if __name__ == '__main__':
    # export_tdw_socring_for_moji()
    # mouse = 'B2767'
    mouse = 'B2700'

    # mouse = 'B2763'
    session = 'bl1'
    state = 'w'
    period = 'dark'
    one_mouse_one_day_theta_gamma(mouse, session, state, period)
