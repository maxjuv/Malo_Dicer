from configuration import *
from select_mice_cata_Malo import get_mice
import scipy.signal
p1 = '/Volumes/crnldata/cmo/scripts/timefreqtools'
sys.path = [p1] + sys.path
import timefreqtools
local_path = os.path.dirname(os.path.realpath(__file__))
print(local_path)

def timefreq_mouse_state(mouse, state):
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600
    # score = ds['score'].values
    dirname = precompute_dir + '/tdw_score/'
    ds_tdw = xr.open_dataset(dirname + 'tdw_score_{}.nc'.format(mouse))
    score = ds_tdw['new_score'].values

    ds_raw = xr.open_dataset(precompute_dir + '/raw/raw_{}.nc'.format(mouse))
    raw = ds_raw['signal'].values.astype('float32')
    sr = ds_raw['sampling_rate'].values

    print('here')
    wt, wt_times, wt_freqs, tfr_sampling_rate = timefreqtools.compute_timefreq(raw, sr,
                0.5, 20,
                delta_freq=.2, f0=3,  normalisation=0,  min_sampling_rate=None,
                returns='all', t_start=0, zero_pad=True, joblib_memory=None)
    wt = np.abs(wt)
    print('hre')
    timefreqtools.plot_tfr(wt, wt_times, wt_freqs, colorbar = True,
            cax =None, orientation='vertical', clim = None)
    plt.show()

if __name__ == '__main__':
    # mouse = 'B3512'
    # mouse = 'B2534'
    # mouse = 'B2533'
    # mouse = 'B2767'
    mouse = 'B2761'
    state ='w'
    timefreq_mouse_state(mouse= mouse, state = state)
    plt.show()
