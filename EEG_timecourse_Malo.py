from configuration import *
from select_mice_cata_Malo import get_mice
import multiprocessing
from joblib import Parallel, delayed
from sklearn.metrics import auc
import scipy.signal
import scipy.stats

local_path = os.path.dirname(os.path.realpath(__file__))
print(local_path)

def wake_bandwidth_power_percentile(bandwidth = [6,9],state = 'w', group ='Control', spectrum_method = 'somno', auc = False, agg_power='mean', band_ref = 'total'):
    debug =0
    # state = 'w'
    if not debug:
        dirname = excel_dir + '{}/EEG_power_time_course/auc_{}_agg_power_{}_bandref_{}/{}/{}hz_to_{}hz_power/'.format(group,auc,agg_power,band_ref, state,  bandwidth[0], bandwidth[1])
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    if debug:
        fig, ax = plt.subplots()
        fig.suptitle(spectrum_method)
    high = 1
    final_ZT = np.array([])
    final_delta = np.array([])
    n_loop = 0
    mice = get_mice(group)
    df_power = pd.DataFrame(index = mice, columns = np.arange(77))
    df_times = pd.DataFrame(index = mice, columns = np.arange(77))
    for mouse in mice:
        print(mouse)
        if mouse == 'B2767':
            continue
        # if mouse != 'B4977':
        #     continue
        n_loop +=1
        if bandwidth[0]>50:
            ds = xr.open_dataset(precompute_dir + '/spectrums/HF_spectrum_scoring_{}.nc'.format(mouse))
            score = ds['score'].values
            times = ds.coords['times_somno'].values/3600
            freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
        else :
            ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
            score = ds['score'].values
            times = ds.coords['times_somno'].values/3600
            freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
        mask_bandwidth = (freqs>bandwidth[0]) & (freqs<bandwidth[1])
        df = (freqs[11]-freqs[1])/10
        spectrum = ds['{}_spectrum'.format(spectrum_method)].values

        tdw_dirname = precompute_dir + '/tdw_score/'
        ds_tdw = xr.open_dataset(tdw_dirname + 'tdw_score_{}.nc'.format(mouse))
        score = ds_tdw['new_score'].values

        if state == 'w':
            score_no_artifact = (score == 'w') | (score =='t')
        else :
            score_no_artifact = score == state

        score_no_transition = score_no_artifact.copy()
        count = 0
        event_length = []
        for num, value in enumerate(score_no_artifact) :
            if value:
                count +=1
            elif value == False:
                event_length.append(count)
                if count == 1 :
                    score_no_transition[num-1] = False
                elif count == 2 :
                    score_no_transition[num-1] = False
                    score_no_transition[num-2] = False
                    count = 0
                elif count > 2 :
                    score_no_transition[num-1] = False
                    score_no_transition[num-count] = False
                count = 0
        if count == 1 :
            score_no_transition[num-1] = False
        elif count == 2 :
            score_no_transition[num-1] = False
            score_no_transition[num-2] = False
            count = 0
        elif count > 2 :
            score_no_transition[num-1] = False
            score_no_transition[num-count] = False
###############
        artefact_tresh = np.median(np.mean(spectrum[score_no_transition],axis = 1)) + 10*scipy.stats.median_absolute_deviation(np.mean(spectrum[score_no_transition],axis = 1))   #####34???
        arte = np.mean(spectrum[score_no_transition],axis = 1)>artefact_tresh
        index = np.arange(score_no_transition.size)
        index = index[score_no_transition][arte]
        # fig, ax = plt.subplots()
        # ax.plot(np.mean(spectrum[score_no_transition],axis = 1))
        # ax.axhline(artefact_tresh)
        # plt.show()

        if index.size!=0:
            for i in [index]:
        #         if value :
        #             corr[i]=False
                score_no_transition[i]=False
##############
        mask_ref = ((times>8) & (times<12)) | ((times>32) & (times<36)) ####derived from sleep
        # mask_ref = ((times>14) & (times<18)) | ((times>24+14) & (times<24+18))
        # mask_ref = ((times>12) & (times<24)) | ((times>36) & (times<48))
        # mask_ref = ((times>0) & (times<48))
        score_ref = score_no_transition[mask_ref]
        spec_ref = spectrum[mask_ref]
        if band_ref == 'total':
            mask_bandwidth_ref = np.ones(freqs.size).astype('bool')
        elif band_ref == 'band':
            mask_bandwidth_ref = mask_bandwidth

        if agg_power == 'mean':
            ref = np.sum(np.mean(spec_ref[score_ref], axis = 0)[mask_bandwidth_ref])
        elif agg_power == 'median':
            ref = np.sum(np.median(spec_ref[score_ref], axis = 0)[mask_bandwidth_ref])
        if auc == True:
            ref *= df
        # ref = 1
        # BLA_ref = np.mean(spec_ref[score_ref], axis = 0)
        # BLA = []
        time_score = times[score_no_transition]
        percentile = []

        # mask = (time_score<12)
        # p = np.linspace(0,100,12+1)
        # percentile.append(np.percentile(time_score[mask], p))
        # mask = (time_score>12)&(time_score<24)
        # p = np.linspace(0,100,6+1)
        # percentile.append(np.percentile(time_score[mask], p))
        # mask = (time_score>24)&(time_score<36)
        # p = np.linspace(0,100,12+1)
        # percentile.append(np.percentile(time_score[mask], p))
        # mask = (time_score>36)&(time_score<48)
        # p = np.linspace(0,100,6+1)
        # percentile.append(np.percentile(time_score[mask], p))
        # mask = (time_score>48)&(time_score<60)
        # p = np.linspace(0,100,12+1)
        # percentile.append(np.percentile(time_score[mask], p))
        # mask = (time_score>60)&(time_score<72)
        # p = np.linspace(0,100,6+1)
        # percentile.append(np.percentile(time_score[mask], p))
        # mask = (time_score>72)&(time_score<84)
        # p = np.linspace(0,100,12+1)
        # percentile.append(np.percentile(time_score[mask], p))
        # mask = (time_score>84)&(time_score<96)
        # p = np.linspace(0,100,6+1)
        # percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score<12)
        if np.sum(mask)==0:
            percentile.append(np.ones(6+1)*-1)
        else :
            p = np.linspace(0,100,6+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>12)&(time_score<24)
        if np.sum(mask)==0:
            percentile.append(np.ones(12+1)*-1)
        else :
            p = np.linspace(0,100,12+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>24)&(time_score<36)
        if np.sum(mask)==0:
            percentile.append(np.ones(6+1)*-1)
        else :
            p = np.linspace(0,100,6+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>36)&(time_score<48)
        if np.sum(mask)==0:
            percentile.append(np.ones(12+1)*-1)
        else :
            p = np.linspace(0,100,12+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>48)&(time_score<54)
        if np.sum(mask)==0:
            percentile.append(np.ones(8+1)*-1)
        else :
            p = np.linspace(0,100,8+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>54)&(time_score<60)
        if np.sum(mask)==0:
            percentile.append(np.ones(3+1)*-1)
        else :
            p = np.linspace(0,100,3+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>60)&(time_score<72)
        if np.sum(mask)==0:
            percentile.append(np.ones(12+1)*-1)
        else :
            p = np.linspace(0,100,12+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>72)&(time_score<84)
        if np.sum(mask)==0:
            percentile.append(np.ones(6+1)*-1)
        else :
            p = np.linspace(0,100,6+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>84)&(time_score<96)
        if np.sum(mask)==0:
            percentile.append(np.ones(12+1)*-1)
        else :
            p = np.linspace(0,100,12+1)
            percentile.append(np.percentile(time_score[mask], p))


        percentile = np.concatenate(percentile)
        absolute_delta = []
        ZT = []
        # exceptions = [12, 19, 32, 39, 52, 59, 72]
        exceptions = [6,19,26, 39, 48, 52,65, 72]
        for i in range(percentile.size-1):
            if i in exceptions :
                continue
            else :
                t1, t2 = percentile[i], percentile[i+1]
                # print(t1,t2)
                if t1 == -1:
                    absolute_delta.append(np.nan)
                    ZT.append(np.nan)
                else :
                    mask = (times>t1) & (times<t2)
                    interval_score =score_no_transition[mask]
                    interval_spectrum = spectrum[mask]
                    # print(i)
                    # print('         ', interval_spectrum[interval_score].shape[0])
                    if agg_power == 'median':
                        mean_spec = np.median(interval_spectrum[interval_score], axis = 0)
                    elif agg_power == 'mean':
                        mean_spec = np.mean(interval_spectrum[interval_score], axis = 0)
                        # BLA.append(mean_spec)

                    if i == 0:
                        MM = mean_spec
                    else :
                        MM = np.vstack((MM, mean_spec))
                    if auc :
                        absolute_delta.append(np.sum(mean_spec[mask_bandwidth])*df)
                    elif auc ==False :
                        absolute_delta.append(np.sum(mean_spec[mask_bandwidth]))
                    ZT.append((t1+t2)/2)
        # fig, ax =plt.subplots()
        # ax.plot(BLA_ref)
        # BLA = np.array(BLA)
        # ax.plot(BLA.T, alpha = .1)
        # plt.show()
        ZT = np.array(ZT)
        absolute_delta = np.array(absolute_delta)
        relative_delta = 100*absolute_delta/ref
        df_power.loc[mouse, :] = relative_delta
        df_times.loc[mouse, :] = ZT

        final_ZT = np.append(final_ZT, ZT)
        final_delta = np.append(final_delta, relative_delta)

        if debug:
            # ax.plot(ZT, relative_delta, label =mouse)
            ax.plot(ZT, relative_delta, color = 'black', alpha = .1)

    points = int(final_ZT.size/n_loop)
    final_ZT = final_ZT.reshape(n_loop, points)
    final_delta = final_delta.reshape(n_loop, points)
    # print(df_power)
    # print(df_times)

    # final_ZT = np.median(final_ZT, axis = 0)
    # final_delta = np.median(final_delta, axis = 0)
    if not debug:
        df_power.to_excel(dirname + '{}_{}hz_to_{}hz_EEG_power_{}.xlsx'.format(group, bandwidth[0], bandwidth[1], spectrum_method))
        df_times.to_excel(dirname + '{}_{}hz_to_{}hz_times_{}.xlsx'.format(group, bandwidth[0], bandwidth[1], spectrum_method))
    if debug:
        ax.plot(final_ZT, final_delta)
        ax.legend()
        # plt.show()
    return final_ZT, final_delta

def plot_compare_wake(bandwidth = [5,10], state = 'w', spectrum_method = 'somno', agg_mice='mean', auc=False, agg_power = 'mean', band_ref = 'total'):
    fig, ax = plt.subplots(figsize = (10,5))
    group_color = {'Control':'black', 'DCR-HCRT':'seagreen'}
    style = {'Control' : 'o', 'DCR-HCRT':'square'}
    stat = []
    time = []
    for group in ['Control', 'DCR-HCRT']:
        final_ZT, final_delta = wake_bandwidth_power_percentile(bandwidth = bandwidth,state = state, group =group, spectrum_method = spectrum_method, auc = auc, agg_power = agg_power, band_ref = band_ref)
        stat.append(final_delta)
        if agg_mice == 'median':
            mad = scipy.stats.median_absolute_deviation(final_delta, axis=0)
            final_ZT = np.median(final_ZT, axis = 0)
            final_delta = np.median(final_delta, axis = 0)
            ax.errorbar(final_ZT, final_delta, yerr=mad, color = group_color[group])

            # time.append(final_ZT)
        elif agg_mice == 'mean':
            sem = scipy.stats.sem(final_delta, axis=0)
            final_ZT = np.mean(final_ZT, axis = 0)
            final_delta = np.mean(final_delta, axis = 0)
            ax.errorbar(final_ZT, final_delta, yerr=sem, color = group_color[group])

        time.append(final_ZT)
        # ax.plot(final_ZT, final_delta,label = group)
        ax.plot(final_ZT, final_delta, color = group_color[group], lw = 1)
        # ax.errorbar(final_ZT, final_delta, yerr=mad, color = group_color[group])
        ax.scatter(final_ZT, final_delta, color = group_color[group], edgecolor = 'black', label = group)
    stars = []
    if agg_mice == 'median':
        time_course = np.median(np.array(time), axis = 0)
    elif agg_mice == 'mean':
        time_course = np.mean(np.array(time), axis = 0)
    ymin, _ = ax.get_ylim()
    for i in range(final_delta.size):
        if agg_mice == 'median':
            test = 'kruskal'
            statitci, pvalue = scipy.stats.kruskal(stat[0][:,i], stat[1][:,i])

        elif agg_mice == 'mean':
            test = 'f_oneway'
            statistic, pvalue = scipy.stats.f_oneway(stat[0][:,i], stat[1][:,i])
        print(pvalue)

        if pvalue < .05:
            ax.text(time_course[i], 1.01*ymin, '*', fontsize = 15)
        elif pvalue < .01:
            ax.text(time_course[i], 1.01*ymin, '*\n*', fontsize = 15)
        # elif pvalue < .1:
        #     ax.text(time_course[i], 80, 'a')

    fig.suptitle('bandwidth = {}, state = {}, spectrum = {}, agg = {}, band_ref = {}'.format(bandwidth, state, spectrum_method, agg_mice, band_ref))
    # plt.rcParams.update({'font.size': 40})
    ax.set_xticks(np.arange(16)*6)
    ax.set_xticklabels(np.arange(16)*6%24)
    # plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=40)
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.legend(frameon = False, fontsize = 10, loc =2)
    # ax.set_ylim(65, 220)
    # plt.show()
    # exit()
    # fig.suptitle('w {} {}hz to {}hz power'.format(spectrum_method, bandwidth[0], bandwidth[1]))
    dirname = excel_dir + 'Control/EEG_power_time_course/auc_{}_agg_power_{}_bandref_{}/{}/'.format(auc, agg_power, band_ref,state)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(dirname + '{}_{}_{}hz_to_{}hz_power_{}.png'.format(state,spectrum_method, bandwidth[0], bandwidth[1], agg_mice), dpi = 1200)
    dirname = excel_dir + 'DCR-HCRT/EEG_power_time_course/auc_{}_agg_power_{}_bandref_{}/{}/'.format(auc, agg_power, band_ref,state)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(dirname + '{}_{}_{}hz_to_{}hz_power_{}.png'.format(state,spectrum_method, bandwidth[0], bandwidth[1], agg_mice), dpi = 1200)
    # plt.show()


def sleep_bandwidth_power_percentile(bandwidth = [.75,4], group ='Control', state = 'n', spectrum_method = 'somno', auc = False, agg_power='mean',band_ref = 'total'):
    debug =0
    # state = 'n'
    if debug:
        fig, ax = plt.subplots()
        fig.suptitle(spectrum_method)
    if not debug:
        dirname = excel_dir + '{}/EEG_power_time_course/auc_{}_agg_power_{}_bandref_{}/{}/{}hz_to_{}hz_power/'.format(group,auc, agg_power,band_ref, state,  bandwidth[0], bandwidth[1])
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    high = 1
    final_ZT = np.array([])
    final_delta = np.array([])
    n_loop = 0
    mice = get_mice(group)
    df_power = pd.DataFrame(index = mice, columns = np.arange(66))
    df_times = pd.DataFrame(index = mice, columns = np.arange(66))
    for mouse in mice:
        print(mouse)
        if mouse == 'B2767':
            continue
        # if mouse != 'B4977':
        #     continue
        n_loop +=1
        if bandwidth[0]>50:
            ds = xr.open_dataset(precompute_dir + '/spectrums/HF_spectrum_scoring_{}.nc'.format(mouse))
            score = ds['score'].values
            times = ds.coords['times_somno'].values/3600
            freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
        else :
            ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
            score = ds['score'].values
            times = ds.coords['times_somno'].values/3600
            freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
        mask_bandwidth = (freqs>bandwidth[0]) & (freqs<bandwidth[1])
        df = (freqs[11]-freqs[1])/10
        spectrum = ds['{}_spectrum'.format(spectrum_method)].values
        score_no_artifact = score == state
        score_no_transition = score_no_artifact.copy()
        count = 0
        event_length = []
        for num, value in enumerate(score_no_artifact) :
            if value:
                count +=1
            elif value == False:
                event_length.append(count)
                if count == 1 :
                    score_no_transition[num-1] = False
                elif count == 2 :
                    score_no_transition[num-1] = False
                    score_no_transition[num-2] = False
                    count = 0
                elif count > 2 :
                    score_no_transition[num-1] = False
                    score_no_transition[num-count] = False
                count = 0
        if count == 1 :
            score_no_transition[num-1] = False
        elif count == 2 :
            score_no_transition[num-1] = False
            score_no_transition[num-2] = False
            count = 0
        elif count > 2 :
            score_no_transition[num-1] = False
            score_no_transition[num-count] = False
#################
        artefact_tresh = np.median(np.mean(spectrum[score_no_transition],axis = 1)) + 10*scipy.stats.median_absolute_deviation(np.mean(spectrum[score_no_transition],axis = 1))   #####34???
        arte = np.mean(spectrum[score_no_transition],axis = 1)>artefact_tresh
        index = np.arange(score_no_transition.size)
        index = index[score_no_transition][arte]
        # fig, ax = plt.subplots()
        # ax.plot(np.mean(spectrum[score_no_transition],axis = 1))
        # ax.axhline(artefact_tresh)
        # plt.show()

        if index.size!=0:
            for i in [index]:
        #         if value :
        #             corr[i]=False
                score_no_transition[i]=False
####################



        mask_ref = ((times>8) & (times<12)) | ((times>32) & (times<36))
        # mask_ref = ((times>0) & (times<12)) | ((times>24) & (times<36))
        # mask_ref = ((times>0) & (times<48))
        score_ref = score_no_transition[mask_ref]
        spec_ref = spectrum[mask_ref]

        if band_ref == 'total':
            mask_bandwidth_ref = np.ones(freqs.size).astype('bool')
        elif band_ref == 'band':
            mask_bandwidth_ref = mask_bandwidth


        # ref = np.sum(np.median(spec_ref[score_ref], axis = 0)[mask_delta])*df
        if agg_power == 'mean':
            ref = np.sum(np.mean(spec_ref[score_ref], axis = 0)[mask_bandwidth_ref])
        elif agg_power == 'median':
            ref = np.sum(np.median(spec_ref[score_ref], axis = 0)[mask_bandwidth_ref])
        if auc == True:
            ref *= df
        # BLA_ref = np.mean(spec_ref[score_ref], axis = 0)
        # BLA = []
        time_score = times[score_no_transition]
        percentile = []

        mask = (time_score<12)
        if np.sum(mask)==0:
            percentile.append(np.ones(12+1)*-1)
        else:
            p = np.linspace(0,100,12+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>12)&(time_score<24)
        if np.sum(mask)==0:
            percentile.append(np.ones(6+1)*-1)
        else:
            p = np.linspace(0,100,6+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>24)&(time_score<36)
        if np.sum(mask)==0:
            percentile.append(np.ones(12+1)*-1)
        else:
            p = np.linspace(0,100,12+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>36)&(time_score<48)
        if np.sum(mask)==0:
            percentile.append(np.ones(6+1)*-1)
        else:
            p = np.linspace(0,100,6+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>54)&(time_score<60)
        if np.sum(mask)==0:
            percentile.append(np.ones(6+1)*-1)
        else:
            p = np.linspace(0,100,6+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>60)&(time_score<72)
        if np.sum(mask)==0:
            percentile.append(np.ones(6+1)*-1)
        else:
            p = np.linspace(0,100,6+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>72)&(time_score<84)
        if np.sum(mask)==0:
            percentile.append(np.ones(12+1)*-1)
        else:
            p = np.linspace(0,100,12+1)
            percentile.append(np.percentile(time_score[mask], p))

        mask = (time_score>84)&(time_score<96)
        if np.sum(mask)==0:
            percentile.append(np.ones(6+1)*-1)
        else:
            p = np.linspace(0,100,6+1)
            percentile.append(np.percentile(time_score[mask], p))



        percentile = np.concatenate(percentile)
        absolute_delta = []
        ZT = []
        # exceptions = [12, 19, 32, 39, 48, 55, 68 ]
        exceptions = [12, 19, 32, 39, 46, 53, 66]
        for i in range(percentile.size-1):
            if i in exceptions :
                continue

            else :
                t1, t2 = percentile[i], percentile[i+1]
                if t1 == - 1:
                    ZT.append(np.nan)
                    absolute_delta.append(np.nan)
                else :
                    mask = (times>t1) & (times<t2)
                    interval_score =score_no_transition[mask]
                    interval_spectrum = spectrum[mask]
                    # print(i)
                    # print('         ', interval_spectrum[interval_score].shape[0])
                    if agg_power == 'median':
                        mean_spec = np.median(interval_spectrum[interval_score], axis = 0)
                    elif agg_power == 'mean':
                        mean_spec = np.mean(interval_spectrum[interval_score], axis = 0)
                    # BLA.append(mean_spec)
                    if i == 0:
                        MM = mean_spec
                    else :
                        MM = np.vstack((MM, mean_spec))
                    if auc:
                        absolute_delta.append(np.sum(mean_spec[mask_bandwidth])*df)
                    elif auc ==False:
                        absolute_delta.append(np.sum(mean_spec[mask_bandwidth]))
                    ZT.append((t1+t2)/2)
        ZT = np.array(ZT)
        absolute_delta = np.array(absolute_delta)
        relative_delta = 100*absolute_delta/ref
        # relative_delta = absolute_delta
        df_power.loc[mouse, :] = relative_delta
        df_times.loc[mouse, :] = ZT
        final_ZT = np.append(final_ZT, ZT)
        final_delta = np.append(final_delta, relative_delta)

        if debug:
            fig, ax = plt.subplots()
        # ax.plot(ZT, relative_delta, label =mouse)
            ax.plot(ZT, relative_delta, color = 'black', alpha = .1)
            plt.show()

    points = int(final_ZT.size/n_loop)
    final_ZT = final_ZT.reshape(n_loop, points)
    final_delta = final_delta.reshape(n_loop, points)

    # final_ZT = np.median(final_ZT, axis = 0)
    # final_delta = np.median(final_delta, axis = 0)


    if not debug:
        df_power.to_excel(dirname + '{}_{}hz_to_{}hz_EEG_power_{}.xlsx'.format(group, bandwidth[0], bandwidth[1], spectrum_method))
        df_times.to_excel(dirname + '{}_{}hz_to_{}hz_times_{}.xlsx'.format(group, bandwidth[0], bandwidth[1], spectrum_method))

    if debug:
        ax.plot(final_ZT, final_delta)
        # ax.axhline(ref)
        ax.legend()
        plt.show()
    return  final_ZT, final_delta

def plot_compare_sleep(bandwidth = [.75, 4], state = 'n', spectrum_method = 'somno', agg_mice='mean', auc = False, agg_power='mean', band_ref='total'):
    fig, ax = plt.subplots(figsize = (10, 5))
    group_color = {'Control':'black', 'DCR-HCRT':'seagreen'}
    style = {'Control' : 'o', 'DCR-HCRT':'square'}
    stat = []
    time = []
    for group in ['Control', 'DCR-HCRT']:
        final_ZT, final_delta = sleep_bandwidth_power_percentile(bandwidth = bandwidth,group =group, state = state, spectrum_method = spectrum_method, auc = auc, agg_power = agg_power, band_ref = band_ref)
        stat.append(final_delta)
        if agg_mice == 'median':
            mad = scipy.stats.median_absolute_deviation(final_delta, axis=0)
            final_ZT = np.median(final_ZT, axis = 0)
            final_delta = np.median(final_delta, axis = 0)
            ax.errorbar(final_ZT, final_delta, yerr=mad, color = group_color[group])

            # time.append(final_ZT)
        elif agg_mice == 'mean':
            sem = scipy.stats.sem(final_delta, axis=0)
            final_ZT = np.mean(final_ZT, axis = 0)
            final_delta = np.mean(final_delta, axis = 0)
            ax.errorbar(final_ZT, final_delta, yerr=sem, color = group_color[group])

        time.append(final_ZT)
        # ax.plot(final_ZT, final_delta,label = group)
        ax.plot(final_ZT, final_delta, color = group_color[group], lw = 1)
        # ax.errorbar(final_ZT, final_delta, yerr=mad, color = group_color[group])
        ax.scatter(final_ZT, final_delta, color = group_color[group], edgecolor = 'black', label = group)
    stars = []
    if agg_mice == 'median':
        time_course = np.median(np.array(time), axis = 0)
    elif agg_mice == 'mean':
        time_course = np.mean(np.array(time), axis = 0)
    ymin, _ = ax.get_ylim()
    print(stat[0].shape, stat[1].shape)
    for i in range(final_delta.size):
        if agg_mice == 'median':
            test = 'kruskal'
            statitci, pvalue = scipy.stats.kruskal(stat[0][:,i], stat[1][:,i])

        elif agg_mice == 'mean':
            test = 'f_oneway'
            statistic, pvalue = scipy.stats.f_oneway(stat[0][:,i], stat[1][:,i])
        print(pvalue)

        if pvalue < .05:
            ax.text(time_course[i], 1.01*ymin, '*', fontsize = 15)
        elif pvalue < .01:
            ax.text(time_course[i], 1.01*ymin, '*\n*', fontsize = 15)
        # elif pvalue < .1:
        #     ax.text(time_course[i], 80, 'a')
    fig.suptitle('bandwidth = {}, state = {}, spectrum = {}, agg = {}, test = {}'.format(bandwidth, state, spectrum_method, agg_mice, test))
    # ax.legend()
    ax.set_xticks(np.arange(16)*6)
    ax.set_xticklabels(np.arange(16)*6%24)
    # plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=40)
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.legend(frameon = False, fontsize = 10, loc =2)
    # ax.set_ylim(65, 220)

    # fig.suptitle('{} {} {}hz to {}hz power'.format(state, spectrum_method, bandwidth[0], bandwidth[1]), agg_mice)
    dirname = excel_dir + 'Control/EEG_power_time_course/auc_{}_agg_power_{}_bandref_{}/{}/'.format(auc, agg_power, band_ref,state)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(dirname + '{}_{}_{}hz_to_{}hz_power_{}.png'.format(state, spectrum_method, bandwidth[0], bandwidth[1], agg_mice), dpi = 1200)
    dirname = excel_dir + 'DCR-HCRT/EEG_power_time_course/auc_{}_agg_power_{}_bandref_{}/{}/'.format(auc, agg_power, band_ref,state)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(dirname + '{}_{}_{}hz_to_{}hz_power_{}.png'.format(state, spectrum_method, bandwidth[0], bandwidth[1], agg_mice), dpi = 1200)
    # plt.show()


def get_delta_ref_one_mouse_from_spectrum(mouse, state, spectrum_method = 'somno'):
    debug = 0
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600
    absolute_spectrum = ds['{}_spectrum'.format(spectrum_method)].values
    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
    df = (freqs[11]-freqs[1])/10
    spectrum_ref = np.empty(freqs.size)
    for bl in ['bl1', 'bl2']:
        mydict = get_clean_score_one_condition(mouse, bl, state)
        mask = mydict['condition_mask']
        clean_score = mydict['clean_score']
        times_bl = times[mask]-times[mask][0]
        spectrum_last_light = absolute_spectrum[mask]
        time_of_ref = (times_bl>=8) & (times_bl<=12)
        score_ref = clean_score[time_of_ref]
        spectrum_last_light = spectrum_last_light[time_of_ref][score_ref]
        spectrum_ref = np.vstack((spectrum_ref, spectrum_last_light))

    spectrum_ref = spectrum_ref[1:,:]
    mask_delta = (freqs>=1) & (freqs<=4)
    delta_ref = np.median(spectrum_ref,axis = 0)
    delta_ref = np.sum(delta_ref[mask_delta])*df
    return delta_ref


def get_delta_ratio_one_condition_from_spectrum(mouse, condition, state, spectrum_method = 'somno'):
    debug = 0
    print('get delta ratio for {} during {}'.format(mouse, condition))
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600

    mydict = get_clean_score_one_condition(mouse, condition, state)

    mask = mydict['condition_mask']
    mask_dark = mydict['mask_dark']
    mask_light = mydict['mask_light']
    clean_score = mydict['clean_score']

    times = times[mask]-times[mask][0]
    if condition == 'sd':
        times+=6

    absolute_spectrum = ds['{}_spectrum'.format(spectrum_method)].values
    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
    absolute_spectrum = absolute_spectrum[mask]
    ref = get_delta_ref_one_mouse_from_spectrum(mouse, state)
    # relative_spectrum = 100*absolute_spectrum/ref

    dark_intervals = 6
    # dark_intervals = 12

    ZT = []
    delta = []
    mask_delta = (freqs>=1) & (freqs<=4)
    df = (freqs[11]-freqs[1])/10

    if condition == 'sd':
        light_intervals = 8
        # light_intervals = 16
        light_intervals_borders = np.linspace(6,12,light_intervals+1)
        for interval in np.arange(light_intervals):
            ZT1 = light_intervals_borders[interval]
            ZT2 = light_intervals_borders[interval+1]
            interval_mask = (times>=ZT1) & (times<ZT2)
            interval_score = clean_score[interval_mask]
            interval_spectrum = absolute_spectrum[interval_mask]
            interval_spectrum = interval_spectrum[interval_score]
            # print(np.sum(interval_score))
            interval_spectrum = np.mean(interval_spectrum, axis = 0)[mask_delta]
            ZT.append((ZT2+ZT1)/2)
            delta.append(100*np.sum(interval_spectrum)*df/ref)

    else :
        light_intervals = 12
        # light_intervals = 24
        light_intervals_borders = np.linspace(0,12,light_intervals+1)
        for interval in np.arange(light_intervals):
            ZT1 = light_intervals_borders[interval]
            ZT2 = light_intervals_borders[interval+1]
            interval_mask = (times>=ZT1) & (times<ZT2)
            interval_score = clean_score[interval_mask]
            interval_spectrum = absolute_spectrum[interval_mask]
            interval_spectrum = interval_spectrum[interval_score]
            # print(np.sum(interval_score))
            # fig, ax = plt.subplots()
            # ax.plot(freqs, np.mean(interval_spectrum, axis = 0), color='orange')
            # inf, med, sup = np.percentile(interval_spectrum, q = [25, 50, 75], axis = 0)
            interval_spectrum = np.median(interval_spectrum, axis = 0)[mask_delta]
            # ax.plot(freqs, med)
            # ax.fill_between(freqs, inf, sup, alpha = .2)
            # ax.set_xlim(0,10)
            # plt.show()
            ZT.append((ZT2+ZT1)/2)
            delta.append(100*np.sum(interval_spectrum)*df/ref)

    dark_intervals_borders = np.linspace(12,24,dark_intervals+1)
    for interval in np.arange(dark_intervals):
        ZT1 = dark_intervals_borders[interval]
        ZT2 = dark_intervals_borders[interval+1]
        interval_mask = (times>=ZT1) & (times<ZT2)
        interval_score = clean_score[interval_mask]
        interval_spectrum = absolute_spectrum[interval_mask]
        interval_spectrum = interval_spectrum[interval_score]
        # print(np.sum(interval_score))
        interval_spectrum = np.mean(interval_spectrum, axis = 0)[mask_delta]
        ZT.append((ZT2+ZT1)/2)
        delta.append(100*np.sum(interval_spectrum)*df/ref)

    if debug :
        fig,ax = plt.subplots()
        ax.scatter(ZT, delta)
        ax.axhline(100, color = 'black', ls ='--' )
        ax.plot(ZT, delta)
        # plt.show()
    return {'ZT':np.array(ZT), 'delta':np.array(delta) }


# def plot_ratio_compare(spectrum_method = 'somno'):
    control_list = get_mice('Control')
    DCR_list = get_mice('DCR-HCRT')
    groups = {'Control' : control_list, 'DCR-HCRT' : DCR_list}
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_B2533.nc')
    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values.tolist()
    df_ratio = pd.DataFrame(index = control_list+ DCR_list, columns = np.arange(96))
    indices = []
    for state in ['w', 'n', 'r', 'a']:
        for session in ['bl1', 'bl2', 'sd', 'r1' ]:
            indices+=([i + '_'+session+ '_'+state for i in control_list+ DCR_list])
    df_spectrum = pd.DataFrame(index = indices, columns = ['group', 'session', 'state'] + freqs)

    fig,ax = plt.subplots(nrows = 4, ncols= 4, sharex = True, sharey = True)
    fig. suptitle('Spectrum per state')
    for group in groups:
        for mouse in groups[group]:
            ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
            print(ds)
            # mask = [0:int(12*3600*sr)]
            score = ds['score'].values
            score_behavior = score.copy()
            for f, n in zip(['1', '2', '3'], ['w', 'n', 'r']) :
                score_behavior = np.where(score_behavior == f, n, score_behavior)

            freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
            times = ds.coords['epochs'].values
            t_start = 8.
            real_times = times + t_start

            spectrums = ds['{}_spectrum'.format(spectrum_method)].values
            # print(spectrums.values[np.isnan(spectrums.values)])
            # print(spectrums.max(dims = 'freqs_{}'.format(spectrum_method)))
            # spectrums = (spectrums/spectrums.max(dims = 'freqs_{}'.format(spectrum_method))).values
            # print(np.max(spectrums, axis = 1).shape)

            # spectrums = spectrums/np.max(spectrums, axis = 1)
            # spectrums = spectrums*freqs
            sr =200

            delta_power = ds['{}_delta_power'.format(spectrum_method)].values
            states = ['w', 'n', 'r', 'a']

            hours = np.arange(int(times.size*4/3600))
            delta_power_by_hour = np.zeros(hours.size)
            std_delta_power_by_hour = delta_power_by_hour.copy()
            for h in hours[:-1]:
                subdata = delta_power[int(3600/4*h):int(3600/4*(h+1))]
                subscore = score[int(3600/4*h):int(3600/4*(h+1))]
                # artifact_free = (subscore != '1') & (subscore != '2') & (subscore != '3')
                nrem = subscore == 'n'
                subdata = subdata[nrem]
                m = np.mean(subdata)
                std = np.std(subdata)
                delta_power_by_hour[h] = m
                std_delta_power_by_hour[h] = std
            norm_delta_power_by_hour = delta_power_by_hour/max(delta_power_by_hour) ####A REPRENDRE
            df_ratio.at[mouse, :] = norm_delta_power_by_hour

            mydict = {'bl1' : (0, 24), 'bl2' : (24,48), 'sd':(48,72), 'r1':(78, 102)}
            for s, state in enumerate(states) :
                for row, phase in enumerate(mydict):
                    ind1, ind2 = mydict[phase][0]*900, mydict[phase][1]*900
                    score_phase = score[ind1 : ind2]
                    spectrum_phase = spectrums[ind1 :ind2,:]
                    score_no_transition = score_phase == state
                    count = 0
                    event_length = []
                    for num, value in enumerate(score_no_transition) :
                        if value:
                            count +=1
                        elif value == False:
                            event_length.append(count)
                            if count == 1:
                                score_no_transition[num-1] = False
                            count = 0
                    state_spectrum = spectrum_phase[score_no_transition]
                    if state_spectrum.shape[0] <= 1:
                        continue

                    m = state_spectrum.mean(axis=0)
                    df_spectrum.at['{}_{}_{}'.format(mouse,phase, state), 'state'] = state
                    df_spectrum.at['{}_{}_{}'.format(mouse,phase, state), 'session'] = phase
                    df_spectrum.at['{}_{}_{}'.format(mouse,phase, state), 'group'] = group
                    df_spectrum.at['{}_{}_{}'.format(mouse,phase, state), 3:] = m
        # df_spectrum = df_spectrum.dropna()
    fig_diff, ax_diff = plt.subplots(nrows = 4, ncols= 4, sharex = True, sharey = True)
    fig_diff.suptitle('Diff DCR - ctrl')
    for row, session in enumerate(mydict):
        for col, state in enumerate(states):
            if state == 'a':
                continue
            # print(df_spectrum[(df_spectrum.session == session)& (df_spectrum.state == state)])
            data_dcr = df_spectrum[(df_spectrum.session == session) & (df_spectrum.group == 'DCR-HCRT') & (df_spectrum.state == state)]
            data_ctrl = df_spectrum[(df_spectrum.session == session) & (df_spectrum.group == 'Control') & (df_spectrum.state == state)]
            data_dcr = (data_dcr[freqs]).values
            data_ctrl = (data_ctrl[freqs]).values
            # print()
            # for i, maxi in enumerate(np.max(data_dcr, axis = 1)):
            #     data_dcr[i] = data_dcr[i]/maxi
            # for i, maxi in enumerate(np.max(data_ctrl, axis = 1)):
            #     data_ctrl[i] = data_ctrl[i]/maxi
            diff = np.mean(data_dcr, axis=0) - np.mean(data_ctrl, axis=0)
            # ax[row, col].plot(data.T, color = 'black', alpha = .1)
            # ax[row, col].plot(data.mean(axis = 0), color = 'red')
            ax_diff[row, col].plot(freqs, diff )

    for group in ['Control', 'DCR-HCRT']:
        for row, session in enumerate(mydict):
            for col, state in enumerate(states):
                if group == 'Control' and state == 'a':
                    continue
                print(group, session, state)
                # print(df_spectrum[(df_spectrum.session == session)& (df_spectrum.state == state)])
                data = df_spectrum[(df_spectrum.session == session) & (df_spectrum.group == group) & (df_spectrum.state == state)]
                data = (data[freqs]).values
                # for i, maxi in enumerate(np.max(data, axis = 1)):
                #     data[i] = data[i]/maxi
                plot = np.mean(data, axis=0)
                inf, med, sup = np.percentile(data, [25,50,75], axis=0)
                inf = np.array(inf, dtype = 'float64')
                sup = np.array(sup, dtype = 'float64')
                med = np.array(med, dtype = 'float64')
                # print(np.isfinite(inf))
                # ax[row, col].plot(data.T, color = 'black', alpha = .1)
                # ax[row, col].plot(data.mean(axis = 0), color = 'red')
                ax[row, col].plot(freqs, med )
                ax[row, col].fill_between(x =freqs, y1 = inf, y2 = sup, alpha = .5 )


                # plt.show()
    plt.show()
    exit()


    fig, ax = plt.subplots()
    ax.plot(hours+t_start, delta_power_by_hour, color = 'darkgreen')
    ax.fill_between(hours+t_start, delta_power_by_hour-std_delta_power_by_hour, delta_power_by_hour+std_delta_power_by_hour, color = 'darkgreen', alpha = .5)
    fig.suptitle('Delta power for {}'.format(mouse))

    fig, ax = plt.subplots()
    ax.plot(times, delta_power, color = 'black')

    fig,ax = plt.subplots(nrows = 4, ncols= 4, sharex = True, sharey = True)
    # fig,ax = plt.subplots(nrows = 2, ncols= 4, sharex = True)
    mydict = {'bl1' : (0, 24), 'bl2' : (24,48), 'sd':(48,72), 'r1':(78, 102)}
    for s, state in enumerate(states) :
        for row, phase in enumerate(mydict):
            ind1, ind2 = mydict[phase][0]*900, mydict[phase][1]*900
            print(ind1,ind2)
            score_phase = score[ind1 : ind2]
            spectrum_phase = spectrums[ind1 :ind2,:]
            score_no_transition = score_phase == state
            count = 0
            event_length = []
            for num, value in enumerate(score_no_transition) :
                if value:
                    count +=1
                elif value == False:
                    event_length.append(count)
                    if count == 1:
                        score_no_transition[num-1] = False
                    count = 0

            state_spectrum = spectrum_phase[score_no_transition]
            if state_spectrum.shape[0] <= 1:
                continue

            print(state_spectrum.shape)
            print(state)
            # ax[1,s].plot(freqs,state_spectrum.T, color = 'black', alpha = .01)

            m = state_spectrum.mean(axis=0)
            std = state_spectrum.std(axis=0)
            p90, median, p30 = np.percentile(state_spectrum, q = (90, 50, 30), axis=0)
            # print(m)
            ax[row,s].plot(freqs,median, color = 'darkgreen')
            # ax[row,s].fill_between(freqs, p30, p90, color = 'darkgreen', alpha = .3)
            # ax[s].fill_between(freqs, m-std, m+std, color = 'darkgreen', alpha = .3)
            # ax[1,s].plot(freqs, m+std, color = 'darkgreen', alpha = .6, ls = '--')
            lim = max(median+p90)
            ax[row,s].set_ylim(0, lim*1.25)
            # ax[0,s].set_ylim(0, 30*10**-5)
            # ax[1,s].set_ylim(0, lim*4)
            # ax[1,s].set_ylim(0, 30*10**-5)
            ax[0,s].set_title(state)
    plt.show()


def get_clean_score_one_condition(mouse, condition, state):
    debug = 0
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600
    score = ds['score'].values
    if condition == 'bl1':
        t1 = 12
        mask_light = (times<t1)
        t1, t2 = 12, 24
        mask_dark = (times>=t1) & (times<t2)
        mask = (times<t2)
    elif condition == 'bl2':
        t1, t2 = 24, 36
        mask_light = (times>=t1) & (times<t2)
        t1, t2 = 36, 48
        mask_dark = (times>=t1) & (times<t2)
        t1, t2 = 24, 48
        mask = (times>=t1) & (times<t2)
    elif condition == 'sd':
        t1, t2 = 54, 60    #####caution 6 first hours is SD 48 to 54
        mask_light = (times>=t1) & (times<t2)
        t1, t2 = 60, 72
        mask_dark = (times>=t1) & (times<t2)
        t1, t2 = 54, 72
        mask = (times>=t1) & (times<t2)
    elif condition == 'r1':
        t1, t2 = 72, 84
        mask_light = (times>=t1) & (times<t2)
        t1 = 84
        mask_dark = (times>=t1)
        t1 = 72
        mask = (times>=t1)
    else :
        print('Condition does not exist !')

    mask_dark = mask_dark[mask]
    mask_light = mask_light[mask]
    score = score[mask]

    score_no_artifact = score == state
    score_no_transition = score_no_artifact.copy()
    count = 0
    event_length = []
    for num, value in enumerate(score_no_artifact) :
        if value:
            count +=1
        elif value == False:
            event_length.append(count)
            if count == 1 :
                score_no_transition[num-1] = False
            elif count == 2 :
                score_no_transition[num-1] = False
                score_no_transition[num-2] = False
                count = 0
            elif count > 2 :
                score_no_transition[num-1] = False
                score_no_transition[num-count] = False
            count = 0
    if count == 1 :
        score_no_transition[num-1] = False
    elif count == 2 :
        score_no_transition[num-1] = False
        score_no_transition[num-2] = False
        count = 0
    elif count > 2 :
        score_no_transition[num-1] = False
        score_no_transition[num-count] = False


    if debug :
        fig, ax = plt.subplots()
        s = ds.coords['times_somno'].values.size
        for i in range(5):
            ax.axvline(i*s/(3600))
        h = ds.coords['times_somno'].values/3600
        ax.plot(h, np.ones(s))
        ax.plot(h[mask], np.ones(s)[mask])
        ax.plot(h[mask], score == state)

        fig, ax = plt.subplots()
        ax.scatter(h[mask], score_no_artifact)
        ax.plot(h[mask], score_no_transition, color = 'green')
        print(count)
        plt.show()
    if sum(score_no_transition) == 0:
        print('_________ No {} for {} during {} {}________'.format(state, mouse, condition, period))
    Nd= sum(score_no_transition[mask_dark])
    Nl= sum(score_no_transition[mask_light])
    return {'clean_score' :score_no_transition,
            'condition_mask': mask,
            'mask_dark': mask_dark,
            'mask_light': mask_light,
            'Nd': Nd,
            'Nl':Nl }


def get_delta_ref_one_mouse(mouse, state, spectrum_method = 'somno'):
    debug = 0
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600
    absolute_delta_power = ds['{}_delta_power'.format(spectrum_method)].values
    delta_ref = np.array([])
    for bl in ['bl1', 'bl2']:
        mydict = get_clean_score_one_condition(mouse, bl, state)
        mask = mydict['condition_mask']
        clean_score = mydict['clean_score']
        times_bl = times[mask]-times[mask][0]
        absolute = absolute_delta_power[mask]
        absolute[clean_score==0] = np.zeros(np.sum(clean_score==0))
        size=5
        std= 2
        kernel = scipy.signal.gaussian(size, std)
        smooth = scipy.signal.fftconvolve(absolute, kernel, mode='same')
        time_of_ref = (times_bl>=8) & (times_bl<=12)
        score_ref = clean_score[time_of_ref]

        delta_ref = np.concatenate((delta_ref, smooth[time_of_ref][score_ref]))

        # fig, ax = plt.subplots(nrows=2, sharex =True)
        # ax[0].plot(times_bl,absolute)
        # ax[0].plot(times_bl[time_of_ref],absolute[time_of_ref], color ='green')
        # ax[0].plot(times_bl[time_of_ref][clean_score[time_of_ref]],absolute[time_of_ref][clean_score[time_of_ref]], color ='red')
        # ax[1].plot(times_bl,clean_score)
        # plt.show()
    ref = np.mean(delta_ref)
    # fig,ax = plt.subplots()
    # ax.plot(times_bl,100*absolute/ref)
    # plt.show()


    return ref


def get_delta_ratio_one_condition(mouse, condition, state, spectrum_method = 'somno'):
    debug = 0
    print('get delta ratio for {} during {}'.format(mouse, condition))
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600

    mydict = get_clean_score_one_condition(mouse, condition, state)

    mask = mydict['condition_mask']
    mask_dark = mydict['mask_dark']
    mask_light = mydict['mask_light']
    clean_score = mydict['clean_score']

    times = times[mask]-times[mask][0]
    if condition == 'sd':
        times+=6

    absolute_delta_power = ds['{}_delta_power'.format(spectrum_method)].values
    absolute_delta_power = absolute_delta_power[mask]
    ref = get_delta_ref_one_mouse(mouse, state)
    relative_delta_power = 100*absolute_delta_power/ref
    relative_delta_power_state = absolute_delta_power.copy()
    relative_delta_power_state[clean_score==0] = np.zeros(np.sum(clean_score==0))
    size=5
    std= 2
    kernel = scipy.signal.gaussian(size, std)
    smooth = scipy.signal.fftconvolve(relative_delta_power_state, kernel, mode='same')
    relative_mooth = 100*smooth/ref
    # fig,ax = plt.subplots()
    #
    # # ax.plot(times,relative_delta_power, alpha = .2)
    # ax.plot(times,relative_delta_power_state/max(relative_delta_power_state), alpha = .6)
    # ax.plot(times,relative_mooth)
    # plt.show()
    # exit()
    # dark_intervals = 6
    dark_intervals = 12

    ZT = []
    delta = []

    if condition == 'sd':
        # light_intervals = 8
        light_intervals = 16
        light_intervals_borders = np.linspace(6,12,light_intervals+1)
        for interval in np.arange(light_intervals):
            ZT1 = light_intervals_borders[interval]
            ZT2 = light_intervals_borders[interval+1]
            sliding_mask = (times>=ZT1) & (times<ZT2)
            sliding_score = clean_score[sliding_mask]
            sliding_delta = relative_delta_power[sliding_mask]
            sliding_delta = sliding_delta[sliding_score]
            ZT.append((ZT2+ZT1)/2)
            delta.append(np.mean(sliding_delta))

    else :
        # light_intervals = 12
        light_intervals = 24
        light_intervals_borders = np.linspace(0,12,light_intervals+1)
        for interval in np.arange(light_intervals):
            ZT1 = light_intervals_borders[interval]
            ZT2 = light_intervals_borders[interval+1]
            sliding_mask = (times>=ZT1) & (times<ZT2)
            sliding_score = clean_score[sliding_mask]
            sliding_delta = relative_delta_power[sliding_mask]
            sliding_delta = sliding_delta[sliding_score]
            ZT.append((ZT2+ZT1)/2)
            delta.append(np.mean(sliding_delta))

    dark_intervals_borders = np.linspace(12,24,dark_intervals+1)
    for interval in np.arange(dark_intervals):
        ZT1 = dark_intervals_borders[interval]
        ZT2 = dark_intervals_borders[interval+1]
        sliding_mask = (times>=ZT1) & (times<ZT2)
        sliding_score = clean_score[sliding_mask]
        sliding_delta = relative_delta_power[sliding_mask]
        sliding_delta = sliding_delta[sliding_score]
        ZT.append((ZT2+ZT1)/2)
        delta.append(np.mean(sliding_delta))

    if debug :
        fig,ax = plt.subplots()
        ax.scatter(ZT, delta)
        ax.axhline(100, color = 'black', ls ='--' )
        ax.plot(ZT, delta)
        # plt.show()

    return {'ZT':np.array(ZT), 'delta':np.array(delta) }
def get_time_course():
    print('get_time_course')
    mouse = 'B4977'
    time_course = []
    for i, session in enumerate(['bl1', 'bl2', 'sd', 'r1' ]):
        # dict = get_delta_ratio_one_condition(mouse, session, 'n')
        dict = get_delta_ratio_one_condition_from_spectrum(mouse, session, 'n')
        ZT = (dict['ZT']+i*24).tolist()
        time_course += ZT

    return time_course


def plot_compare_ratio(state, spectrum_method = 'somno'):
    debug = 1
    control_list = get_mice('Control')
    DCR_list = get_mice('DCR-HCRT')
    groups = {'Control' : control_list, 'DCR-HCRT' : DCR_list}
    time_course = get_time_course()
    indices = []
    df_ratio = pd.DataFrame(index = control_list+DCR_list, columns = ['group'] + time_course )

    for group in groups:
        for mouse in groups[group]:
            print(mouse)
            df_ratio.loc[mouse, 'group'] = group
            delta = np.array([])
            for i, session in enumerate(['bl1', 'bl2', 'sd', 'r1' ]):
                # dict = get_delta_ratio_one_condition(mouse, session, state)
                dict = get_delta_ratio_one_condition_from_spectrum(mouse, session, state, spectrum_method =spectrum_method)
                delta = np.append(delta, dict['delta'])
            df_ratio.loc[mouse, time_course] = delta
    for group in groups :
        dirname = excel_dir + '/{}/delta_ratio/'.format(group)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        name = 'delta_ratio_{}_spectrum_{}_{}.xlsx'.format(state,spectrum_method,group)
        df = df_ratio[df_ratio.group == group]
        df.to_excel(dirname+name)
    if debug :
        fig, ax = plt.subplots()
        fig.suptitle('Delta ratio for ' + state)
        for group in groups:
            subdata = df_ratio[df_ratio.group == group]
            subdata = subdata[time_course].values

            # inf, med, sup = np.nanpercentile(np.array(subdata, dtype = 'float64'), q=[25, 50,75], axis = 0)
            # inf = np.array(inf, dtype = 'float64')
            # sup = np.array(sup, dtype = 'float64')
            # med = np.array(med, dtype = 'float64')
            # ax.plot(time_course, med, label = group, ls ='--')
            # ax.fill_between(time_course, inf, sup, alpha = .2)

            ax.plot(time_course, np.nanmean(subdata, axis = 0), label = group)
            # ax.plot(time_course, subdata.T, alpha = .1)

            # ax.scatter(time_course, np.nanmean(subdata, axis = 0))
        ax.legend()
    # plt.show()

def get_event_number_day_light(state):
    for group in ['Control', 'DCR-HCRT']:
        print(group)
        d =  0
        l =  0
        distrib = { 'bl1' : np.zeros(21600),
                    'bl2' : np.zeros(21600),
                    'sd' : np.zeros(16200),
                    'r1' : np.zeros(21600)
                    }
        fig, ax = plt.subplots(nrows = 4, sharex = True)
        for mouse in get_mice(group):
            for condition in ['bl1', 'bl2', 'sd', 'r1']:
                dict = get_clean_score_one_condition(mouse, condition, state)
                d+=dict['Nd']
                l+=dict['Nl']
                clean_score = dict['clean_score']
                dist = distrib[condition]
                dist +=clean_score*1
        plot = distrib['bl1']
        ax[0].plot((4*np.arange(plot.size)/3600), plot)
        plot = distrib['bl2']
        ax[1].plot((4*np.arange(plot.size)/3600), plot)
        plot = distrib['sd']
        ax[2].plot((4*np.arange(plot.size)/3600)+6, plot)
        plot = distrib['r1']
        ax[3].plot((4*np.arange(plot.size)/3600), plot)
        plt.show()
        print(l)
        print(d)
        print(d/l)

def plot_delta_by_mouse(state = 'n',spectrum_method = 'somno'):
    control_list = get_mice('Control')
    DCR_list = get_mice('DCR-HCRT')
    groups = {'Control' : control_list, 'DCR-HCRT' : DCR_list}
    time_course = get_time_course()

    for group in groups:
        for mouse in groups[group]:
            fig, ax = plt.subplots()
            fig.suptitle(mouse + ' -- ' + spectrum_method + ' -- ' + group)
            print(mouse)
            delta = np.array([])
            for i, session in enumerate(['bl1', 'bl2', 'sd', 'r1' ]):
                # dict = get_delta_ratio_one_condition(mouse, session, state)
                dict = get_delta_ratio_one_condition_from_spectrum(mouse, session, state, spectrum_method =spectrum_method)
                delta = np.append(delta, dict['delta'])
            ax.plot(time_course,delta)

            dirname = work_dir+'/pyFig/{}/{}/delta/'.format(spectrum_method,group)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            plt.savefig(dirname + mouse+'.png')


def TF_percentile(state = 'w', group ='Control', spectrum_method = 'somno', auc = False, agg_power='mean'):
    dirname = excel_dir + '{}/TF_time_course/auc_{}_agg_power_{}/{}/'.format(group,auc, agg_power, state)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    final_ZT = np.array([])
    final_delta = np.array([])
    n_loop = 0
    mice = get_mice(group)

    mouse = 'B2700'
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    ds_hf = xr.open_dataset(precompute_dir + '/spectrums/HF_spectrum_scoring_{}.nc'.format(mouse))

    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
    freqs_hf = ds_hf.coords['freqs_{}'.format(spectrum_method)].values
    freqs = np.concatenate((freqs, freqs_hf))
    if state in ['w','t']:
        percentiles = np.arange(77)
    elif state in ['n','r']:
        percentiles = np.arange(66)
    df_power = pd.DataFrame(index = mice, columns = percentiles)
    df_times = pd.DataFrame(index = mice, columns = percentiles)

    coords = {'percentiles' : percentiles,'freqs' : freqs }
    # ds_tf = xr.Dataset(coords = coords)
    TF_all_mice = np.zeros((percentiles.size, freqs.size, len(mice)))
    ZT_all_mice = np.zeros((percentiles.size, len(mice)))

    for m,mouse in enumerate(mice):
        print(mouse)
        if mouse == 'B2767':
            continue
        # if mouse != 'B4977':
        #     continue
        n_loop +=1
        ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
        ds_hf = xr.open_dataset(precompute_dir + '/spectrums/HF_spectrum_scoring_{}.nc'.format(mouse))

        score = ds['score'].values
        times = ds.coords['times_somno'].values/3600
        freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
        freqs_hf = ds_hf.coords['freqs_{}'.format(spectrum_method)].values
        freqs = np.concatenate((freqs, freqs_hf))

        df = (freqs[11]-freqs[1])/10
        spectrum = ds['{}_spectrum'.format(spectrum_method)].values
        spectrum_hf = ds_hf['{}_spectrum'.format(spectrum_method)].values
        spectrum = np.concatenate((spectrum, spectrum_hf), axis = 1)


        tdw_dirname = precompute_dir + '/tdw_score/'
        ds_tdw = xr.open_dataset(tdw_dirname + 'tdw_score_{}.nc'.format(mouse))
        score = ds_tdw['new_score'].values

        if state == 'w':
            score_no_artifact = (score == 'w') | (score =='t')
        else :
            score_no_artifact = score == state

        score_no_transition = score_no_artifact.copy()
        count = 0
        event_length = []
        for num, value in enumerate(score_no_artifact) :
            if value:
                count +=1
            elif value == False:
                event_length.append(count)
                if count == 1 :
                    score_no_transition[num-1] = False
                elif count == 2 :
                    score_no_transition[num-1] = False
                    score_no_transition[num-2] = False
                    count = 0
                elif count > 2 :
                    score_no_transition[num-1] = False
                    score_no_transition[num-count] = False
                count = 0
        if count == 1 :
            score_no_transition[num-1] = False
        elif count == 2 :
            score_no_transition[num-1] = False
            score_no_transition[num-2] = False
            count = 0
        elif count > 2 :
            score_no_transition[num-1] = False
            score_no_transition[num-count] = False
#######################
        artefact_tresh = np.median(np.mean(spectrum[score_no_transition],axis = 1)) + 10*scipy.stats.median_absolute_deviation(np.mean(spectrum[score_no_transition],axis = 1))   #####34???
        arte = np.mean(spectrum[score_no_transition],axis = 1)>artefact_tresh
        index = np.arange(score_no_transition.size)
        index = index[score_no_transition][arte]
        # fig, ax = plt.subplots()
        # ax.plot(np.mean(spectrum[score_no_transition],axis = 1))
        # ax.axhline(artefact_tresh)
        # plt.show()

        if index.size!=0:
            for i in [index]:
        #         if value :
        #             corr[i]=False
                score_no_transition[i]=False
#####################
        mask_ref = ((times>8) & (times<12)) | ((times>32) & (times<36)) ####derived from sleep
        # mask_ref = ((times>14) & (times<18)) | ((times>24+14) & (times<24+18))
        # mask_ref = ((times>12) & (times<24)) | ((times>36) & (times<48))
        # mask_ref = ((times>0) & (times<48))
        score_ref = score_no_transition[mask_ref]
        spec_ref = spectrum[mask_ref]
        mask_bandwidth_ref = np.ones(freqs.size).astype('bool')


        if agg_power == 'mean':
            ref = np.mean(spec_ref[score_ref], axis = 0)
        elif agg_power == 'median':
            ref = np.median(spec_ref[score_ref], axis = 0)
        if auc == True:
            ref *= df

        time_score = times[score_no_transition]
        percentile = []

        if state in ['w','t']:
            mask = (time_score<12)
            if np.sum(mask)==0:
                percentile.append(np.ones(6+1)*-1)
            else :
                p = np.linspace(0,100,6+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>12)&(time_score<24)
            if np.sum(mask)==0:
                percentile.append(np.ones(12+1)*-1)
            else :
                p = np.linspace(0,100,12+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>24)&(time_score<36)
            if np.sum(mask)==0:
                percentile.append(np.ones(6+1)*-1)
            else :
                p = np.linspace(0,100,6+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>36)&(time_score<48)
            if np.sum(mask)==0:
                percentile.append(np.ones(12+1)*-1)
            else :
                p = np.linspace(0,100,12+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>48)&(time_score<54)
            if np.sum(mask)==0:
                percentile.append(np.ones(8+1)*-1)
            else :
                p = np.linspace(0,100,8+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>54)&(time_score<60)
            if np.sum(mask)==0:
                percentile.append(np.ones(3+1)*-1)
            else :
                p = np.linspace(0,100,3+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>60)&(time_score<72)
            if np.sum(mask)==0:
                percentile.append(np.ones(12+1)*-1)
            else :
                p = np.linspace(0,100,12+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>72)&(time_score<84)
            if np.sum(mask)==0:
                percentile.append(np.ones(6+1)*-1)
            else :
                p = np.linspace(0,100,6+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>84)&(time_score<96)
            if np.sum(mask)==0:
                percentile.append(np.ones(12+1)*-1)
            else :
                p = np.linspace(0,100,12+1)
                percentile.append(np.percentile(time_score[mask], p))


            percentile = np.concatenate(percentile)
            ZT = []
            # exceptions = [12, 19, 32, 39, 52, 59, 72]
            exceptions = [6,19,26, 39, 48, 52,65, 72]



        elif state in ['r', 'n']:
            mask = (time_score<12)
            if np.sum(mask)==0:
                percentile.append(np.ones(12+1)*-1)
            else:
                p = np.linspace(0,100,12+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>12)&(time_score<24)
            if np.sum(mask)==0:
                percentile.append(np.ones(6+1)*-1)
            else:
                p = np.linspace(0,100,6+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>24)&(time_score<36)
            if np.sum(mask)==0:
                percentile.append(np.ones(12+1)*-1)
            else:
                p = np.linspace(0,100,12+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>36)&(time_score<48)
            if np.sum(mask)==0:
                percentile.append(np.ones(6+1)*-1)
            else:
                p = np.linspace(0,100,6+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>54)&(time_score<60)
            if np.sum(mask)==0:
                percentile.append(np.ones(6+1)*-1)
            else:
                p = np.linspace(0,100,6+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>60)&(time_score<72)
            if np.sum(mask)==0:
                percentile.append(np.ones(6+1)*-1)
            else:
                p = np.linspace(0,100,6+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>72)&(time_score<84)
            if np.sum(mask)==0:
                percentile.append(np.ones(12+1)*-1)
            else:
                p = np.linspace(0,100,12+1)
                percentile.append(np.percentile(time_score[mask], p))

            mask = (time_score>84)&(time_score<96)
            if np.sum(mask)==0:
                percentile.append(np.ones(6+1)*-1)
            else:
                p = np.linspace(0,100,6+1)
                percentile.append(np.percentile(time_score[mask], p))

            percentile = np.concatenate(percentile)
            ZT = []
            # exceptions = [12, 19, 32, 39, 48, 55, 68 ]
            exceptions = [12, 19, 32, 39, 46, 53, 66]


        z = 0
        for i in range(percentile.size-1):
            if i in exceptions :
                continue
            else :
                t1, t2 = percentile[i], percentile[i+1]
                # print(t1,t2)
                if t1 == -1:
                    # absolute_delta.append(np.nan)
                    ZT.append(np.nan)
                    # print('erre')
                else :
                    mask = (times>t1) & (times<t2)
                    interval_score =score_no_transition[mask]
                    interval_spectrum = spectrum[mask]
                    # print(i)
                    # print('         ', interval_spectrum[interval_score].shape[0])
                    if agg_power == 'median':
                        mean_spec = np.median(interval_spectrum[interval_score], axis = 0)
                    elif agg_power == 'mean':
                        mean_spec = np.mean(interval_spectrum[interval_score], axis = 0)
                        # BLA.append(mean_spec)

                    if i == 0:
                        MM = mean_spec
                    else :
                        MM = np.vstack((MM, mean_spec))
                    if auc :
                        # absolute_delta.append(mean_spec)*df)
                        TF_all_mice[z,:,m] = 100*mean_spec*df/ref
                    elif auc ==False :
                        # absolute_delta.append(mean_spec))
                        TF_all_mice[z,:,m] = 100*mean_spec/ref
                    ZT.append((t1+t2)/2)
                    z+=1
        # fig, ax =plt.subplots()
        # ax.plot(BLA_ref)
        # BLA = np.array(BLA)
        # ax.plot(BLA.T, alpha = .1)
        # plt.show()
        ZT = np.array(ZT)
        ZT_all_mice[:,m] = ZT
        # absolute_delta = np.array(absolute_delta)
        # relative_delta = 100*absolute_delta/ref





    # x = np.mean(ZT_all_mice, axis =1)
    # y = np.mean(TF_all_mice, axis = 2)
    #
    #
    # print(x.shape)
    # print(y.shape)
    # fig,ax = plt.subplots()
    # ax.imshow(y.T, origin = 'lower',aspect = 'auto', vmax = 150)
    # f = freqs[::10]
    # t = x[::6]
    # t = np.round(t, 1)
    # f = np.round(f,1)
    # ax.set_yticks(np.arange(f.size)*10)
    # ax.set_yticklabels(f)
    # ax.set_xticks(np.arange(t.size)*6)
    # ax.set_xticklabels(t, rotation = 45)
    #
    # plt.show()

    coords = {'percentiles' : percentiles,'freqs' : freqs, 'mice':mice }

    ds_tf = xr.Dataset(coords = coords)
    ds_tf['TF'] = xr.DataArray(TF_all_mice, dims = ['percentiles', 'freqs','mice'])
    ds_tf['ZT'] = xr.DataArray(ZT_all_mice, dims = ['percentiles','mice'])

    return ds_tf

def contrast(state = 'w',spectrum_method = 'somno', auc = False, agg_power='mean'):
    ds_control = TF_percentile(state = state, group ='Control', spectrum_method = spectrum_method, auc = auc, agg_power=agg_power)
    ds_dicer = TF_percentile(state = state, group ='DCR-HCRT', spectrum_method = spectrum_method, auc = auc, agg_power=agg_power)

    tf_control = ds_control['TF'].mean(dim ='mice').values
    tf_dicer = ds_dicer['TF'].mean(dim ='mice').values

    # contrast = tf_dicer/tf_control
    contrast = (tf_dicer-tf_control)/(tf_dicer+tf_control)

    t = ds_dicer['ZT'].mean(dim='mice').values
    freqs = ds_dicer['freqs'].values
    f = freqs[::10]
    t = t[::6]
    t = np.round(t, 1)
    f = np.round(f,1)
    dirname1 = excel_dir + '/Control/TimeFreq/auc_{}_agg_power_{}/{}/'.format(auc, agg_power, state)
    if not os.path.exists(dirname1):
        os.makedirs(dirname1)
    dirname2 = excel_dir + '/DCR-HCRT/TimeFreq/auc_{}_agg_power_{}/{}/'.format(auc, agg_power, state)
    if not os.path.exists(dirname2):
        os.makedirs(dirname2)
    y =tf_control
    fig,ax = plt.subplots()
    ax.imshow(y.T, origin = 'lower',aspect = 'auto', vmax = 150)
    ax.set_yticks(np.arange(f.size)*10)
    ax.set_yticklabels(f)
    ax.set_xticks(np.arange(t.size)*6)
    ax.set_xticklabels(t, rotation = 45)
    fig.suptitle('Control - ' +state)
    fig.savefig(dirname1 + 'TF_control_{}.png'.format(state))
    fig.savefig(dirname2 + 'TF_control_{}.png'.format(state))


    y =tf_dicer
    fig,ax = plt.subplots()
    ax.imshow(y.T, origin = 'lower',aspect = 'auto', vmax = 150)
    ax.set_yticks(np.arange(f.size)*10)
    ax.set_yticklabels(f)
    ax.set_xticks(np.arange(t.size)*6)
    ax.set_xticklabels(t, rotation = 45)
    fig.suptitle('DCR-HCRT - ' +state)
    fig.savefig(dirname1 + 'TF_dicer_{}.png'.format(state))
    fig.savefig(dirname2 + 'TF_dicer_{}.png'.format(state))

    y =contrast
    fig,ax = plt.subplots()
    p  = ax.imshow(y.T, origin = 'lower',aspect = 'auto', vmax = .1, vmin = -.2)
    ax.set_yticks(np.arange(f.size)*10)
    ax.set_yticklabels(f)
    ax.set_xticks(np.arange(t.size)*6)
    ax.set_xticklabels(t, rotation = 45)
    fig.colorbar(p)
    fig.suptitle('contrast - ' +state)
    fig.savefig(dirname1 + 'TF_contrast_{}.png'.format(state))
    fig.savefig(dirname2 + 'TF_contrast_{}.png'.format(state))
    # fig, ax = plt.subplots()
    # ax.plot(freqs, np.mean(y,axis = 0))
    #
    # plt.show()


def compute_all():
    for agg in ['mean', 'median']:
        contrast(state = 'w', spectrum_method = 'welch', auc = False, agg_power=agg)
        contrast(state = 't', spectrum_method = 'welch', auc = False, agg_power=agg)
        contrast(state = 'n', spectrum_method = 'welch', auc = False, agg_power=agg)
        contrast(state = 'r', spectrum_method = 'welch', auc = False, agg_power=agg)
        for band in ['total', 'band']:
            plot_compare_wake(bandwidth = [1,4],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [1,2.25],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [2.25,4],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_sleep(bandwidth = [1,4],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_sleep(bandwidth = [1,2.25],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_sleep(bandwidth = [2.25,4],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [7,11],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [7,9],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [9,11],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [7,11],state = 't', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [7,9],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [9,11],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_sleep(bandwidth = [7,11],state = 'r', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_sleep(bandwidth = [7,9],state = 'r', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_sleep(bandwidth = [9,11],state = 'r', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [32,45],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [.75,1.5],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [4,6],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [6,8],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [9,12],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [13,17],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [34,44],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_sleep(bandwidth = [.75,1.5],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_sleep(bandwidth = [1.5,4],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
            plot_compare_wake(bandwidth = [9,12],state = 't', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)

def band_directed_compute_all():
    for band in ['total', 'band']:
        plot_compare_wake(bandwidth = [.75,1.5],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
        plot_compare_wake(bandwidth = [4,6],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
        plot_compare_wake(bandwidth = [6,8],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
        plot_compare_wake(bandwidth = [9,12],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
        plot_compare_wake(bandwidth = [13,17],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
        plot_compare_wake(bandwidth = [34,44],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
        plot_compare_sleep(bandwidth = [.75,1.5],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
        plot_compare_sleep(bandwidth = [1.5,4],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)
        plot_compare_wake(bandwidth = [9,12],state = 't', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = agg, band_ref = band)


if __name__ == '__main__':
    # mouse = 'B3512'
    # mouse = 'B2534'
    # mouse = 'B2533'
    # mouse = 'B2767'
    # mouse = 'B2761'
    # mouse = 'B2700'
    # mouse = 'B2762'
    # mouse = 'B2763'
    # mouse = 'B3072'
    # mouse = 'B3140'
    # mouse = 'B3513'
    # mouse = 'B3512'
    # mouse = 'B3766'
    # mouse = 'B4112'
    # mouse = 'B4113'
    # mouse = 'B4975'
    # mouse = 'B4976'
    # mouse = 'B4907'

    group = 'DCR-HCRT'
    # rec = 'b1'

    # mouse = 'B4904'
    mouse = 'B4906'
    # mouse  ='B2762'
    # mouse  ='B2763'
    # mouse  ='B3072'
    # mouse  ='B3140'
    # mouse  ='B3513'
    # mouse = 'B4977'
    # mouse = 'B2767'
    # group = 'Control'
    # plot_compare_wake(bandwidth = [55,62],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'band')
    # plot_compare_wake(bandwidth = [66,80],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'band')
    # # plt.show()
    # # exit()
    # # wake_TF_percentile(state = 'w', group ='Control', spectrum_method = 'welch', auc = False, agg_power='median')
    # contrast(state = 'n', spectrum_method = 'welch', auc = False, agg_power='median')
    # # band_directed_compute_all()
    # compute_all()
    # plt.show()
    # plot_compare_wake(bandwidth = [1,4],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'band')
    # plot_compare_wake(bandwidth = [2,4],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'band')
    # plot_compare_sleep(bandwidth = [1,4],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'band')

    # plot_compare_wake(bandwidth = [9,11],state = 't', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'band')
    # plot_compare_wake(bandwidth = [9,11],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'band')
    # plt.show()
    # plot_compare_wake(bandwidth = [6,11],state = 'r', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'total')
    # plot_compare_wake(bandwidth = [6,9],state = 't', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'total')
    # plot_compare_wake(bandwidth = [32,45],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'total')
    # plt.show()
    # plot_compare_wake(bandwidth = [1,4],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'total')
    # # exit()
    # # plot_compare_sleep(bandwidth = [6,9],state = 'r', spectrum_method = 'welch', agg_mice='mean')
    # plot_compare_sleep(bandwidth = [1,4],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'total')
    # # exit()
    # plot_compare_wake(bandwidth = [6,9],state = 'w', spectrum_method = 'welch', agg_mice='median', auc = False, agg_power = 'median', band_ref = 'total')
    # plot_compare_wake(bandwidth = [6,9],state = 't', spectrum_method = 'welch', agg_mice='median', auc = False, agg_power = 'median', band_ref = 'total')
    # plot_compare_wake(bandwidth = [32,45],state = 'w', spectrum_method = 'welch', agg_mice='median', auc = False, agg_power = 'median', band_ref = 'total')
    # plot_compare_wake(bandwidth = [1,4],state = 'w', spectrum_method = 'welch', agg_mice='median', auc = False, agg_power = 'median', band_ref = 'total')
    # # plot_compare_sleep(bandwidth = [6,9],state = 'r', spectrum_method = 'welch', agg_mice='mean')
    # plot_compare_sleep(bandwidth = [1,4],state = 'n', spectrum_method = 'welch', agg_mice='median', auc = False, agg_power = 'median', band_ref = 'total')
    #
    plot_compare_sleep(bandwidth = [10,15],state = 'n', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'total')

    # exit()
    #####erase below
    # plot_compare_wake(bandwidth = [7,11],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'band')
    # plot_compare_wake(bandwidth = [8.5,9.75],state = 'w', spectrum_method = 'welch', agg_mice='mean', auc = False, agg_power = 'mean', band_ref = 'band')
    plt.show()
