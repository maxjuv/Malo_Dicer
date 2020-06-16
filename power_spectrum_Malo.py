from configuration import *
from select_mice_cata_Malo import get_mice
from sklearn.metrics import auc

local_path = os.path.dirname(os.path.realpath(__file__))
print(local_path)


def get_clear_spectral_score_one_mouse_one_condition(mouse, condition, state):
    debug = False
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600
    # score = ds['score'].values
    dirname = precompute_dir + '/tdw_score/'
    ds_tdw = xr.open_dataset(dirname + 'tdw_score_{}.nc'.format(mouse))
    score = ds_tdw['new_score'].values


    if condition == 'bl1':
        t2 = 24
        mask = (times<t2)
    elif condition == 'bl2':
        t1, t2 = 24, 48
        mask = (times>=t1) & (times<t2)
    elif condition == 'sd':
        t1, t2 = 48+6, 72
        mask = (times>=t1) & (times<t2)
    elif condition == 'r1':
        t1 = 72
        mask = (times>=t1)
    else :
        print('Condition does not exist !')

    # mask = (times>=t1) & (times<t2)
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

    return score_no_transition, mask

def get_relative_spectrums_one_mouse_one_condition(mouse, condition, spectrum_method = 'somno'):
    debug = 0
    cleared_score_w, _= get_clear_spectral_score_one_mouse_one_condition(mouse, condition, 'w')
    cleared_score_r, _ = get_clear_spectral_score_one_mouse_one_condition(mouse, condition, 'r')
    cleared_score_t, _ = get_clear_spectral_score_one_mouse_one_condition(mouse, condition, 't')
    cleared_score_n, mask_condition = get_clear_spectral_score_one_mouse_one_condition(mouse, condition, 'n')

    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))

    spectrum = ds['{}_spectrum'.format(spectrum_method)].values
    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
    spectrum = spectrum[mask_condition, :]

    total_epochs = sum(cleared_score_w) + sum(cleared_score_r) + sum(cleared_score_n) + sum(cleared_score_t)

    proportion_w = sum(cleared_score_w)/total_epochs
    proportion_r = sum(cleared_score_r)/total_epochs
    proportion_n = sum(cleared_score_n)/total_epochs
    proportion_t = sum(cleared_score_t)/total_epochs

    absolute_spectrum_w = np.median(spectrum[cleared_score_w, :], axis = 0)
    absolute_spectrum_r = np.median(spectrum[cleared_score_r, :], axis = 0)
    absolute_spectrum_n = np.median(spectrum[cleared_score_n, :], axis = 0)
    absolute_spectrum_t = np.median(spectrum[cleared_score_t, :], axis = 0)

    #########CAUTION power spectrum is area under curv#########
    auc_w = auc(freqs, absolute_spectrum_w)
    auc_r = auc(freqs, absolute_spectrum_r)
    auc_n = auc(freqs, absolute_spectrum_n)
    auc_t = auc(freqs, absolute_spectrum_t)
    ref_values = proportion_w*auc_w + proportion_r*auc_r + proportion_n*auc_n + proportion_t*auc_t


    ####FALSE
    # mean_w = np.mean(absolute_spectrum_w)
    # mean_r = np.mean(absolute_spectrum_r)
    # mean_n = np.mean(absolute_spectrum_n)
    # ref_values = proportion_w*mean_w + proportion_r*mean_r + proportion_n*mean_n



    #####ALi's
    # sum_w = np.sum(absolute_spectrum_w)
    # sum_r = np.sum(absolute_spectrum_r)
    # sum_n = np.sum(absolute_spectrum_n)
    # ref_values = (proportion_w*sum_w + proportion_r*sum_r + proportion_n*sum_n)/100
    #
    # relative_spectrum_w = 100*absolute_spectrum_w / ref_values
    # relative_spectrum_r = 100*absolute_spectrum_r / ref_values
    # relative_spectrum_n = 100*absolute_spectrum_n / ref_values

    # print(ref_values)
    # fig, ax = plt.subplots()
    # global_spec = np.mean(spectrum[cleared_score_w + cleared_score_r + cleared_score_n], axis = 0)
    # print(auc(freqs,global_spec))
    # ax.plot(freqs,global_spec )
    # ax.plot(freqs,global_spec/ref_values )
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(freqs, relative_spectrum_w)
    # ax.plot(freqs, relative_spectrum_r)
    # ax.plot(freqs, relative_spectrum_n)
    # plt.show()

    if debug == True:
        fig, ax = plt.subplots()
        ax.plot(absolute_spectrum_w, color = 'blue')
        ax.plot(absolute_spectrum_r, color = 'blue')
        ax.plot(absolute_spectrum_n, color = 'blue')
        ax.plot(absolute_spectrum_t, color = 'blue')

        ax.plot(relative_spectrum_w, color = 'green')
        ax.plot(relative_spectrum_r, color = 'green', alpha = .75)
        ax.plot(relative_spectrum_n, color = 'green', alpha = .50)
        ax.plot(relative_spectrum_t, color = 'green', alpha = .25)
        plt.show()

    return ref_values, {'w' : relative_spectrum_w, 't' : relative_spectrum_t, 'r':relative_spectrum_r, 'n' :relative_spectrum_n}

def get_clear_spectral_score_one_mouse_one_condition_day_night(mouse, condition, state, period = 'light'):
    debug = 0
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600
    # score = ds['score'].values
    dirname = precompute_dir + '/tdw_score/'
    ds_tdw = xr.open_dataset(dirname + 'tdw_score_{}.nc'.format(mouse))
    score = ds_tdw['new_score'].values

    if condition == 'bl1':
        if period == 'light':
            t1 = 12
            mask = (times<t1)
        elif period == 'dark':
            t1, t2 = 12, 24
            mask = (times>=t1) & (times<t2)
    elif condition == 'bl2':
        if period == 'light':
            t1, t2 = 24, 36
            mask = (times>=t1) & (times<t2)
        elif period == 'dark':
            t1, t2 = 36, 48
            mask = (times>=t1) & (times<t2)
    elif condition == 'sd':
        if period == 'light':
            t1, t2 = 54, 60    #####caution 6 first hours is SD 48 to 54
            mask = (times>=t1) & (times<t2)
        elif period == 'dark':
            t1, t2 = 60, 72
            mask = (times>=t1) & (times<t2)
    elif condition == 'r1':
        if period == 'light':
            t1, t2 = 72, 84
            mask = (times>=t1) & (times<t2)
        elif period == 'dark':
            t1 = 84
            mask = (times>=t1)
    else :
        print('Condition does not exist !')

    # mask = (times>=t1) & (times<t2)
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
    return score_no_transition, mask

def get_relative_spectrums_one_mouse_one_condition_day_night(mouse, condition, spectrum_method = 'somno', period = 'light'):
    debug = 0
    cleared_score_w, _= get_clear_spectral_score_one_mouse_one_condition_day_night(mouse, condition, 'w', period = period)
    cleared_score_t, _= get_clear_spectral_score_one_mouse_one_condition_day_night(mouse, condition, 't', period = period)
    cleared_score_r, _ = get_clear_spectral_score_one_mouse_one_condition_day_night(mouse, condition, 'r', period = period)
    cleared_score_n, mask_condition = get_clear_spectral_score_one_mouse_one_condition_day_night(mouse, condition, 'n', period = period)

    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))

    spectrum = ds['{}_spectrum'.format(spectrum_method)].values
    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
    spectrum = spectrum[mask_condition, :]

    total_epochs = sum(cleared_score_w) + sum(cleared_score_r) + sum(cleared_score_n) + sum(cleared_score_t)

    proportion_w = sum(cleared_score_w)/total_epochs
    proportion_r = sum(cleared_score_r)/total_epochs
    proportion_n = sum(cleared_score_n)/total_epochs
    proportion_t = sum(cleared_score_t)/total_epochs

    absolute_spectrum_w = np.nanmedian(spectrum[cleared_score_w, :], axis = 0)
    absolute_spectrum_r = np.nanmedian(spectrum[cleared_score_r, :], axis = 0)
    absolute_spectrum_n = np.nanmedian(spectrum[cleared_score_n, :], axis = 0)
    absolute_spectrum_t = np.nanmedian(spectrum[cleared_score_t, :], axis = 0)

    df = (freqs[11]-freqs[1])/10


    if np.sum(cleared_score_w) !=0:
        sum_w = np.sum(absolute_spectrum_w)*df
    else:
        sum_w =0

    if np.sum(cleared_score_t) !=0:
        sum_t = np.sum(absolute_spectrum_t)*df
    else:
        sum_t =0

    if np.sum(cleared_score_r) !=0:
        sum_r = np.sum(absolute_spectrum_r)*df
    else:
        sum_r =0

    if np.sum(cleared_score_n) !=0:
        sum_n = np.sum(absolute_spectrum_n)*df
    else:
        sum_n = 0


    #########CAUTION power spectrum is area under curv#########
    # auc_w = auc(freqs, absolute_spectrum_w)
    # auc_r = auc(freqs, absolute_spectrum_r)
    # auc_n = auc(freqs, absolute_spectrum_n)
    # ref_values = proportion_w*auc_w + proportion_r*auc_r + proportion_n*auc_n


    ####FALSE
    # mean_w = np.mean(absolute_spectrum_w)
    # mean_r = np.mean(absolute_spectrum_r)
    # mean_n = np.mean(absolute_spectrum_n)
    # ref_values = proportion_w*mean_w + proportion_r*mean_r + proportion_n*mean_n


    #####ALi's
    # sum_w = np.sum(absolute_spectrum_w)*df
    # sum_r = np.sum(absolute_spectrum_r)*df
    # sum_n = np.sum(absolute_spectrum_n)*df
    ref_values = (proportion_w*sum_w + proportion_r*sum_r + proportion_n*sum_n + proportion_t*sum_t)


    relative_spectrum_w = 100*absolute_spectrum_w / ref_values
    relative_spectrum_r = 100*absolute_spectrum_r / ref_values
    relative_spectrum_n = 100*absolute_spectrum_n / ref_values
    relative_spectrum_t = 100*absolute_spectrum_t / ref_values

    # print(ref_values)
    # fig, ax = plt.subplots()
    # global_spec = np.mean(spectrum[cleared_score_w + cleared_score_r + cleared_score_n], axis = 0)
    # print(auc(freqs,global_spec))
    # ax.plot(freqs,global_spec )
    # ax.plot(freqs,global_spec/ref_values )
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(freqs, relative_spectrum_w)
    # ax.plot(freqs, relative_spectrum_r)
    # ax.plot(freqs, relative_spectrum_n)
    # plt.show()

    if debug == True:
        fig, ax = plt.subplots()
        ax.plot(absolute_spectrum_w, color = 'blue')
        ax.plot(absolute_spectrum_r, color = 'blue')
        ax.plot(absolute_spectrum_n, color = 'blue')
        ax.plot(absolute_spectrum_t, color = 'blue')

        ax.plot(relative_spectrum_w, color = 'green')
        ax.plot(relative_spectrum_r, color = 'green', alpha = .75)
        ax.plot(relative_spectrum_n, color = 'green', alpha = .5)
        ax.plot(relative_spectrum_t, color = 'green', alpha = .25)
        plt.show()
    if mouse in get_mice('DCR-HCRT'):
        cleared_score_a, _ = get_clear_spectral_score_one_mouse_one_condition_day_night(mouse, condition, 'a', period = period)
        absolute_spectrum_a = np.mean(spectrum[cleared_score_a, :], axis = 0)
        relative_spectrum_a = 100*absolute_spectrum_a / ref_values
        return ref_values, {'w' : relative_spectrum_w, 't' : relative_spectrum_t, 'r':relative_spectrum_r, 'n' :relative_spectrum_n, 'a' :relative_spectrum_a}
    else :
        return ref_values, {'w' : relative_spectrum_w, 't' : relative_spectrum_t,'r':relative_spectrum_r, 'n' :relative_spectrum_n}




def plot_spectrum_compare_global(spectrum_method = 'somno'):
    control_list = get_mice('Control')
    DCR_list = get_mice('DCR-HCRT')
    groups = {'Control' : control_list, 'DCR-HCRT' : DCR_list}
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_B2533.nc')
    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values.tolist()

    indices = []
    for state in ['w', 'n', 'r', 't']:
        for session in ['bl1', 'bl2', 'sd', 'r1' ]:
            indices+=([i + '_'+session+ '_'+state for i in control_list+ DCR_list])
    df_spectrum = pd.DataFrame(index = indices, columns = ['group', 'session', 'state', 'ref_values'] + freqs)



    for group in groups:
        for mouse in groups[group]:
            ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
            freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
            ZT = ds.coords['times_somno'].values
            print('collecting relative spectrums ', mouse)

            states = ['w', 'n', 'r', 't']
            conditions = ['bl1', 'bl2', 'sd', 'r1']

            for condition in conditions:
                ref_values, relative_spectrums = get_relative_spectrums_one_mouse_one_condition(mouse, condition)
                for state in states :
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 'state'] = state
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 'session'] = condition
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 'group'] = group
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 'ref_values'] = ref_values
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 4:] = relative_spectrums[state]

        # df_spectrum = df_spectrum.dropna()
    fig,ax = plt.subplots(nrows = 4, ncols= 4, sharex = True, sharey = True)
    fig.suptitle('Spectrum per state')

    fig_diff, ax_diff = plt.subplots(nrows = 4, ncols= 4, sharex = True, sharey = True)
    fig_diff.suptitle('Diff DCR - ctrl')
    for row, session in enumerate(conditions):
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
            diff = np.median(data_dcr, axis=0) - np.median(data_ctrl, axis=0)
            # ax[row, col].plot(data.T, color = 'black', alpha = .1)
            # ax[row, col].plot(data.mean(axis = 0), color = 'red')
            ax_diff[row, col].plot(freqs, diff )

    for group in ['Control', 'DCR-HCRT']:
        for row, session in enumerate(conditions):
            for col, state in enumerate(states):
                if group == 'Control' and state == 'a':
                    continue
                print(group, session, state)
                data = df_spectrum[(df_spectrum.session == session) & (df_spectrum.group == group) & (df_spectrum.state == state)]
                data = (data[freqs]).values
                plot = np.median(data, axis=0)
                # inf, med, sup = np.percentile(data, [25,50,75], axis=0)
                # inf = np.array(inf, dtype = 'float64')
                # sup = np.array(sup, dtype = 'float64')
                # med = np.array(med, dtype = 'float64')

                # ax[row, col].plot(data.T, color = 'black', alpha = .1)
                ax[row, col].plot(freqs,plot)

                # ax[row, col].plot(freqs, med )
                # ax[row, col].fill_between(x =freqs, y1 = inf, y2 = sup, alpha = .5 )


                # plt.show()
    plt.show()


def plot_backup_spectrum_compare(spectrum_method = 'somno'):
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


def plot_spectrum_compare_day_night(spectrum_method = 'somno', period = 'light'):
    control_list = get_mice('Control')
    DCR_list = get_mice('DCR-HCRT')
    groups = {'Control' : control_list, 'DCR-HCRT' : DCR_list}
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_B2533.nc')
    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
    freqs = np.round(freqs, 2).tolist()

    indices = []
    for state in ['w', 't', 'n', 'r']:
        for session in ['bl1', 'bl2', 'sd', 'r1' ]:
            indices+=([i + '_'+session+ '_'+state for i in control_list+ DCR_list])
    df_spectrum = pd.DataFrame(index = indices, columns = ['group', 'session', 'state', 'ref_values'] + freqs)



    for group in groups:
        for mouse in groups[group]:
            ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
            freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
            freqs = np.round(freqs, 2)

            ZT = ds.coords['times_somno'].values
            print('collecting relative spectrums ', mouse)

            states = ['w', 't', 'n', 'r']
            conditions = ['bl1', 'bl2', 'sd', 'r1']

            for condition in conditions:
                ref_values, relative_spectrums = get_relative_spectrums_one_mouse_one_condition_day_night(mouse, condition, spectrum_method, period )
                for state in states :
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 'state'] = state
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 'session'] = condition
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 'group'] = group
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 'ref_values'] = ref_values
                    df_spectrum.at['{}_{}_{}'.format(mouse,condition, state), 4:] = relative_spectrums[state]

        # df_spectrum = df_spectrum.dropna()
    fig,ax = plt.subplots(nrows = 4, ncols= 4, sharex = True, sharey = True)
    fig.suptitle('Spectrum per state -- ' + period)

    fig_diff, ax_diff = plt.subplots(nrows = 4, ncols= 4, sharex = True, sharey = True)
    fig_diff.suptitle('Diff DCR - ctrl')

    for i in range(4):
        ax[i,0].set_ylabel(conditions[i])
        ax_diff[i,0].set_ylabel(conditions[i])

    for row, session in enumerate(conditions):
        for col, state in enumerate(states):
            if state == 'a':
                continue
            # print(df_spectrum[(df_spectrum.session == session)& (df_spectrum.state == state)])
            data_dcr = df_spectrum[(df_spectrum.session == session) & (df_spectrum.group == 'DCR-HCRT') & (df_spectrum.state == state)]
            data_ctrl = df_spectrum[(df_spectrum.session == session) & (df_spectrum.group == 'Control') & (df_spectrum.state == state)]
            print(data_dcr)
            print(data_dcr.columns)
            print(freqs)
            data_dcr = data_dcr.dropna()
            data_ctrl = data_ctrl.dropna()
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
    for group in groups:
        dirname = excel_dir + '/{}/power_spectrum/'.format(group)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for session in conditions:
            for state in states:
                name = 'spectrum_{}_{}_{}_{}.xlsx'.format(spectrum_method,state, session,period)
                df = df_spectrum[(df_spectrum.group == group) & (df_spectrum.session == session) & (df_spectrum.state==state)]
                df.to_excel(dirname+name)
    for group in ['Control', 'DCR-HCRT']:
        for row, session in enumerate(conditions):
            for col, state in enumerate(states):
                if group == 'Control' and state == 'a':
                    continue
                print(group, session, state)
                data = df_spectrum[(df_spectrum.session == session) & (df_spectrum.group == group) & (df_spectrum.state == state)]
                data = data.dropna()
                data = (data[freqs]).values

                print(data.shape)
                plot = np.mean(data, axis=0)
                # inf, med, sup = np.percentile(data, [25,50,75], axis=0)
                # inf = np.array(inf, dtype = 'float64')
                # sup = np.array(sup, dtype = 'float64')
                # med = np.array(med, dtype = 'float64')

                ax[row, col].plot(freqs,plot)

                # ax[row, col].plot(freqs, med )
                # ax[row, col].fill_between(x =freqs, y1 = inf, y2 = sup, alpha = .5 )


                # plt.show()
    # plt.show()
def plot_spectrum_by_mouse(spectrum_method = 'somno'):
    control_list = get_mice('Control')
    DCR_list = get_mice('DCR-HCRT')
    groups = {'Control' : control_list, 'DCR-HCRT' : DCR_list}

    for group in groups:
        for mouse in groups[group]:
            ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
            freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
            print(mouse)
            period_color = {'dark' : 'black', 'light': 'grey'}
            if group == 'Control':
                states = ['w', 't', 'n', 'r']
            elif group == 'DCR-HCRT':
                states = ['w', 't', 'n', 'r', 'a']
            conditions = ['bl1', 'bl2', 'sd', 'r1']
            fig, ax = plt.subplots(nrows = 4, ncols= 5, sharex = True)
            ax[0,0].set_title('wake')
            ax[0,1].set_title('tdw')
            ax[0,2].set_title('nrem')
            ax[0,3].set_title('rem')
            ax[0,4].set_title('cata')
            ax[0,0].set_ylabel('bl1')
            ax[1,0].set_ylabel('bl2')
            ax[2,0].set_ylabel('sd')
            ax[3,0].set_ylabel('r1')

            fig.suptitle(mouse + ' -- ' + spectrum_method + ' -- ' + group)
            for period in ['dark', 'light']:
                for col, condition in enumerate(conditions):
                    ref_values, relative_spectrums = get_relative_spectrums_one_mouse_one_condition_day_night(mouse, condition, spectrum_method, period )
                    for row, state in enumerate(states) :
                        ax[col,row].plot(freqs, relative_spectrums[state], color = period_color[period])
                        ax[col,row].set_xlim(0,15)
            dirname = work_dir+'/pyFig/{}/power_spectrum/{}/'.format(group, spectrum_method)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            plt.savefig(dirname + mouse+'.png')
def plot_spectrum_cataplexy_day_night(spectrum_method = 'somno', period = 'light'):
    DCR_list = get_mice('DCR-HCRT')
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_B2533.nc')
    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values.tolist()
    freqs = np.round(freqs, 2).tolist()

    df_spectrum = pd.DataFrame(columns = ['group','session','state', 'ref_values'] + freqs)

    for mouse in DCR_list:
        ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
        freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
        freqs = np.round(freqs, 2)

        ZT = ds.coords['times_somno'].values
        print('collecting relative spectrums ', mouse)
        conditions = ['bl1', 'bl2', 'sd', 'r1']
        for condition in conditions:
            ref_values, relative_spectrums = get_relative_spectrums_one_mouse_one_condition_day_night(mouse, condition, spectrum_method, period )
            df_spectrum.at['{}_{}'.format(mouse,condition), 'group'] = group
            df_spectrum.at['{}_{}'.format(mouse,condition), 'session'] = condition
            df_spectrum.at['{}_{}'.format(mouse,condition), 'state'] = 'a'
            df_spectrum.at['{}_{}'.format(mouse,condition), 'ref_values'] = ref_values
            df_spectrum.at['{}_{}'.format(mouse,condition), freqs] = relative_spectrums['a']


        # df_spectrum = df_spectrum.dropna()
    fig,ax = plt.subplots(nrows = 4, sharex = True, sharey = True)
    fig.suptitle('Spectrum per state -- ' + period)


    for i in range(4):
        ax[i].set_ylabel(conditions[i])

    dirname = excel_dir + '/DCR-HCRT/power_spectrum/'.format(group)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for session in conditions:
            name = 'spectrum_{}_a_{}_{}.xlsx'.format(spectrum_method, session,period)
            df = df_spectrum[(df_spectrum.session == session)]
            df.to_excel(dirname+name)

    for row, session in enumerate(conditions):
        data = df_spectrum[(df_spectrum.session == session)]
        data = data.dropna()
        data = (data[freqs]).values
        plot = np.mean(data, axis=0)
        ax[row].plot(freqs,plot)



            # plt.show()
    plt.show()


def theta_peak_frequency(spectrum_method = 'welch', state = 'r'):
    periods = ['dark', 'light']
    conditions = ['bl1', 'bl2', 'sd', 'r1']
    control_list = get_mice('Control')
    DCR_list = get_mice('DCR-HCRT')
    if state == 'r':
        groups = {'Control' : control_list, 'DCR-HCRT' : DCR_list}
    elif state == 'a':
        groups = {'DCR-HCRT' : DCR_list}
    for period in periods :
        for condition in conditions:
            for group in groups:
                df = pd.DataFrame(np.zeros((len(groups[group]),1)), index = groups[group], columns = ['TPF'])
                for mouse in groups[group]:
                    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
                    freqs = ds.coords['freqs_{}'.format(spectrum_method)].values
                    fake_freqs = np.arange(freqs.size)*.25
                    _, relative_spectrums = get_relative_spectrums_one_mouse_one_condition_day_night(mouse, condition, spectrum_method, period )
                    spec = relative_spectrums[state]
                    if spec[np.isnan(spec)==0].size >0 :
                        TPF = fake_freqs[spec == np.max(spec)]
                    else :
                        TPF = np.nan
                    df.at[mouse, 'TPF'] = TPF
                print(df)

                dirname = excel_dir + '/{}/TPF/{}/{}/{}/'.format(group, condition, period, state)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                df.to_excel(dirname + 'TPF_{}_{}_{}_{}.xlsx'.format(group, condition, period, state))

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
    # mouse  ='B2762'
    # mouse  ='B2763'
    # mouse  ='B3072'
    # mouse  ='B3140'
    # mouse  ='B3513'
    # mouse = 'B4977'
    mouse = 'B2767'
    # group = 'Control'
    # rec = 'b2'
    # compute_all()
    # store_scoring_and_spectrums_one_mouse_one_session_day_(group, mouse)
    # get_clear_spectral_score_one_mouse_one_condition(mouse, 'bl1', 'r')
    # get_relative_spectrums_one_mouse_one_condition(mouse, 'bl1')

    # plot_spectrum_compare_global(spectrum_method = 'somno')
    # plot_spectrum_compare_day_night(spectrum_method = 'somno', period = 'dark')
    # plot_spectrum_compare_day_night(spectrum_method = 'somno', period = 'dark')
    plot_spectrum_compare_day_night(spectrum_method = 'welch', period = 'light')
    plt.show()
    plot_spectrum_compare_day_night(spectrum_method = 'welch', period = 'dark')
    plt.show()
    # plot_spectrum_cataplexy_day_night(spectrum_method = 'welch',period ='light')
    # plot_spectrum_cataplexy_day_night(spectrum_method = 'welch',period ='dark')
    # get_clear_spectral_score_one_mouse_one_condition_day_night(mouse, 'bl1', 'r', 'dark')
    # get_relative_spectrums_one_mouse_one_condition_day_night(mouse = mouse, condition ='bl1', spectrum_method = 'somno', period = 'dark')
    # plot_spectrum_by_mouse('welch')
    # plt.show()

    # theta_peak_frequency(state='r')
    # theta_peak_frequency(state = 'a')
