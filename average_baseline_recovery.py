from configuration import *
from select_mice_cata_Malo import get_mice

def average_baseline_spectrum(spectrum_method = 'somno'):
    for group in ['Control', 'DCR-HCRT']:
        dirname = excel_dir + '{}/power_spectrum'.format(group)
        if group == 'Control':
            states = ['w', 'n', 'r']
        elif group == 'DCR-HCRT':
            states = ['w', 'n', 'r', 'a']
        for state in states :
            period = 'light'
            print(dirname+'/spectrum_{}_{}_bl1_{}.xlsx'.format(spectrum_method, state, period))
            df1 = pd.read_excel(dirname+'/spectrum_{}_{}_bl1_{}.xlsx'.format(spectrum_method, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/spectrum_{}_{}_bl2_{}.xlsx'.format(spectrum_method, state, period), index_col =0)
            mice1 = df1.index
            mice2 = df2.index
            freqs = df1.columns[4:].to_list()
            df = pd.DataFrame(index = get_mice(group), columns = freqs )
            for ind in range(mice1.size) :
                mouse1 = mice1[ind]
                mouse2 = mice2[ind]
                mouse = mouse1[:5]
                values1 = df1.loc[mouse1, freqs].values
                values2 = df2.loc[mouse2, freqs].values
                if sum(np.isnan(np.array(values1, dtype = 'float64'))) == values1.size  or sum(np.isnan(np.array(values2, dtype = 'float64'))) == values2.size :
                    weighted_mean = np.nansum((values1,values2), axis = 0)
                else :
                    weighted_mean = (12/24)*values1 + (12/24)*values2
                df.loc[mouse,freqs] = weighted_mean
            df.to_excel(dirname + '/spectrum_{}_{}_{}_baseline_{}.xlsx'.format(group, spectrum_method, state, period))
            #
            period = 'dark'
            df1 = pd.read_excel(dirname+'/spectrum_{}_{}_bl1_{}.xlsx'.format(spectrum_method, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/spectrum_{}_{}_bl2_{}.xlsx'.format(spectrum_method, state, period), index_col =0)
            mice1 = df1.index
            mice2 = df2.index
            freqs = df1.columns[4:].to_list()
            df = pd.DataFrame(index = get_mice(group), columns = ['ref_values'] + freqs )
            for ind in range(mice1.size) :
                mouse1 = mice1[ind]
                mouse2 = mice2[ind]
                mouse = mouse1[:5]
                values1 = df1.loc[mouse1, freqs].values
                values2 = df2.loc[mouse2, freqs].values
                if sum(np.isnan(np.array(values1, dtype = 'float64'))) == values1.size  or sum(np.isnan(np.array(values2, dtype = 'float64'))) == values2.size :
                    weighted_mean = np.nansum((values1,values2), axis = 0)
                else :
                    weighted_mean = (12/24)*values1 + (12/24)*values2
                df.loc[mouse,freqs] = weighted_mean
            df.to_excel(dirname + '/spectrum_{}_{}_{}_baseline_{}.xlsx'.format(group, spectrum_method, state, period))

def average_recovery_spectrum(spectrum_method = 'somno'):
    for group in ['Control', 'DCR-HCRT']:
        dirname = excel_dir + '{}/power_spectrum'.format(group)
        if group == 'Control':
            states = ['w', 'n', 'r']
        elif group == 'DCR-HCRT':
            states = ['w', 'n', 'r', 'a']
        for state in states :
            period = 'light'
            df1 = pd.read_excel(dirname+'/spectrum_{}_{}_sd_{}.xlsx'.format(spectrum_method, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/spectrum_{}_{}_r1_{}.xlsx'.format(spectrum_method, state, period), index_col =0)
            mice1 = df1.index
            mice2 = df2.index
            freqs = df1.columns[4:].to_list()
            df = pd.DataFrame(index = get_mice(group), columns = freqs )
            for ind in range(mice1.size) :
                mouse1 = mice1[ind]
                mouse2 = mice2[ind]
                mouse = mouse1[:5]
                values1 = df1.loc[mouse1, freqs].values
                values2 = df2.loc[mouse2, freqs].values
                if sum(np.isnan(np.array(values1, dtype = 'float64'))) == values1.size  or sum(np.isnan(np.array(values2, dtype = 'float64'))) == values2.size :
                    weighted_mean = np.nansum((values1,values2), axis = 0)
                else :
                    weighted_mean = (6/18)*values1 + (12/18)*values2
                df.loc[mouse,freqs] = weighted_mean
            df.to_excel(dirname + '/spectrum_{}_{}_{}_recovery_{}.xlsx'.format(group, spectrum_method, state, period))
            #
            period = 'dark'
            df1 = pd.read_excel(dirname+'/spectrum_{}_{}_sd_{}.xlsx'.format(spectrum_method, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/spectrum_{}_{}_r1_{}.xlsx'.format(spectrum_method, state, period), index_col =0)
            mice1 = df1.index
            mice2 = df2.index
            freqs = df1.columns[4:].to_list()
            df = pd.DataFrame(index = get_mice(group), columns =  freqs )
            for ind in range(mice1.size) :
                mouse1 = mice1[ind]
                mouse2 = mice2[ind]
                mouse = mouse1[:5]
                values1 = df1.loc[mouse1, freqs].values
                values2 = df2.loc[mouse2, freqs].values
                if sum(np.isnan(np.array(values1, dtype = 'float64'))) == values1.size  or sum(np.isnan(np.array(values2, dtype = 'float64'))) == values2.size :
                    weighted_mean = np.nansum((values1,values2), axis = 0)
                else :
                    weighted_mean = (12/24)*values1 + (12/24)*values2
                df.loc[mouse,freqs] = weighted_mean
            df.to_excel(dirname + '/spectrum_{}_{}_{}_recovery_{}.xlsx'.format(group, spectrum_method, state, period))
            #
            #

def average_baseline_fragmentation():
    encoding_to_state = {'a':'cata', 'w':'wake', 'n':'nrem', 'r':'rem'}
    for group in ['Control', 'DCR-HCRT']:
        dirname = excel_dir + '{}/sleep_fragmentation'.format(group)
        if group == 'Control':
            encoding = ['w', 'n', 'r']
        elif group == 'DCR-HCRT':
            encoding = ['w', 'n', 'r', 'a']
        for e in encoding :
            state = encoding_to_state[e]
            period = 'light'
            df1 = pd.read_excel(dirname+'/{}/{}_{}1.xlsx'.format(state, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/{}/{}_{}2.xlsx'.format(state, state, period), index_col =0)
            df = pd.DataFrame(index = df1.index, columns = df1.columns)
            mouse = df1.index.to_list()
            bouts = df1.columns.to_list()
            for mouse in mouse :
                values1 = df1.loc[mouse, bouts].values
                values2 = df2.loc[mouse, bouts].values
                weighted_mean = (12/24)*values1 + (12/24)*values2
                df.loc[mouse, bouts] = weighted_mean
            df.to_excel(dirname+'/{}/{}_baseline_{}.xlsx'.format(state, state, period))
            period = 'dark'
            df1 = pd.read_excel(dirname+'/{}/{}_{}1.xlsx'.format(state, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/{}/{}_{}2.xlsx'.format(state, state, period), index_col =0)
            df = pd.DataFrame(index = df1.index, columns = df1.columns)
            mouse = df1.index.to_list()
            bouts = df1.columns.to_list()
            for mouse in mouse :
                values1 = df1.loc[mouse, bouts].values
                values2 = df2.loc[mouse, bouts].values
                weighted_mean = (12/24)*values1 + (12/24)*values2
                df.loc[mouse, bouts] = weighted_mean
            df.to_excel(dirname+'/{}/{}_baseline_{}.xlsx'.format(state, state, period))

def average_recovery_fragmentation():
    encoding_to_state = {'a':'cata', 'w':'wake', 'n':'nrem', 'r':'rem'}
    for group in ['Control', 'DCR-HCRT']:
        dirname = excel_dir + '{}/sleep_fragmentation'.format(group)
        if group == 'Control':
            encoding = ['w', 'n', 'r']
        elif group == 'DCR-HCRT':
            encoding = ['w', 'n', 'r', 'a']
        for e in encoding :
            state = encoding_to_state[e]
            period = 'light'
            df1 = pd.read_excel(dirname+'/{}/{}_{}3.xlsx'.format(state, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/{}/{}_{}4.xlsx'.format(state, state, period), index_col =0)
            df = pd.DataFrame(index = df1.index, columns = df1.columns)
            mouse = df1.index.to_list()
            bouts = df1.columns.to_list()
            for mouse in mouse :
                values1 = df1.loc[mouse, bouts].values
                values2 = df2.loc[mouse, bouts].values
                weighted_mean = (6/18)*values1 + (12/18)*values2
                df.loc[mouse, bouts] = weighted_mean
            df.to_excel(dirname+'/{}/{}_recovery_{}.xlsx'.format(state, state, period))
            period = 'dark'
            df1 = pd.read_excel(dirname+'/{}/{}_{}3.xlsx'.format(state, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/{}/{}_{}4.xlsx'.format(state, state, period), index_col =0)
            df = pd.DataFrame(index = df1.index, columns = df1.columns)
            mouse = df1.index.to_list()
            bouts = df1.columns.to_list()
            for mouse in mouse :
                values1 = df1.loc[mouse, bouts].values
                values2 = df2.loc[mouse, bouts].values
                weighted_mean = (12/24)*values1 + (12/24)*values2

                df.loc[mouse, bouts] = weighted_mean
            df.to_excel(dirname+'/{}/{}_recovery_{}.xlsx'.format(state, state, period))



def average_baseline_fragmentation_mean():
    encoding_to_state = {'a':'cata', 'w':'wake', 'n':'nrem', 'r':'rem'}
    for group in ['Control', 'DCR-HCRT']:
        dirname = excel_dir + '{}/sleep_fragmentation_mean'.format(group)
        if group == 'Control':
            encoding = ['w', 'n', 'r']
        elif group == 'DCR-HCRT':
            encoding = ['w', 'n', 'r', 'a']
        for e in encoding :
            state = encoding_to_state[e]
            period = 'light'
            df1 = pd.read_excel(dirname+'/{}/{}_{}1.xlsx'.format(state, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/{}/{}_{}2.xlsx'.format(state, state, period), index_col =0)
            df = pd.DataFrame(index = df1.index, columns = df1.columns)
            mouse = df1.index.to_list()
            for mouse in mouse :
                values1 = df1.loc[mouse, 'mean_duration']
                values2 = df2.loc[mouse, 'mean_duration']
                weighted_mean = (12/24)*values1 + (12/24)*values2
                df.loc[mouse, 'mean_duration'] = weighted_mean
            df.to_excel(dirname+'/{}/{}_baseline_{}.xlsx'.format(state, state, period))
            period = 'dark'
            df1 = pd.read_excel(dirname+'/{}/{}_{}1.xlsx'.format(state, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/{}/{}_{}2.xlsx'.format(state, state, period), index_col =0)
            df = pd.DataFrame(index = df1.index, columns = df1.columns)
            mouse = df1.index.to_list()
            for mouse in mouse :
                values1 = df1.loc[mouse, 'mean_duration']
                values2 = df2.loc[mouse, 'mean_duration']
                weighted_mean = (12/24)*values1 + (12/24)*values2
                df.loc[mouse, 'mean_duration'] = weighted_mean
            df.to_excel(dirname+'/{}/{}_baseline_{}.xlsx'.format(state, state, period))

def average_recovery_fragmentation_mean():
    encoding_to_state = {'a':'cata', 'w':'wake', 'n':'nrem', 'r':'rem'}
    for group in ['Control', 'DCR-HCRT']:
        dirname = excel_dir + '{}/sleep_fragmentation_mean'.format(group)
        if group == 'Control':
            encoding = ['w', 'n', 'r']
        elif group == 'DCR-HCRT':
            encoding = ['w', 'n', 'r', 'a']
        for e in encoding :
            state = encoding_to_state[e]
            period = 'light'
            df1 = pd.read_excel(dirname+'/{}/{}_{}3.xlsx'.format(state, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/{}/{}_{}4.xlsx'.format(state, state, period), index_col =0)
            df = pd.DataFrame(index = df1.index, columns = df1.columns)
            mouse = df1.index.to_list()
            for mouse in mouse :
                values1 = df1.loc[mouse, 'mean_duration']
                values2 = df2.loc[mouse, 'mean_duration']
                weighted_mean = (6/18)*values1 + (12/18)*values2
                df.loc[mouse, 'mean_duration'] = weighted_mean
            df.to_excel(dirname+'/{}/{}_recovery_{}.xlsx'.format(state, state, period))
            period = 'dark'
            df1 = pd.read_excel(dirname+'/{}/{}_{}3.xlsx'.format(state, state, period), index_col =0)
            df2 = pd.read_excel(dirname+'/{}/{}_{}4.xlsx'.format(state, state, period), index_col =0)
            df = pd.DataFrame(index = df1.index, columns = df1.columns)
            mouse = df1.index.to_list()
            for mouse in mouse :
                values1 = df1.loc[mouse, 'mean_duration']
                values2 = df2.loc[mouse, 'mean_duration']
                weighted_mean = (12/24)*values1 + (12/24)*values2

                df.loc[mouse, 'mean_duration'] = weighted_mean
            df.to_excel(dirname+'/{}/{}_recovery_{}.xlsx'.format(state, state, period))

def fusion_baseline_recovery_REM_latency():
    for m, merging in enumerate(['Baseline', 'Recovery']):
        day1, day2 = 2*m+1, 2*(m+1)
        for group in ['Control', 'DCR-HCRT']:
            dirname = excel_dir + '/{}/REM_latency/'.format(group)
            for period in ['dark', 'light']:
                df1 = pd.read_excel(dirname+'/REM_latency_{}_{}{}.xlsx'.format(group, period,day1), index_col =0)
                df2 = pd.read_excel(dirname+'/REM_latency_{}_{}{}.xlsx'.format(group, period,day2), index_col =0)
                df = pd.DataFrame(index = df1.index, columns = np.arange(4))
                mice = np.unique(df1.index.to_list()+df2.index.to_list())
                for mouse in mice:
                    list1 = df1.loc[mouse,:]
                    list2 = df2.loc[mouse,:]
                    list1 = list1[~np.isnan(list1)].tolist()
                    list2 = list2[~np.isnan(list2)].tolist()
                    latencies = list1 + list2
                    if len(df.columns.to_list())<len(latencies):
                        df = df.reindex(columns = np.arange(len(latencies)))
                    df.loc[mouse, np.arange(len(latencies))] = np.array(latencies)
                filename = dirname + 'REM_latency_{}_{}_{}.xlsx'.format(merging, group, period)
                df.to_excel(filename)



if __name__ == '__main__':
    # average_baseline_spectrum('welch')
    # average_recovery_spectrum('welch')
    # average_baseline_fragmentation()
    # average_recovery_fragmentation()
    # average_baseline_fragmentation_mean()
    # average_recovery_fragmentation_mean()
    fusion_baseline_recovery_REM_latency()
