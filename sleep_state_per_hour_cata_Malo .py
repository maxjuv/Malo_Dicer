from configuration_Malo import *
import os
os.system('pwd')
os.system('which python')
os.system('python --version')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from select_mice_cata_Malo import get_mice

# data_dir = 'C:/Users/maxime.juventin/Desktop/scripts_ML/data/'
# data_dir = "/Volumes/FBM/DP/GROUPS/RESEARCH/MT/PRIVATE/common/Marie-Laure/DICER ANALYSIS/scripts_cataplexy"
# precompute_dir = "/Volumes/FBM/DP/GROUPS/RESEARCH/MT/PRIVATE/common/Marie-Laure/DICER ANALYSIS/scripts_cataplexy/data/"
# excel_dir = "/Volumes/FBM/DP/GROUPS/RESEARCH/MT/PRIVATE/common/Marie-Laure/DICER ANALYSIS/scripts_cataplexy/excels/"


    #
    # RNAs = os.listdir(data_dir)
    # groups = ['control', 'hcrt_cre']
    # print(RNAs, groups)
    # for RNA in RNAs :
    #     for group in groups :
    #         print(RNA, group)
    #         precompute_sleep_state_by_epoch(RNA, group)
    #         sleep_state_statistics(RNA, group)
    #         sleep_bouts(RNA, group)

def compute_all():
    groups = ['Control', 'DCR-HCRT']
    print(groups)
    for group in groups :
        print(group)
        precompute_sleep_state_by_epoch(group)
        sleep_state_statistics(group)
        sleep_bouts(group)

def precompute_sleep_state_by_epoch(group):

    files = get_mice(group)
    mice = np.unique([str(y.split('.')[0][:-2]) for y in files]).tolist()
    print(mice)
    # days = np.unique([str(y.split('.')[0][-2:]) for y in files]).tolist()
    # print(days)
    days = [ 'b1', 'b2', 'sd', 'r1' ]
    # days = ['b0', 'b1', 'b2', 'sd']



    all_mice_all_days = {}
    numbers = ['1', '2', '3', '4' ,'5' ,'6']
    letters = ['w', 'n', 'r', 'w', 'n', 'r']
    for mouse in mice :
        data = []
        for day in days :
            print(data_dir  + group+ "/" + mouse + day + ".txt")

            data_per_day = np.loadtxt(data_dir  +"/"+ group+ "/" + mouse + day + ".txt", dtype = str)
            # print(data_per_day.size)
            for number, letter in zip (numbers, letters):
                data_per_day = np.where(data_per_day == number, letter, data_per_day)
            data.append(data_per_day)

        one_mouse = np.concatenate(data)
        # print(one_mouse.shape)
        all_mice_all_days[mouse] = one_mouse
    all_mice_all_days = pd.DataFrame.from_dict(all_mice_all_days)

    # all_mice_all_days = all_mice_all_days.T

    print('find w, n, r')
    all_mice_w_epoch = all_mice_all_days == 'w'
    all_mice_n_epoch = all_mice_all_days == 'n'
    all_mice_r_epoch = all_mice_all_days == 'r'
    all_mice_cata_epoch = all_mice_all_days == 'a'





    print('create Dataset')



    path = precompute_dir + '/' + group + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # ds.to_netcdf(path + 'summed_sleep_by_epoch.nc', mode='w')
    epochs = np.arange(all_mice_all_days.shape[0])
    # print(epochs)
    # print(type(mice))
    coords = {'mice':mice, 'epochs':epochs }
    ds = xr.Dataset(coords = coords)
    # print(all_mice_w_epoch)

    ds['wake_all_mice'] = xr.DataArray(all_mice_w_epoch*1, dims = ['epochs','mice'])
    ds['nrem_all_mice'] = xr.DataArray(all_mice_n_epoch*1, dims = ['epochs','mice'])
    ds['cata_all_mice'] = xr.DataArray(all_mice_cata_epoch*1, dims = ['epochs','mice'])

    ds['rem_all_mice'] = xr.DataArray(all_mice_r_epoch*1, dims = ['epochs','mice'])
    # print(ds)
    # exit()
    print('save')
    ds.to_netcdf(path + 'sleep_by_epoch.nc', mode='w')

    #
def sleep_state_statistics( group):
    path = precompute_dir+'/'+ group + '/sleep_by_epoch.nc'
    ds = xr.open_dataset(path)
    level = [0, 1]
    # print(ds)
    all_mice_w_epoch = ds['wake_all_mice'].to_pandas()
    all_mice_n_epoch = ds['nrem_all_mice'].to_pandas()
    all_mice_r_epoch = ds['rem_all_mice'].to_pandas()
    all_mice_cata_epoch = ds['cata_all_mice'].to_pandas()

    number_of_epochs = all_mice_w_epoch.shape[0]
    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    hours = np.arange(0, int(duration_in_hours),time)
    window = int(time * 3600 / epoch_duration)

    fake = np.zeros(number_of_epochs)

    all_mice_w_by_time = []
    all_mice_cata_by_time = []
    all_mice_r_by_time = []
    all_mice_n_by_time = []
    for hour in hours :
        i1, i2 = int(hour*window), int((hour+1)*window)
        # tarace[hour] = all_mice_w_epoch[i1:i2].sum(axis =0)
        all_mice_w_by_time.append(all_mice_w_epoch[i1:i2].sum(axis =0)*4/60)
        all_mice_r_by_time.append(all_mice_r_epoch[i1:i2].sum(axis =0)*4/60)
        all_mice_n_by_time.append(all_mice_n_epoch[i1:i2].sum(axis =0)*4/60)
        all_mice_cata_by_time.append(all_mice_cata_epoch[i1:i2].sum(axis =0)*4/60)

    all_mice_cata_by_time = pd.concat(all_mice_cata_by_time, axis = 1)
    all_mice_w_by_time = pd.concat(all_mice_w_by_time, axis = 1)
    all_mice_r_by_time = pd.concat(all_mice_r_by_time, axis = 1)
    all_mice_n_by_time = pd.concat(all_mice_n_by_time, axis = 1)

    if not os.path.exists(excel_dir+  '/'+ group + '/time_dynamics/'):
        os.makedirs(excel_dir+ '/'+ group + '/time_dynamics/')

    all_mice_w_by_time.to_excel(excel_dir+ '/'+ group + '/time_dynamics/'+ '/wake_event_by_mouse_by_hour.xlsx')
    all_mice_n_by_time.to_excel(excel_dir+ '/'+ group +  '/time_dynamics/'+ '/NREM_event_by_mouse_by_hour.xlsx')
    all_mice_r_by_time.to_excel(excel_dir+ '/'+ group +  '/time_dynamics/'+ '/REM_event_by_mouse_by_hour.xlsx')
    all_mice_cata_by_time.to_excel(excel_dir+ '/'+ group +  '/time_dynamics/'+ '/cata_event_by_mouse_by_hour.xlsx')




def plot_sleep_state_accross_time(group):
    # data_dir = 'C:/Users/maxime.juventin/Desktop/scripts_ML/data/'
    path = precompute_dir + '/' + group + '/sleep_by_epoch.nc'
    ds = xr.open_dataset(path)

    print(ds)
    number_of_epochs = ds['wake_all_mice'].shape[0]
    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    wake = ds['wake_all_mice'].values
    nrem = ds['nrem_all_mice'].values
    rem = ds['rem_all_mice'].values
    cata = ds['cata_all_mice'].values

    time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    window = int(time * 3600 / epoch_duration)
    wake_by_time = np.zeros(int(duration_in_hours/time))
    rem_by_time = np.zeros(int(duration_in_hours/time))
    cata_by_time = np.zeros(int(duration_in_hours/time))
    nrem_by_time = np.zeros(int(duration_in_hours/time))
    for h, i in enumerate(np.arange(0, number_of_epochs, window)):
        wake_by_time[h] = wake[i : i+window].sum()
        cata_by_time[h] = cata[i : i+window].sum()
        rem_by_time[h] = rem[i : i+window].sum()
        nrem_by_time[h] = nrem[i : i+window].sum()
    print(wake_by_time.shape)
    fig, ax = plt.subplots(nrows=4)
    times = np.arange(0, duration_in_hours,time)
    ax[0].plot(times, wake_by_time*epoch_duration/60, color = 'black', label = 'wake')
    ax[1].plot(times, rem_by_time*epoch_duration/60, color = 'red', label='rem')
    ax[2].plot(times, nrem_by_time*epoch_duration/60, color = 'blue', label = 'nrem')
    ax[3].plot(times, cata_by_time*epoch_duration/60, color = 'blue', label = 'cata')
    for i,j in zip([0,28-time,76-time], [4-time,52-time,100-time]):
        ax[0].axvspan(i,j , color = 'black', alpha = .3)
        ax[1].axvspan(i,j , color = 'black', alpha = .3)
        ax[2].axvspan(i,j , color = 'black', alpha = .3)
        ax[3].axvspan(i,j , color = 'black', alpha = .3)
    ax[0].set_title('Sleep state amount for '+ group + ' , sum per ' + str(time) + ' hours')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    plt.show()


def plot_compare_sleep_state_accross_time(control ='Control', test ='DCR_HCRT'):
    # data_dir = 'C:/Users/maxime.juventin/Desktop/scripts_ML/data/'
    path_control = precompute_dir +  '/' +control+'/sleep_by_epoch.nc'
    path_cre = precompute_dir + '/'+ test+'/sleep_by_epoch.nc'
    ds_control = xr.open_dataset(path_control)
    ds_cre = xr.open_dataset(path_cre)

    number_of_epochs = ds_cre['wake_all_mice'].shape[0]
    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    wake_control = ds_control['wake_all_mice'].to_pandas()
    nrem_control = ds_control['nrem_all_mice'].to_pandas()
    rem_control = ds_control['rem_all_mice'].to_pandas()
    cata_control = ds_control['cata_all_mice'].to_pandas()

    wake_cre = ds_cre['wake_all_mice'].to_pandas()
    nrem_cre = ds_cre['nrem_all_mice'].to_pandas()
    rem_cre = ds_cre['rem_all_mice'].to_pandas()
    cata_cre = ds_cre['cata_all_mice'].to_pandas()

    time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    window = int(time * 3600 / epoch_duration)
    wake_by_time_control = []
    rem_by_time_control = []
    nrem_by_time_control = []
    cata_by_time_control = []

    wake_by_time_cre = []
    rem_by_time_cre = []
    nrem_by_time_cre = []
    cata_by_time_cre = []

    for h, i in enumerate(np.arange(0, number_of_epochs, window)):

        wake_by_time_control.append(wake_control[i : i+window].sum())
        rem_by_time_control.append(rem_control[i : i+window].sum())
        nrem_by_time_control.append(nrem_control[i : i+window].sum())
        cata_by_time_control.append(cata_control[i : i+window].sum())

        wake_by_time_cre.append(wake_cre[i : i+window].sum())
        rem_by_time_cre.append(rem_cre[i : i+window].sum())
        nrem_by_time_cre.append(nrem_cre[i : i+window].sum())
        cata_by_time_cre.append(cata_cre[i : i+window].sum())
    wake_by_time_control = pd.concat(wake_by_time_control, axis = 1)
    cata_by_time_control = pd.concat(cata_by_time_control, axis = 1)
    rem_by_time_control = pd.concat(rem_by_time_control, axis = 1)
    nrem_by_time_control = pd.concat(nrem_by_time_control, axis = 1)
    wake_by_time_cre = pd.concat(wake_by_time_cre, axis = 1)
    rem_by_time_cre = pd.concat(rem_by_time_cre, axis = 1)
    nrem_by_time_cre = pd.concat(nrem_by_time_cre, axis = 1)
    cata_by_time_cre = pd.concat(cata_by_time_cre, axis = 1)

    # print(wake_by_time.shape)
    fig, ax = plt.subplots(nrows=3)
    times = np.arange(0, duration_in_hours)
    plot  = np.arange(3)
    controls = [wake_by_time_control, rem_by_time_control, nrem_by_time_control, cata_by_time_control]
    cres = [wake_by_time_cre, rem_by_time_cre, nrem_by_time_cre, cata_by_time_cre]
    # colors = [ 'black', 'red', 'blue']
    # colors = [ 'black', 'green', 'black', 'green', 'black', 'green']
    labels = ['control', 'control' ,'control', 'control', 'DCR', 'DCR', 'DCR' 'DCR']

    for i, data_control, data_cre, label in zip(plot, controls, cres,labels) :
        m = data_control.mean(axis=0)*epoch_duration/60
        s =data_control.std(axis=0)*epoch_duration/60
        ax[i].plot(times, m, color = 'black', label = label)
        ax[i].fill_between(times, m-s, m+s , color = 'black', alpha = .2,)
        m = data_cre.mean(axis=0)*epoch_duration/60
        s =data_cre.std(axis=0)*epoch_duration/60
        ax[i].plot(times, m, color = 'green', label = label)
        ax[i].fill_between(times, m-s, m+s , color = 'green', alpha = .2)


    nights = { 'dark0' : [0, 4],
                 'dark1' : [16, 28],
                 'dark2' : [40, 52],
                 'dark3':[64, 76],
                 'dark4':[88, 100]}

    for night in nights :
        h1, h2 = nights[night][0], nights[night][1]
        ax[0].axvspan(h1, h2 , color = 'black', alpha = .3)
        ax[1].axvspan(h1, h2 , color = 'black', alpha = .3)
        ax[2].axvspan(h1, h2 , color = 'black', alpha = .3)
        ax[3].axvspan(h1, h2 , color = 'black', alpha = .3)
    height0_1, height0_2 =ax[0].get_ylim()[1]*.95, ax[0].get_ylim()[1]*1.05
    height1_1, height1_2 = ax[1].get_ylim()[1]*.95, ax[1].get_ylim()[1]*1.05
    height2_1, height2_2 = ax[2].get_ylim()[1]*.95, ax[2].get_ylim()[1]*1.05
    height3_1, height3_2 = ax[3].get_ylim()[1]*.95, ax[3].get_ylim()[1]*1.05
    for x1, x2 in zip([0,28,76], [4,52,100]):
        lim = 110
        x1 += 5
        x2 +=5
        ax[0].axhspan(height0_1, height0_2, xmin =x1/lim, xmax = x2/lim, fill = False, edgecolor = 'black', hatch = '////')
        ax[1].axhspan(height1_1, height1_2, xmin =x1/lim, xmax = x2/lim, fill = False, edgecolor = 'black', hatch = '////')
        ax[2].axhspan(height2_1, height2_2, xmin =x1/lim, xmax = x2/lim, fill = False, edgecolor = 'black', hatch = '////')
        ax[2].axhspan(height3_1, height3_2, xmin =x1/lim, xmax = x2/lim, fill = False, edgecolor = 'black', hatch = '////')
    x1 = 52 + 5
    x2 = 58 +5
    ax[0].axhspan(height0_1, height0_2, xmin =x1/lim, xmax = x2/lim, color = 'red')
    ax[1].axhspan(height1_1, height1_2, xmin =x1/lim, xmax = x2/lim, color = 'red')
    ax[2].axhspan(height2_1, height2_2, xmin =x1/lim, xmax = x2/lim, color = 'red')
    ax[3].axhspan(height3_1, height3_2, xmin =x1/lim, xmax = x2/lim, color = 'red')
        # ax[0].axhspan(i,j , color = 'black', alpha = .3)
        # ax[1].axhspan(i,j , color = 'black', alpha = .3)
        # ax[2].axhspan(i,j , color = 'black', alpha = .3)
    ax[0].set_title('Sleep state amount DCR vs control for , sum per ' + str(time) + ' hours')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[0].set_ylabel('wake')
    ax[1].set_ylabel('rem')
    ax[2].set_ylabel('nrem')
    ax[3].set_ylabel('cata')

    plt.show()

def sleep_bouts(group):
    # B0 begins at 4h, B1 etc begins at 8
    path = precompute_dir + group + '/sleep_by_epoch.nc'
    print(precompute_dir)
    print(path)
    ds = xr.open_dataset(path)


    all_mice_w_epoch = ds['wake_all_mice'].to_pandas()
    all_mice_n_epoch = ds['nrem_all_mice'].to_pandas()
    all_mice_cata_epoch = ds['cata_all_mice'].to_pandas()
    all_mice_r_epoch = ds['rem_all_mice'].to_pandas()    #.unstack().T*1
    data_by_state = {'wake':all_mice_w_epoch, 'nrem' : all_mice_n_epoch, 'rem' :all_mice_r_epoch, 'cata':all_mice_cata_epoch}
    number_of_epochs = ds['wake_all_mice'].shape[0]
    # print(number_of_epochs)
    epoch_duration = 4
    duration_in_hours = int(number_of_epochs*epoch_duration/3600)
    # protocol = ['bl1', 'bl2', 'sd', 'sr']
    # hours = np.array(([4, 28], [28, 52], [52,76], [76, 100]) )
    # days_info = { 'bl1' : [4, 28], 'bl2' : [28, 52], 'sd' : [52,76], 'sr':[76, 100]}
    # day_cycles = { 'dark1' : [4, 4+8],
    #              'light1' : [4+8, 4+20],
    #              'dark2' : [4+20, 28+8],
    #              'light2' : [28+8, 28+20],
    #              'dark3' : [28+20, 52+8],
    #              'sd' : [52+8, 52+14],
    #              'light3' : [52+14,52+20],
    #              'dark4':[52+20, 76+8],
    #              'light4':[76+8, 76+20],
    #              'dark5':[76+20, 100]}
    day_cycles = {
                 'light1' : [0, 12],
                 'dark1' : [12, 24],
                 'light2' : [24, 36],
                 'dark2' : [36, 48],
                 'sd' : [48, 54],
                 'light3' : [54,60],
                 'dark3':[60, 72],
                 'light4':[72,84 ],
                 'dark4':[84, 96]}
    # time = int(1) # window to look at in HOUR. CAUTION must be a multiple of 100
    # window = int(time * 3600 / epoch_duration)
    # hours_in_epochs = hours*3600/epoch_duration
    # print(hours_in_epochs)
    mice = ds.coords['mice'].values
    # fig, ax = plt.subplots()

    # for mouse in mice :
    #     # print(mouse)
    #     plot = []
    #     timing = []
    #     for i in np.arange(0*window, number_of_epochs-window,window):
    #         w_one_mouse = all_mice_w_epoch[mouse].values
    #         plot.append(w_one_mouse[i:i+window].sum())
    #         timing.append(i/window)
    #     # ax.plot(timing, plot, label = mouse)



    # fig, ax = plt.subplots(nrows = 3, ncols = 10)
    for s, state in enumerate(data_by_state):
        data = data_by_state[state]


        for c, cycle in enumerate(day_cycles):
            cycle_hours = day_cycles[cycle]
            # print(cycle_hours)
            i1, i2 = int(cycle_hours[0]), int(cycle_hours[1])
            i1 = int(i1 *3600/epoch_duration)
            i2 = int(i2 *3600/epoch_duration)
            selected_data = data[i1: i2]
            # print(diff.where(diff == -1))
            all_mice_count = {}
            print('je suis la')
            for mouse in mice :
                one_and_zeros_one_mouse = selected_data[mouse].values
                counter = 0
                bouts = []
                for i in one_and_zeros_one_mouse:
                    if i ==0:
                        bouts.append(counter)
                        counter = 0
                    if i ==1:
                        counter +=1
                bouts.append(counter)
                bouts=np.array(bouts)
                mask = bouts!=0
                bouts = bouts[mask]

            # all_mice_count = {}
            # for mouse in mice :
            #     diff_one_mouse = diff[mouse].values
            #     ini = np.where(diff_one_mouse  == 1)[0]
            #     end = np.where(diff_one_mouse  == -1)[0]
            #     bouts = []
            #
            #     # print(len(ini), len(end))
            #     ######TODO iterate on data 1110000 and make count that starts when i==1
            #     # print(ini[0] > end[0], ini[0] , end[0])
            #     if len(ini) == 0 and len(end)==0 :
            #         bouts.append(0)
            #     elif len(ini) == 1 and len(end)==0 :
            #         bouts.append(diff_one_mouse.size-ini[0])
            #     elif len(ini) == 0 and len(end)==1 :
            #         bouts.append(end[0])
            #     elif len(ini) == 1 and len(end)==1 :
            #         if ini[0]<end[0]:
            #             bouts.append(end[0]-ini[0])
            #         else:
            #             bouts.append(end[0])
            #             bouts.append(diff_one_mouse.size-ini[0])
            #     # if 0 :
            #
            #     else :
            #         print(data[i1: i2][mouse].values)
            #         print(ini, end)
            #         if ini[0] > end[0] :
            #             first_bout = end[0]
            #             bouts.append(first_bout)
            #             end = np.delete(end, 0)
            #             # print(first_bout, 'first')
            #             # print(len(end))
            #         if ini[-1] > end[-1] :
            #             last_bout = diff_one_mouse.size - ini[-1]
            #             bouts.append(last_bout)
            #             ini = np.delete(ini, -1)
            #             # print(last_bout, 'last')
            #             # print(len(ini))
            #
            #         for i in np.arange(len(ini)):
            #             bout = end[i]-ini[i]
            #             bouts.append(bout)
                    # print(bouts)
                #one_mouse_bouts.append(bouts)
                #all_mice_bouts = np.concatenate(all_mice_bouts)*4
                bouts = bouts*4
                bins_sleep = [2**i for i in np.arange(2,12)]
                #bins_sleep += [max(bouts)]
                count, bins= np.histogram(bouts, bins = bins_sleep)
                all_mice_count[mouse] = count
                # fig, ax = plt.subplots()
                # ax.bar(bins[:-1], count)
                # fig.suptitle(state)
                # print(bouts)
                # print(sum(bouts>=1024), sum(bouts>=512))
                # print(sum(bouts==4))
                # plt.show()
            # print(all_mice_count)
            df = pd.DataFrame.from_dict(all_mice_count, orient = 'index', columns = bins[:-1] )
            output_path = excel_dir + '/'+ group + '/sleep_fragmentation/' + state + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            df.to_excel(output_path + state + '_' + cycle + '.xlsx')


            # if np.unique(all_mice_bouts).size > 1 :
                # sns.distplot(all_mice_bouts, color = 'black', ax = ax[s,c])
            # ax[s,c].hist(all_mice_bouts, 51, range = [0,50])

            #
            # fig, axx = plt.subplots(nrows=2)
            # axx[0].bar(bins[:-1], count, width = 10)
            # axx[1].bar(bins[:-1], count*bins[:-1], width = 10)
            # axx[0].set_title(state +' - ' +cycle)
            # plt.show()
            # ax[s,c].bar(bins[:-1], count, width=1, color = 'black')
            # #
            # # ax[s,c].set_ylim(0, 30)
            # ax[s,0].set_ylabel(state)
            # ax[0,c].set_title(cycle)

        # ax.legend()
        # ax.set_xlim(0.5, 100)
        # ax.set_ylim(0, 10)
        # fig.suptitle('Wake bouts during ' + cycle)






if __name__ == '__main__' :


    group = "DCR-HCRT"
    #group = "Control"

    # precompute_sleep_state_by_epoch(group)
    sleep_bouts(group)
    # sleep_state_statistics(group)
    # plot_sleep_state_accross_time(group)
    # plot_compare_sleep_state_accross_time(group)

    #compute_all()
    # sleep_bouts_all()
    # sleep_state_statistics_all()
    # plt.show()
