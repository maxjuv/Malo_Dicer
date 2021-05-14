from configuration import *
from select_mice_cata_Malo import get_mice


def get_microwake(mouse, max_epochs_criterium):
    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600
    score = ds['score'].values
    score_behavior = score.copy()
    for f, n in zip(['1', '2', '3'], ['w', 'n', 'r']) :
        score_behavior = np.where(score_behavior == f, n, score_behavior)
    # print('Remaining possible score : ', np.unique(score_behavior))



    wake = score_behavior == 'w'
    micro_wake = np.zeros(wake.size)

    counter = 0
    bouts = []
    positions = []
    for pos,w in enumerate(wake):
        if w ==0:
            bouts.append(counter)
            if counter >0 and counter <= max_epochs_criterium:
                # print(positions)
                micro_wake[positions] = np.array(counter*[1])
            counter = 0
            positions = []
        if w ==1:
            positions.append(pos)
            counter +=1
    return micro_wake

def plot_hypnogram_one_mouse(mouse, time_by_line = 12):
    ctrl = get_mice('Control')
    dcr = get_mice('DCR-HCRT')
    group_mice = {'Control' : ctrl, 'DCR-HCRT':dcr}
    if mouse in group_mice['Control'] :
        group = 'Control'
    elif mouse in group_mice['DCR-HCRT'] :
        group = 'DCR-HCRT'

    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600
    score = ds['score'].values
    score_behavior = score.copy()
    # group_color = {'Control':'black', 'DCR-HCRT':'seagreen'}
    group_color = {'Control':'black', 'DCR-HCRT':'black'}
    for f, n in zip(['1', '2', '3'], ['w', 'n', 'r']) :
        score_behavior = np.where(score_behavior == f, n, score_behavior)
    print('Remaining possible score : ', np.unique(score_behavior))
    height_a = 4
    height_r = 1
    height_n = 2
    height_w = 3

    a = (score_behavior == 'a')*height_a
    r = (score_behavior == 'r')*height_r
    n = (score_behavior == 'n')*height_n
    w = (score_behavior == 'w')*height_w

    hypno = a + r +n + w
    fig, ax = plt.subplots(figsize=(20,10))
    h = time_by_line
    loop = int(96/h)
    for i in range(loop+1):
        t1, t2 = h*i, h*(i+1)
        mask = (times > t1) & (times<t2)
        ax.plot(times[mask] - h*i, 3.5*i + hypno[mask], color = group_color[group])
    # ax.set_xlim(0,4)
    fig.suptitle('{} -- a = {}, r = {}, n = {}, w = {}'.format(mouse, height_a, height_r, height_n, height_w) )
    return fig
def save_all_hypnogram(time_by_line = 12):
    control_list = get_mice('Control')
    DCR_list = get_mice('DCR-HCRT')
    groups = {'Control' : control_list, 'DCR-HCRT' : DCR_list}
    for group in groups:
        dirname = work_dir+'/pyFig/{}/hypnogram/'.format(group)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for mouse in groups[group]:
            fig = plot_hypnogram_one_mouse(mouse, time_by_line = time_by_line)
            plt.savefig(dirname + mouse+'.png')




def plot_hypnogram_one_mouse_no_microwake(mouse, time_by_line = 12, max_epochs_criterium=2):
    ctrl = get_mice('Control')
    dcr = get_mice('DCR-HCRT')
    group_mice = {'Control' : ctrl, 'DCR-HCRT':dcr}
    if mouse in group_mice['Control'] :
        group = 'Control'
    elif mouse in group_mice['DCR-HCRT'] :
        group = 'DCR-HCRT'

    ds = xr.open_dataset(precompute_dir + '/spectrums/spectrum_scoring_{}.nc'.format(mouse))
    times = ds.coords['times_somno'].values/3600
    score = ds['score'].values

    score_behavior = score.copy()
    # group_color = {'Control':'black', 'DCR-HCRT':'seagreen'}
    group_color = {'Control':'black', 'DCR-HCRT':'black'}
    for f, n in zip(['1', '2', '3'], ['w', 'n', 'r']) :
        score_behavior = np.where(score_behavior == f, n, score_behavior)
    print('Remaining possible score : ', np.unique(score_behavior))
    microwake = get_microwake(mouse = mouse, max_epochs_criterium = max_epochs_criterium)
    mask_microwake = microwake==1
    score_behavior[mask_microwake] = np.array(np.sum(mask_microwake)*['n'])
    height_a = 4
    height_r = 1
    height_n = 2
    height_w = 3

    a = (score_behavior == 'a')*height_a
    r = (score_behavior == 'r')*height_r
    n = (score_behavior == 'n')*height_n
    w = (score_behavior == 'w')*height_w

    hypno = a + r +n + w
    fig, ax = plt.subplots(figsize=(20,10))
    h = time_by_line
    loop = int(96/h)
    for i in range(loop+1):
        t1, t2 = h*i, h*(i+1)
        mask = (times > t1) & (times<t2)
        ax.plot(times[mask] - h*i, 3.5*i + hypno[mask], color = group_color[group])
    # ax.set_xlim(0,4)
    fig.suptitle('{} -- a = {}, r = {}, n = {}, w = {}'.format(mouse, height_a, height_r, height_n, height_w) )
    return fig
def save_all_hypnogram_no_microwake(time_by_line = 12, max_epochs_criterium = 2):
    control_list = get_mice('Control')
    DCR_list = get_mice('DCR-HCRT')
    groups = {'Control' : control_list, 'DCR-HCRT' : DCR_list}
    for group in groups:
        dirname = work_dir+'/pyFig/{}/hypnogram_no_microwake/'.format(group)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for mouse in groups[group]:
            fig = plot_hypnogram_one_mouse_no_microwake(mouse, time_by_line = time_by_line, max_epochs_criterium=max_epochs_criterium)
            plt.savefig(dirname + mouse+'.png')


if __name__ == '__main__':

    mouse = 'B2700'
    # mouse = 'B2761'
    # get_microwake(mouse, 2)
    # plot_hypnogram_one_mouse(mouse,time_by_line = 12)
    # plot_hypnogram_one_mouse_no_microwake(mouse,time_by_line = 12,max_epochs_criterium=2)
    # plt.show()
    # save_all_hypnogram()
    save_all_hypnogram_no_microwake(max_epochs_criterium=3)
