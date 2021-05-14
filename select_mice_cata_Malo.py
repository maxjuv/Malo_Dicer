from configuration import *
os.system('pwd')
os.system('which python')
os.system('python --version')



# def get_mice(group):
#     print(data_dir)
#     print(os.listdir(data_dir))
#     path = '{}{}/'.format(data_dir, group)
#     print(path)
#     files = os.listdir(path)
#     if '.DS_Store' in files:
#         files.remove('.DS_Store')
#
#     return files
#
#
# def get_mice_for_spectrum(group):
#     path = '{}{}/'.format(data_spectrum_dir, group)
#     files = os.listdir(path)
#     if '.DS_Store' in files:
#         files.remove('.DS_Store')
#     files = [f[4:] for f in files]
#     return files

def get_mice(group):
    df_excel = pd.read_excel(work_dir + 'datetime_reference_DICER.xls', index_col = 0)
    mylist = df_excel[df_excel.group == group].index.to_list()
    files = [i[4:] for i in mylist]

    return files

def get_mouse_info(mouse):
    df_excel = pd.read_excel(work_dir + 'datetime_reference_DICER.xls', index_col = 0)
    group = df_excel.at['MTA-'+mouse,'group']
    return group


if __name__ == '__main__':
    print(get_mice('Control'))

    # print(get_mice_for_spectrum('Control'))
    # print(get_mice_for_spectrum('DCR-HCRT'))
