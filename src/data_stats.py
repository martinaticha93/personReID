import os

from datareader import SEQUENCE_LEN
from trainLSTM import LOCAL_MARS_EDGES_20


def number_of_videos_per_identities_MARS(data_path):
    data = {}
    identities = os.listdir(data_path)
    identities.sort()
    for identity in identities:
        data[identity] = 0
        for image in os.listdir(os.path.join(data_path, identity)):
            if image[11:15] == 'F001':
                data[identity] = data[identity] + 1
    return data


def number_of_videos_per_identities_MARS_selected(data_path):
    data = {}
    identities = os.listdir(data_path)
    identities.sort()
    for identity in identities:
        data[identity] = int(len(os.listdir(os.path.join(data_path, identity))) / SEQUENCE_LEN)
        if (data[identity]) == 3:
            stop = True
    return data


def compare_two_folders(data_path_first, data_path_second):
    files_first = os.listdir(data_path_first)
    files_first.sort()

    for file in files_first:
        list_first = os.listdir(os.path.join(data_path_first, file))
        list_second = os.listdir(os.path.join(data_path_second, file))

        files_not_in_first = [file for file in list_first if file not in list_second]
        files_not_in_second = [file for file in list_second if file not in list_first]

        if len(files_not_in_first) != 0 or len(files_not_in_second) != 0:
            print(f'files {file} do not equal')
        print(file)


if __name__ == '__main__':
    # compare_two_folders(LOCAL_MARS_EDGES_20, LOCAL_MARS_EDGES_POSTPRO_20)

    MARS_selected_stats = number_of_videos_per_identities_MARS_selected(LOCAL_MARS_EDGES_20)
    min_MARS_selected = min(MARS_selected_stats.values())
    max_MARS_selected = max(MARS_selected_stats.values())

    print("min num of videos for identity..." + str(min_MARS_selected))
    print("max num of videos for identity..." + str(max_MARS_selected))
