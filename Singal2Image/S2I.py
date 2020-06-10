import glob
import os
import numpy as np

class Signal2Image:
    def __init__(self, filepath, max_ch=9):
        self.MAX_CH = max_ch
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.Acc_file_paths = glob.glob(os.path.join(base_dir, filepath, '*Acc.CSV'))
        self.Vel_file_paths = glob.glob(os.path.join(base_dir, filepath, '*Vel.CSV'))

        list.sort(self.Acc_file_paths)
        list.sort(self.Vel_file_paths)


        # print(Acc_file_paths[0])
        # print(Vel_file_paths[0])

    def extract_datas(self, category='Acc', by='Type'):
        if category == 'Acc':
            file_paths = np.array(self.Acc_file_paths)
        elif category == 'Vel':
            file_paths = np.array(self.Vel_file_paths)
        else:
            print('Unknown category')
            return None

        print(np.shape(file_paths))
        file_paths = file_paths.reshape(-1, self.MAX_CH)

        if by == 'Type':
            return file_paths
        elif by == 'Time':
            return np.transpose(file_paths)
        else:
            print('Unknown \'by\'')
            return None

        # print(np.transpose(self.Vel_file_paths)[0])
        # print(np.shape(self.Acc_file_paths))

a = Signal2Image('Datasets')
print(a.extract_datas()[0])