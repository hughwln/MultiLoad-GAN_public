'''
Describe: Load and process the New River data set.
Author: Yi Hu
Email: yhu28@ncsu.edu
'''
import pandas as pd
import pickle5 as pickle
import os
from os import walk
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import kmeans_pytorch
from sys import platform

class newRiver():

    def __init__(self):
        self.smids = [['74684774', '74684775', '74685050', '74715007', '74715009', '74715075', '74715076', '74715078'],
                      ['74684542', '74715516', '74715517', '74715518', '74715739', '74715740', '74715741', '74715748'],
                      ['74713715', '74713716', '74713717', '74713718', '74713760', '74713771', '74713772', '74713773'],
                      ['74684959', '74684960', '74714511', '74714599', '74714600', '74714601', '74717163', '74717164'],
                      ['74714539', '74714540', '74714541', '74714542', '74714607', '74714608', '74714609', '74714610'],
                      ['74714580', '74714582', '74714774', '74716552', '74716596', '74716664', '74716665', '75044146'],
                      ['74715841', '74715842', '74716696', '74716697', '74716784', '74716785', '74716786', '74717216'],
                      ['74685466', '74685467', '74685483', '74685497', '74685521', '74685545', '74685546', '74685547']]
        self.dfs = [pd.DataFrame(columns=self.smids[0]),
                    pd.DataFrame(columns=self.smids[1]),
                    pd.DataFrame(columns=self.smids[2]),
                    pd.DataFrame(columns=self.smids[3]),
                    pd.DataFrame(columns=self.smids[4]),
                    pd.DataFrame(columns=self.smids[5]),
                    pd.DataFrame(columns=self.smids[6]),
                    pd.DataFrame(columns=self.smids[7])]

    def load_data(self):
        if platform == "linux" or platform == "linux2":
            filepath = '/home/yhu28/Downloads/BadSMsFixedv2/'
        elif platform == "darwin":
            filepath = 'C:\\Users\\yhu28\\Downloads\\BadSMsFixedv2\\'
        elif platform == "win32":
            filepath = 'C:\\Users\\yhu28\\Downloads\\BadSMsFixedv2\\'

        for (dirpath, dirnames, filenames) in walk(filepath):
            if 'figs' in dirpath:
                continue
            if 'probSMs' in dirpath:
                continue
            if 'removedSMsFixed' in dirpath:
                continue
            for file in filenames:
                if file[-3:] != 'pkl':
                    continue
                if file[3:11] in self.smids[0]:
                    txid = 0
                elif file[3:11] in self.smids[1]:
                    txid = 1
                elif file[3:11] in self.smids[2]:
                    txid = 2
                elif file[3:11] in self.smids[3]:
                    txid = 3
                elif file[3:11] in self.smids[4]:
                    txid = 4
                elif file[3:11] in self.smids[5]:
                    txid = 5
                elif file[3:11] in self.smids[6]:
                    txid = 6
                elif file[3:11] in self.smids[7]:
                    txid = 7
                else:
                    continue

                fullname = os.path.join(dirpath, file)
                with open(fullname, 'rb') as f:
                    data = pickle.load(f)
                    data = data.loc['2017-7-24':'2020-12-27']
                    self.dfs[txid][file[3:11]] = data['usage']

        for i in range(len(self.dfs)):
            self.dfs[i] = self.dfs[i].loc[:, self.dfs[i].mean().sort_values(ascending=True).index]
            self.dfs[i].columns = range(8)

    def get_random_seleted_group(self, times=2):
        time = {0: ['2017-7-24', '2019-12-29', '2019-12-30', '2020-12-20'],
                1: ['2017-7-25', '2019-12-30', '2019-12-31', '2020-12-21'],
                2: ['2017-7-26', '2019-12-31', '2020-1-1', '2020-12-22'],
                3: ['2017-7-27', '2020-1-1', '2020-1-2', '2020-12-23'],
                4: ['2017-7-28', '2020-1-2', '2020-1-3', '2020-12-24'],
                5: ['2017-7-29', '2020-1-3', '2020-1-4', '2020-12-25'],
                6: ['2017-7-30', '2020-1-4', '2020-1-5', '2020-12-26']}

        # training set
        dfs_training = [df.loc[time[0][0]: time[0][3]] for df in self.dfs]

        if platform == "linux" or platform == "linux2":
            t_path = '~/Documents/Code/Data/new_river/'
        elif platform == "darwin":
            t_path = 'C:\\Users\\yhu28\\Documents\\Code\\Data\\new_river\\'
        elif platform == "win32":
            t_path = 'C:\\Users\\yhu28\\Documents\\Code\\Data\\new_river\\'
        t_file = t_path + 'temp_Boone.csv'
        d = pd.read_csv(t_file)
        d.index = pd.to_datetime(d["Date"])
        d = d.loc[time[0][0]: time[0][3]]

        all_df = pd.concat(dfs_training, axis=1)
        datasetlist = []
        for i in range(times):
            sample = all_df.sample(n=8, axis='columns')
            sample.columns = range(8)
            sample = sample.loc[:, sample.mean().sort_values(ascending=True).index]
            sample.columns = range(8)
            sample['Temp'] = d['Temp'].astype(np.float32)
            datasetlist.append(sample)

        dataset = pd.concat(datasetlist)
        dataset = dataset.to_numpy(dtype=float)
        return dataset



    def groupDataset(self, day_offset=0):
        time = {0: ['2017-7-24', '2019-12-29', '2019-12-30', '2020-12-20'],
                1: ['2017-7-25', '2019-12-30', '2019-12-31', '2020-12-21'],
                2: ['2017-7-26', '2019-12-31', '2020-1-1', '2020-12-22'],
                3: ['2017-7-27', '2020-1-1', '2020-1-2', '2020-12-23'],
                4: ['2017-7-28', '2020-1-2', '2020-1-3', '2020-12-24'],
                5: ['2017-7-29', '2020-1-3', '2020-1-4', '2020-12-25'],
                6: ['2017-7-30', '2020-1-4', '2020-1-5', '2020-12-26']}

        if platform == "linux" or platform == "linux2":
            t_path = '~/Documents/Code/Data/new_river/'
        elif platform == "darwin":
            t_path = 'C:\\Users\\yhu28\\Documents\\Code\\Data\\new_river\\'
        elif platform == "win32":
            t_path = 'C:\\Users\\yhu28\\Documents\\Code\\Data\\new_river\\'

        if day_offset == 7:
            datas = []
            for day in range(7):
                # training set
                dfs_training = [df.loc[time[day][0]: time[day][3]] for df in self.dfs]

                t_file = t_path + 'temp_Boone.csv'
                d = pd.read_csv(t_file)
                d.index = pd.to_datetime(d["Date"])
                d = d.loc[time[day][0]: time[day][3]]

                for i in range(len(dfs_training)):
                    dfs_training[i] = dfs_training[i].loc[:, dfs_training[i].mean().sort_values(ascending=True).index]
                    dfs_training[i]['Temp'] = d['Temp'].astype(np.float32)

                data = pd.concat(dfs_training)
                datas.append(data)
            dataset = pd.concat(datas)

        else:
            # training set
            dfs_training = [df.loc[time[day_offset][0]: time[day_offset][3]] for df in self.dfs]

            t_file = t_path + 'temp_Boone.csv'
            d = pd.read_csv(t_file)
            d.index = pd.to_datetime(d["Date"])
            d = d.loc[time[day_offset][0]: time[day_offset][3]]

            for i in range(len(dfs_training)):
                dfs_training[i] = dfs_training[i].loc[:, dfs_training[i].mean().sort_values(ascending=True).index]
                dfs_training[i]['Temp'] = d['Temp'].astype(np.float32)

            dataset = pd.concat(dfs_training)

        if platform == "linux" or platform == "linux2":
            trainDataFile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/GANData' + str(day_offset) + '.csv'
        elif platform == "darwin":
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData' + str(day_offset) + '.csv'
        elif platform == "win32":
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\GANData' + str(day_offset) + '.csv'
        dataset.to_csv(trainDataFile)

    def buildNegativeSamples_bymean(self, dfs_training, week_set, num):
        classified_by_mean = [pd.DataFrame(),
                             pd.DataFrame(),
                             pd.DataFrame(),
                             pd.DataFrame(),
                             pd.DataFrame(),
                             pd.DataFrame()]

        week_mean = week_set.mean()
        upper = week_mean.max()
        # print(week_mean.max(), week_mean.min())

        # fig = plt.figure()
        # plt.hist(week_mean, bins=6)
        # plt.title('Positive sample mean distribution')
        # plt.show()

        for i in range(len(week_mean)):
            if week_mean.iloc[i] < upper / 6.0:
                classified_by_mean[0][len(classified_by_mean[0].columns)] = week_set.iloc[:, i]
            elif week_mean.iloc[i] < upper * 2.0 / 6.0:
                classified_by_mean[1][len(classified_by_mean[1].columns)] = week_set.iloc[:, i]
            elif week_mean.iloc[i] < upper * 3.0 / 6.0:
                classified_by_mean[2][len(classified_by_mean[2].columns)] = week_set.iloc[:, i]
            elif week_mean.iloc[i] < upper * 4.0 / 6.0:
                classified_by_mean[3][len(classified_by_mean[3].columns)] = week_set.iloc[:, i]
            elif week_mean.iloc[i] < upper * 5.0 / 6.0:
                classified_by_mean[4][len(classified_by_mean[4].columns)] = week_set.iloc[:, i]
            else:
                classified_by_mean[5][len(classified_by_mean[5].columns)] = week_set.iloc[:, i]

        # print(classified_by_mean)
        classified_by_mean_num = [len(df.columns) for df in classified_by_mean]
        max_num = max(classified_by_mean_num)
        max_idx = classified_by_mean_num.index(max_num)
        classified_by_mean_1 = classified_by_mean[max_idx]
        aaa = [classified_by_mean[i] for i in range(len(classified_by_mean)) if i!=max_idx]
        classified_by_mean_2 = pd.concat(aaa, axis=1)

        # temp = []
        for i in range(num):
            a = random.randint(0, 4)
            b = 8 - a
            curves = [classified_by_mean_1.sample(n=a, axis='columns'), classified_by_mean_2.sample(n=b, axis='columns')]
            group = pd.concat(curves, axis=1)
            group.columns = range(8)
            # if len(group.columns) != 8:
            #     print(len(group.columns))
            group = group.loc[:, group.mean().sort_values(ascending=True).index]
            if len(group.columns) == 8:
                group.columns = range(8)
            else:
                print("error!!! not 8 columns")
            group['label'] = pd.Series(0, index=group.index)
            dfs_training.append(group)

        # temp = pd.concat(temp, axis=1)
        # temp = temp.max()
        # fig = plt.figure()
        # plt.hist(temp, bins=6)
        # plt.title('Negative sample mean distribution')
        # plt.show()

    def buildNegativeSamples_bypeak(self, dfs_training, week_set, num):
        classified_by_peak = [pd.DataFrame(),
                             pd.DataFrame(),
                             pd.DataFrame(),
                             pd.DataFrame(),
                             pd.DataFrame(),
                             pd.DataFrame()]

        week_peak = week_set.max()

        upper = week_peak.max()
        # print(week_mean.max(), week_mean.min())

        # fig = plt.figure()
        # plt.hist(week_peak, bins=6)
        # plt.title('Positive sample mean distribution')
        # plt.show()

        for i in range(len(week_peak)):
            if week_peak.iloc[i] < upper / 6.0:
                classified_by_peak[0][len(classified_by_peak[0].columns)] = week_set.iloc[:, i]
            elif week_peak.iloc[i] < upper * 2.0 / 6.0:
                classified_by_peak[0][len(classified_by_peak[0].columns)] = week_set.iloc[:, i]
            elif week_peak.iloc[i] < upper * 3.0 / 6.0:
                classified_by_peak[0][len(classified_by_peak[0].columns)] = week_set.iloc[:, i]
            elif week_peak.iloc[i] < upper * 4.0 / 6.0:
                classified_by_peak[1][len(classified_by_peak[1].columns)] = week_set.iloc[:, i]
            elif week_peak.iloc[i] < upper * 5.0 / 6.0:
                classified_by_peak[1][len(classified_by_peak[1].columns)] = week_set.iloc[:, i]
            else:
                classified_by_peak[1][len(classified_by_peak[1].columns)] = week_set.iloc[:, i]

        # print(classified_by_mean)
        classified_by_peak_num = [len(df.columns) for df in classified_by_peak]
        max_num = max(classified_by_peak_num)
        max_idx = classified_by_peak_num.index(max_num)
        classified_by_peak_1 = classified_by_peak[max_idx]
        aaa = [classified_by_peak[i] for i in range(len(classified_by_peak)) if i!=max_idx]
        classified_by_peak_2 = pd.concat(aaa, axis=1)

        # temp = []
        for i in range(num):
            a = random.randint(0, 4)
            b = 8 - a
            curves = [classified_by_peak_1.sample(n=a, axis='columns'), classified_by_peak_2.sample(n=b, axis='columns')]
            group = pd.concat(curves, axis=1)
            group.columns = range(8)
            # if len(group.columns) != 8:
            #     print(len(group.columns))
            group = group.loc[:, group.mean().sort_values(ascending=True).index]
            if len(group.columns) == 8:
                group.columns = range(8)
            else:
                print("error!!! not 8 columns")
            group['label'] = pd.Series(0, index=group.index)
            dfs_training.append(group)

        # temp = pd.concat(temp, axis=1)
        # temp = temp.max()
        # fig = plt.figure()
        # plt.hist(temp, bins=6)
        # plt.title('Negative sample mean distribution')
        # plt.show()

    def buildNegativeSamples_bykmeans(self, dfs_training, week_set, num):
        num_clusters = 32
        classified_by_kmeans = []
        for i in range(num_clusters):
            classified_by_kmeans.append(pd.DataFrame())

        week_peak = week_set.max()
        # for i in range(len(week_set.columns)):
        #     week_set.iloc[:, i] = week_set.iloc[:, i] / week_peak.iloc[i]

        week_np = week_set.to_numpy()
        week_torch = torch.from_numpy(week_np)
        week_torch = torch.transpose(week_torch, dim0=0, dim1=1)


        # kmeans
        cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(
            X=week_torch, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')  # euclidean
        )

        # ======== load generated data=============
        single_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\singleGeneratedData.csv'
        group_file = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\groupGeneratedData_Mo.csv'

        single_data = pd.read_csv(single_file)
        group_data = pd.read_csv(group_file)

        week_single = []
        for n in range(len(single_data.index) // 672):
            week = single_data.iloc[n * 672:n * 672 + 672, 1:]
            week.index = range(len(week.index))
            week_single.append(week)
        df_single = pd.concat(week_single, axis=1)

        week_group = []
        for n in range(len(group_data.index) // 672):
            week = group_data.iloc[n * 672:n * 672 + 672, 1:]
            week.index = range(len(week.index))
            week_group.append(week)
        df_group = pd.concat(week_group, axis=1)

        x_group = torch.from_numpy(df_group.to_numpy())
        x_single = torch.from_numpy(df_single.to_numpy())

        x_single = torch.transpose(x_single, dim0=0, dim1=1)
        x_group = torch.transpose(x_group, dim0=0, dim1=1)

        group_ids = kmeans_pytorch.kmeans_predict(X=x_group, cluster_centers=cluster_centers, device=torch.device('cuda:0'))
        single_ids = kmeans_pytorch.kmeans_predict(X=x_single, cluster_centers=cluster_centers, device=torch.device('cuda:0'))
        group_ids_np = group_ids.numpy()
        single_ids_np = single_ids.numpy()
        # =========================================

        ids_np = cluster_ids_x.numpy()
        plt.hist(ids_np, bins=num_clusters, density=True, histtype='step', label='real', color='green')
        plt.hist(group_ids_np, bins=num_clusters, density=True, histtype='step', label='group', color='blue')
        plt.hist(single_ids_np, bins=num_clusters, density=True, histtype='step', label='single', color='red')
        plt.legend()
        plt.show()

        fig = plt.figure()
        for i in range(cluster_centers.shape[0]):
            plt.plot(cluster_centers[i], label='cluster'+str(i))
        plt.legend()
        plt.show()

        # print(week_mean.max(), week_mean.min())

        # fig = plt.figure()
        # plt.hist(week_peak, bins=6)
        # plt.title('Positive sample mean distribution')
        # plt.show()

        for i in range(len(cluster_ids_x)):
            group_idx = cluster_ids_x[i].item()
            classified_by_kmeans[group_idx][len(classified_by_kmeans[group_idx].columns)] = week_set.iloc[:, i]

        # print(classified_by_mean)
        classified_by_kmeans_num = [len(df.columns) for df in classified_by_kmeans]
        classified_by_kmeans_num_tensor = torch.tensor(classified_by_kmeans_num)
        sorted_tensor, indices = torch.sort(classified_by_kmeans_num_tensor)

        # aaa = classified_by_kmeans[indices[-3:]]
        aaa = [classified_by_kmeans[i] for i in range(len(classified_by_kmeans)) if i in indices[-3:]]
        classified_by_kmeans_1 = pd.concat(aaa, axis=1)
        bbb = [classified_by_kmeans[i] for i in range(len(classified_by_kmeans)) if i in indices[:-3]]
        classified_by_kmeans_2 = pd.concat(bbb, axis=1)

        # temp = []
        for i in range(num):
            a = random.randint(0, 3)
            # a = 0
            b = 8 - a

            curves = [classified_by_kmeans_1.sample(n=a, axis='columns'), classified_by_kmeans_2.sample(n=b, axis='columns')]
            group = pd.concat(curves, axis=1)
            group.columns = range(8)
            # if len(group.columns) != 8:
            #     print(len(group.columns))
            group = group.loc[:, group.mean().sort_values(ascending=True).index]
            if len(group.columns) == 8:
                group.columns = range(8)
            else:
                print("error!!! not 8 columns")
            group['label'] = pd.Series(0, index=group.index)
            dfs_training.append(group)

        # temp = pd.concat(temp, axis=1)
        # temp = temp.max()
        # fig = plt.figure()
        # plt.hist(temp, bins=6)
        # plt.title('Negative sample mean distribution')
        # plt.show()

    def nnDataset(self, day_offset=0, shuffle='origin'):
        time = {0: ['2017-7-24', '2019-12-29', '2019-12-30', '2020-12-20'],
                1: ['2017-7-25', '2019-12-30', '2019-12-31', '2020-12-21'],
                2: ['2017-7-26', '2019-12-31', '2020-1-1', '2020-12-22'],
                3: ['2017-7-27', '2020-1-1', '2020-1-2', '2020-12-23'],
                4: ['2017-7-28', '2020-1-2', '2020-1-3', '2020-12-24'],
                5: ['2017-7-29', '2020-1-3', '2020-1-4', '2020-12-25'],
                6: ['2017-7-30', '2020-1-4', '2020-1-5', '2020-12-26']}

        dfs_week = []

        # training set
        dfs_training = [df.loc[time[day_offset][0]: time[day_offset][3]] for df in self.dfs]
        for i in range(len(dfs_training)):
            # ============shuffle columns============
            if shuffle == 'random':
                dfs_training[i] = dfs_training[i].sample(n=8, axis='columns')
                dfs_training[i].columns = range(8)
            elif shuffle == 'sort':
                dfs_training[i] = dfs_training[i].loc[:, dfs_training[i].mean().sort_values(ascending=True).index]
                dfs_training[i].columns = range(8)
            # ============================================
            dfs_training[i]['label'] = pd.Series(1, index=dfs_training[i].index)

            for n in range(len(dfs_training[i].index) // 672):
                week = dfs_training[i].iloc[n*672:n*672+672, :-1]
                week.index = range(len(week.index))
                dfs_week.append(week)
        week_set = pd.concat(dfs_week, axis=1)

        # self.buildNegativeSamples_bykmeans(dfs_training, week_set, int(len(dfs_training[0].index) / 672 * 8 * 1))
        self.buildNegativeSamples_bymean(dfs_training, week_set, int(len(dfs_training[0].index) / 672 * 8 * 1))
        self.buildNegativeSamples_bypeak(dfs_training, week_set, int(len(dfs_training[0].index)/672 * 8 * 2))


        # unsupervised samples
        # for i in range(8*3):
        #     fake_group = pd.DataFrame(columns=range(8))
        #     randomlist = random.sample(range(0, 64), 8)
        #     print(randomlist)
        #     for c in range(8):
        #         # fake_group[c] = dfs_training[c].iloc[:, (i+c) % 8]
        #         fake_group[c] = dfs_training[randomlist[c] // 8].iloc[:, randomlist[c] % 8]
        #     # ==========shuffle columns=================
        #     if shuffle == 'random':
        #         fake_group = fake_group.sample(n=8, axis='columns')
        #         fake_group.columns = range(8)
        #     elif shuffle == 'sort':
        #         fake_group = fake_group.loc[:, fake_group.mean().sort_values(ascending=True).index]
        #         fake_group.columns = range(8)
        #     # ============================================
        #     fake_group['label'] = pd.Series(0, index=fake_group.index)
        #     dfs_training.append(fake_group)


        training_set = pd.concat(dfs_training)

        if shuffle == 'random':
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classiferRandom'+str(day_offset)+'.csv'
        elif shuffle == 'sort':
            if platform == "linux" or platform == "linux2":
                trainDataFile = '~/Documents/Code/Research/MultiLoad-GAN_public/dataset/newRiver/classifierSorted' + str(
                    day_offset) + '.csv'
            elif platform == "darwin":
                trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classiferSorted' + str(
                day_offset) + '.csv'
            elif platform == "win32":
                trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classiferSorted' + str(
                day_offset) + '.csv'
        elif shuffle == 'origin':
            trainDataFile = 'C:\\Users\\yhu28\\Documents\\Code\\Research\\MultiLoad-GAN_public\\dataset\\newRiver\\classiferOrigin' + str(
                day_offset) + '.csv'
        else:
            print('!!! unknown shuffle !!!')

        training_set.to_csv(trainDataFile)
        # testing_set.to_csv(testDataFile)

if __name__ == '__main__':
    dataloader = newRiver()
    dataloader.load_data()
    # dataloader.nnDataset(0, shuffle='sort')
    # for i in range(1, 7):
    #     dataloader.nnDataset(i, shuffle='sort')
    dataloader.groupDataset(day_offset=0)