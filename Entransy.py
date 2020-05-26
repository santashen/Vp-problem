# V-P problem entransy optimization
# Author: Shen Yang
# Institution: School of Aerospace Engineering, Tsinghua University

import numpy as np
import seaborn as sns
from numba import jit
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import cv2

import re
import os
import pickle

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

plt.style.use("science")


class Entrosy:
    def __init__(
        self,
        n=40,
        S=200,
        low_conduct=0.29,
        ke=10,
        Tw=273.15,
        fill_rate=0.15,
        omega=1.9,
        initial_temp=300,
        whether_save_lambda=False,
        whether_save_temp=False,
        save_directory=None,
        fps=14,
    ):
        self.n = n
        self.dx = 0.2 / n
        self.S = S
        self.low_conduct = low_conduct
        self.high_conduct = ke * low_conduct
        self.Tw = Tw
        self.fill_fate = fill_rate
        self.fill_num = int(n * int(n / 2) * fill_rate)
        self.omega = omega
        self.l_real = np.ones((n, int(n / 2))) * low_conduct
        self.initial_temp = initial_temp
        self.T = np.ones((n, int(n / 2))) * Tw
        self.save_lambda_times = 0
        self.save_temp_times = 0
        self.whether_save_lambda = whether_save_lambda
        self.whether_save_temp = whether_save_temp
        self.save_directory = save_directory
        self.fps = fps
        self.high_binary_list = []

    @staticmethod
    @jit(nopython=True)
    def inner_sor(l_array, n, omega, initial_temp, Tw, S):
        omgea = omega
        Tw = Tw
        S = S
        dx = 0.2 / n
        source_item = S * dx ** 2
        T = np.ones((n, int(n / 2))) * initial_temp
        T_temp = np.zeros(T.shape)
        times = 0
        aE = 2 * l_array[:, :-1] * l_array[:, 1:] / (l_array[:, :-1] + l_array[:, 1:])
        aN = 2 * l_array[:-1, :] * l_array[1:, :] / (l_array[:-1, :] + l_array[1:, :])
        error = np.max(np.abs(T - T_temp))
        error_temp = 0
        left_col = int(n / 2 - n / 20)
        middle_col = int(n / 2 - 1)
        while np.abs(error - error_temp) > 1e-9:
            # for p in range(1):
            error_temp = error
            T_temp = T.copy()
            # 底部第一行扫描
            T[-1, 0] = (
                omega
                * (aE[-1, 0] * T_temp[-1, 1] + aN[-1, 0] * T_temp[-2, 0] + source_item)
                / (aE[-1, 0] + aN[-1, 0])
                + (1 - omega) * T_temp[-1, 0]
            )
            for j in range(1, left_col):
                T[-1, j] = (
                    omega
                    * (
                        aE[-1, j] * T_temp[-1, j + 1]
                        + aE[-1, j - 1] * T[-1, j - 1]
                        + aN[-1, j] * T_temp[-2, j]
                        + source_item
                    )
                    / (aN[-1, j] + aE[-1, j - 1] + aE[-1, j])
                    + (1 - omega) * T_temp[-1, j]
                )
            for j in range(left_col, middle_col):
                T[-1, j] = (
                    omega
                    * (
                        aE[-1, j] * T_temp[-1, j + 1]
                        + aE[-1, j - 1] * T[-1, j - 1]
                        + aN[-1, j] * T_temp[-2, j]
                        + 2 * l_array[-1, j] * Tw
                        + source_item
                    )
                    / (aN[-1, j] + aE[-1, j - 1] + aE[-1, j] + 2 * l_array[-1, j])
                    + (1 - omega) * T_temp[-1, j]
                )
            T[-1, middle_col] = (
                omega
                * (
                    l_array[-1, middle_col] * T_temp[-1, middle_col]
                    + aE[-1, middle_col - 1] * T[-1, middle_col - 1]
                    + aN[-1, middle_col] * T_temp[-2, middle_col]
                    + 2 * l_array[-1, middle_col] * Tw
                    + source_item
                )
                / (
                    l_array[-1, middle_col]
                    + aE[-1, middle_col - 1]
                    + aN[-1, middle_col]
                    + 2 * l_array[-1, middle_col]
                )
                + (1 - omega) * T_temp[-1, middle_col]
            )
            # 倒数第二行到顶部第二行扫描
            for i in range(-2, -n, -1):
                T[i, 0] = (
                    omega
                    * (
                        aE[i, 0] * T_temp[i, 1]
                        + aN[i, 0] * T_temp[i - 1, 0]
                        + aN[i + 1, 0] * T[i + 1, 0]
                        + source_item
                    )
                    / (aE[i, 0] + aN[i, 0] + aN[i + 1, 0])
                    + (1 - omega) * T_temp[i, 0]
                )
                for j in range(1, middle_col):
                    T[i, j] = (
                        omgea
                        * (
                            aE[i, j] * T_temp[i, j + 1]
                            + aE[i, j - 1] * T[i, j - 1]
                            + aN[i, j] * T_temp[i - 1, j]
                            + aN[i + 1, j] * T[i + 1, j]
                            + source_item
                        )
                        / (aE[i, j] + aE[i, j - 1] + aN[i, j] + aN[i + 1, j])
                        + (1 - omega) * T_temp[i, j]
                    )
                T[i, middle_col] = (
                    omega
                    * (
                        aE[i, middle_col - 1] * T[i, middle_col - 1]
                        + l_array[i, middle_col] * T_temp[i, middle_col]
                        + aN[i, middle_col] * T_temp[i - 1, middle_col]
                        + aN[i + 1, middle_col] * T[i + 1, middle_col]
                        + source_item
                    )
                    / (
                        aE[i, middle_col - 1]
                        + l_array[i, middle_col]
                        + aN[i, middle_col]
                        + aN[i + 1, middle_col]
                    )
                    + (1 - omega) * T_temp[i, -1]
                )
            # 顶部第一行扫描
            T[0, 0] = (
                omega
                * (aE[0, 0] * T_temp[0, 1] + aN[0, 0] * T[1, 0] + source_item)
                / (aE[0, 0] + aN[0, 0])
                + (1 - omega) * T_temp[0, 0]
            )
            for j in range(1, middle_col):
                T[0, j] = (
                    omega
                    * (
                        aE[0, j] * T_temp[0, j + 1]
                        + aE[0, j - 1] * T[0, j - 1]
                        + aN[0, j] * T[1, j]
                        + source_item
                    )
                    / (aE[0, j] + aE[0, j - 1] + aN[0, j])
                    + (1 - omega) * T_temp[0, j]
                )
            T[0, -1] = (
                omega
                * (
                    aE[0, -1] * T[0, -2]
                    + aN[0, -1] * T[1, -1]
                    + l_array[0, -1] * T_temp[0, -1]
                    + source_item
                )
                / (aE[0, -1] + aN[0, -1] + l_array[0, -1])
                + (1 - omega) * T_temp[0, -1]
            )
            times = times + 1
            error = np.max(np.abs(np.abs(T - T_temp)))
        return T

    def sor(self, l_array):
        l_array = l_array
        n = self.n
        omega = self.omega
        initial_temp = self.initial_temp
        Tw = self.Tw
        S = self.S
        return self.inner_sor(l_array, n, omega, initial_temp, Tw, S)

    def lambda_field(self):
        sns.heatmap(
            np.concatenate((self.l_real, self.l_real[:, ::-1]), axis=1),
            cmap="hot_r",
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            square=True,
            linewidth=0.2,
            linecolor="black",
        )

    def temp_field(self):
        x = np.linspace(-1, 1, self.n)
        y = np.linspace(-1, 1, self.n)
        X, Y = np.meshgrid(x, y)
        T = np.concatenate((self.T, self.T[:, ::-1]), axis=1)
        plt.contour(
            X, Y, T[::-1, :], levels=20, colors="grey", linewidths=0.5,
        )
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.xticks([])
        plt.yticks([])

    def save_lambda(self):
        self.save_lambda_times += 1
        self.lambda_field()
        plt.savefig(
            os.path.join(
                self.save_directory, "lambda{}.png".format(self.save_lambda_times)
            )
        )
        plt.clf()

    def save_temp(self):
        self.save_temp_times += 1
        self.temp_field()
        plt.savefig(
            os.path.join(self.save_directory, "temp{}.png".format(self.save_temp_times))
        )
        plt.clf()

    def make_lambda_gif(self):
        pngfiles = [
            os.path.join(self.save_directory, "{}".format(img))
            for img in os.listdir(self.save_directory)
            if img.startswith("lambda") and img.endswith("png")
        ]
        vediowriter = cv2.VideoWriter(
            os.path.join(self.save_directory, "lambda.mp4"),
            fourcc=cv2.VideoWriter_fourcc('H','2','6','4'),
            fps=14,
            frameSize=cv2.imread(pngfiles[0]).shape[:2],
        )
        pngfiles.sort(key=lambda file: int(re.search("\d+", file).group()))
        for img in pngfiles:
            vediowriter.write(cv2.imread(img))
        vediowriter.release()

    def merge(self):
        lambdas = [
            os.path.join(self.save_directory, "{}".format(img))
            for img in os.listdir(self.save_directory)
            if img.startswith("lambda") and img.endswith("png")
        ]
        lambdas.sort(key=lambda file: int(re.search("\d+", file).group()))
        temps = [
            os.path.join(self.save_directory, "{}".format(img))
            for img in os.listdir(self.save_directory)
            if img.startswith("temp") and img.endswith("png")
        ]
        temps.sort(key=lambda file: int(re.search("\d+", file).group()))
        for i, (lam, tem) in enumerate(zip(lambdas, temps)):
            a = cv2.imread(lam)
            b = cv2.imread(tem)
            x = cv2.addWeighted(a, 0.3, b, 0.7, 30)
            cv2.imwrite(os.path.join(self.save_directory, "merge{}.png".format(i)), x)

    def make_merge_gif(self):
        pngfiles = [
            os.path.join(self.save_directory, "{}".format(img))
            for img in os.listdir(self.save_directory)
            if img.startswith("merge") and img.endswith("png")
        ]
        vediowriter = cv2.VideoWriter(
            os.path.join(self.save_directory, "temp.mp4"),
            fourcc=cv2.VideoWriter_fourcc('H','2','6','4'),
            fps=14,
            frameSize=cv2.imread(pngfiles[0]).shape[:2],
        )
        pngfiles.sort(key=lambda file: int(re.search("\d+", file).group()))
        for img in pngfiles:
            vediowriter.write(cv2.imread(img))
        vediowriter.release()

    def save_obj(self):
        with open(os.path.join(self.save_directory, "case.pkl"), "wb") as f:
            pickle.dump(self, f)

    def heat(self, t, l_real):
        index1 = int(self.n / 2 - 1)
        gradient = np.gradient(t, self.dx, self.dx)
        square = np.square(gradient)
        flow = np.sqrt(square[0] + square[1])
        return flow

    def ave_temp_difference(self, t, l_real):
        t_positive = np.average(t)
        index1 = int(self.n / 2 - self.n / 20)
        index2 = int(self.n / 2)
        t_negative = np.sum(
            [
                2 * l_real[-1, wq] * (t[-1, wq] - self.Tw) * t[-1, wq]
                for wq in range(index1, index2)
            ]
        ) / (self.S * 0.2 * 0.1)

        return t_positive - t_negative

    def ave_resistence(self, t, l_real):
        return self.ave_temp_difference(t, l_real) / (self.S * 0.2 * 0.1)

    # 寻优
    def optimize(self):
        self.high_binary_list = []
        l_binary = np.ones(self.n * int(self.n / 2)) * self.low_conduct
        l_real = l_binary.reshape(self.n, int(self.n / 2))
        for i in tqdm(range(self.fill_num)):
            # 这一步会在已有的分布上新填充一个高导热材料
            self.high_binary_list = list(self.high_binary_list)
            T = self.sor(l_real)
            flow = self.heat(T, l_real)
            order = np.argsort(flow, axis=None)[::-1]
            for max_heat_position in order:
                if np.isin(max_heat_position, self.high_binary_list):
                    pass
                else:
                    self.high_binary_list.append(max_heat_position)
                    break
            l_binary[max_heat_position] = self.high_conduct
            l_real = l_binary.reshape(self.n, int(self.n / 2))

            resistence = self.ave_resistence(self.sor(l_real), l_real)
            self.high_binary_list = np.array(self.high_binary_list)
            T = self.sor(l_real)
            if self.whether_save_lambda == True:
                self.l_real = l_real
                self.save_lambda()
            if self.whether_save_temp == True:
                self.T = T
                self.save_temp()
            # 这里结束了填充，需要对已有的高导热材料分布进行剪枝
        l_real_temp = np.zeros((self.n, int(self.n / 2)))
        times = 0
        # 在while循环里实现对高导热材料填充的剪枝
        while not np.all(l_real_temp == l_real):
            print("在剪枝中哦..不要着急(∪｡∪)｡｡｡zzz,已经剪枝{}次了".format(times))
            l_real_temp = l_real.copy()
            resistence_addition = []
            for p in self.high_binary_list:
                l_binary[p] = self.low_conduct
                l_real = l_binary.reshape(self.n, int(self.n / 2))
                resistence_new = self.ave_resistence(self.sor(l_real), l_real)
                resistence_addition.append(resistence_new - resistence)
                l_binary[p] = self.high_conduct
            resistence_addition = np.array(resistence_addition)
            # 上面的块里实现了：分别取掉现有的高导热材料节点，计算平均热阻的增加
            pos = np.where(resistence_addition == np.min(resistence_addition))[0][0]
            min_binary_position = self.high_binary_list[pos]
            self.high_binary_list[pos] = 10000
            l_binary[min_binary_position] = self.low_conduct
            l_real = l_binary.reshape((self.n, int(self.n / 2)))
            # 所以找到resistence_addition
            # 中最小元素的下标，就对应high_binary数组中对应下标的节点的位置

            # 此时已经把最小热阻增加的那个节点的高导热材料换成低导热材料了，需要重新计算热流，分配节点
            T = self.sor(l_real)
            flow = self.heat(T, l_real)
            order = np.argsort(flow, axis=None)[::-1]
            for max_heat_position in order:
                if np.isin(max_heat_position, self.high_binary_list):
                    pass
                else:
                    self.high_binary_list[pos] = max_heat_position
                    break
            # 此时并没有向high_binary_list中增添新的元素，而只是替换掉了拿走的元素
            l_binary[max_heat_position] = self.high_conduct
            l_real = l_binary.reshape(self.n, int(self.n / 2))
            if self.whether_save_lambda == True:
                self.l_real = l_real
                self.save_lambda()
            if self.whether_save_temp == True:
                self.T = self.sor(self.l_real)
                self.save_temp()
            times += 1
        if self.whether_save_lambda == True:
            self.make_lambda_gif()
        if self.whether_save_temp == True:
            self.merge()
            self.make_merge_gif()
        self.l_real = l_real
        self.T = self.sor(self.l_real)