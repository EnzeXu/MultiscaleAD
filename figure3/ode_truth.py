import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import time
from datetime import datetime
import argparse
from scipy.integrate import odeint
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as path_effects
# from parameters import *
from config import Start, Config
from utils import MultiSubplotDraw

from const import *

"""
# Chen asked me to add these comments here - if you need :) 
CSF_CN counts=153.0 avg=[203.15882353  64.9379085   27.99281046]
CSF_SMC counts=78.0 avg=[204.64102564  63.8974359   35.9025641 ]
CSF_EMCI counts=41.0 avg=[190.7804878   79.9902439   34.88780488]
CSF_LMCI counts=108.0 avg=[184.77314815  87.7         34.23425926]
CSF_AD counts=307.0 avg=[144.74918567 119.13485342  47.93029316]
CSF counts: [153.  78.  41. 108. 307.]
PET-A counts: [81. 42. 88. 35. 19.]
PET-T counts: [78. 42. 83. 30. 32.]
PET-N counts: [92. 43. 80. 39. 11.]
"""


def get_now_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def numpy_safe_pow(x, n):
    return np.sign(x) * (np.abs(x)) ** n


def my_matmul(m, x, expand=160):
    x = np.ones(expand) * x[0]
    return np.mean(np.matmul(m, x)).reshape(1)


class ConstTruth:
    def __init__(self, **params):
        assert "csf_folder_path" in params and "pet_folder_path" in params, "please provide the save folder paths"
        assert "dataset" in params
        assert "start" in params
        assert "tcsf_scaler" in params
        self.params = params
        self.params["option"] = "option1"
        csf_folder_path, pet_folder_path = params["csf_folder_path"], params["pet_folder_path"]
        label_list = LABEL_LIST  # [[0, 2, 3, 4]]  # skip the second nodes (SMC)
        self.class_num = len(label_list)
        if "x" not in params:
            assert params["start"] in ["ranged", "fixed"]
            if params["start"] == "ranged":
                self.x_all = np.asarray([3, 6, 8, 11, 12]) #I made a change that make 9 to 8
            else:
                self.x_all = np.asarray([0, 3, 6, 8, 9])
        else:
            self.x_all = np.asarray(params.get("x"))
        self.y = dict()
        self.x = dict()

        # y0 = np.load("saves/y0.npy")

        self.lines = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]
        for one_line in self.lines:
            self.y[one_line] = []
            self.x[one_line] = self.x_all
        for i, class_name in enumerate(label_list):
            csf_data = np.load(os.path.join(csf_folder_path, "CSF_{}.npy".format(class_name)))
            pet_data_a = np.load(os.path.join(pet_folder_path, "PET-A_{}.npy".format(class_name)))
            pet_data_t = np.load(os.path.join(pet_folder_path, "PET-T_{}.npy".format(class_name)))
            pet_data_n = np.load(os.path.join(pet_folder_path, "PET-N_{}.npy".format(class_name)))
            self.y["APET"] = self.y["APET"] + [np.mean(pet_data_a)]
            self.y["TPET"] = self.y["TPET"] + [np.mean(pet_data_t)]
            self.y["NPET"] = self.y["NPET"] + [np.mean(pet_data_n)]
            # self.y["APET"] = self.y["APET"] + [y0[:160]]
            # self.y["TPET"] = self.y["TPET"] + [y0[160:320]]
            # self.y["NPET"] = self.y["NPET"] + [y0[320:480]]

            self.y["ACSF"] = self.y["ACSF"] + [csf_data[0]]
            self.y["TtCSF"] = self.y["TtCSF"] + [csf_data[1]]
            self.y["TpCSF"] = self.y["TpCSF"] + [csf_data[2]]
            self.y["TCSF"] = self.y["TCSF"] + [csf_data[1] - csf_data[2]]
        for one_key in self.lines:
            self.y[one_key] = np.asarray(self.y[one_key])
        self.y["NPET"] = 2.0 - self.y[
            "NPET"]  # 1.0 - (self.y["NPET"] - np.min(self.y["NPET"])) / (np.max(self.y["NPET"]) - np.min(self.y["NPET"]))
        assert params["dataset"] in ["all", "chosen_0"]
        if params["dataset"] == "chosen_0":
            for one_key in ["NPET"]:
                self.y[one_key] = self.y[one_key][[]]
                self.x[one_key] = self.x[one_key][[]]
            for one_key in ["ACSF", "TpCSF", "TCSF", "TtCSF"]:
                self.y[one_key] = self.y[one_key][[0, 2, 3, 4]]
                self.x[one_key] = self.x[one_key][[0, 2, 3, 4]]
        # else:
        #     for one_key in ["NPET"]:
        #         self.y[one_key] = self.y[one_key][[]]
        #         self.x[one_key] = self.x[one_key][[]]
        #     pass


class ConstTruthSpatial:
    def __init__(self, **params):
        assert "csf_folder_path" in params and "pet_folder_path" in params, "please provide the save folder paths"
        csf_folder_path, pet_folder_path = params["csf_folder_path"], params["pet_folder_path"]
        label_list = LABEL_LIST
        self.args = params["args"]
        self.params = dict()
        self.params["option"] = "option1"
        self.y = dict()
        self.x = dict()
        self.x_all = np.asarray([3.0, 6.0, 9.0, 11.0, 12.0]) #change 9 to
        # self.x_all = np.asarray([3.0])
        self.lines = ["APET", "TPET", "NPET"]
        for one_key in self.lines:
            self.y[one_key] = []
            self.x[one_key] = self.x_all
        for i, class_name in enumerate(label_list):
            pet_data_a = np.load(os.path.join(pet_folder_path, "PET-A_{}.npy".format(class_name)))
            pet_data_t = np.load(os.path.join(pet_folder_path, "PET-T_{}.npy".format(class_name)))
            pet_data_n = np.load(os.path.join(pet_folder_path, "PET-N_{}.npy".format(class_name)))
            self.y["APET"] = self.y["APET"] + [pet_data_a]
            self.y["TPET"] = self.y["TPET"] + [pet_data_t]
            self.y["NPET"] = self.y["NPET"] + [pet_data_n]
        for one_key in self.lines:
            self.y[one_key] = np.asarray(self.y[one_key])

        # self.y["NPET"] = 0.9 - (self.y["NPET"] - np.min(self.y["NPET"])) / (np.max(self.y["NPET"]) - np.min(self.y["NPET"])) * 0.1

        self.const_min_max_scale = {
            "APET": [0.35, 0.50],
            "TPET": [0.36, 0.51],
            "NPET": [0.55, 0.70],
        }


class ADSolver:
    def __init__(self, class_name, const_truth=None):
        self.n = 160  # Config.N_dim
        # self.L = Config().L
        #        self.t = np.linspace(0, 10 - 0.1, 100)
        self.T = 12.01
        self.T_unit = 0.01
        self.t = np.linspace(0.0, self.T - self.T_unit, int(self.T / self.T_unit))  # expand time
        # print(self.t[:10], "...", self.t[-10:])
        self.class_name = class_name
        self.const_truth = const_truth
        self.Ls = Config(self.const_truth.args.threshold).Ls
        self.L = None
        self.y0 = Start(class_name).all
        # print("ODE size: {}".format(self.y0.shape))

        self.lines = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]

        # print("output has {} curves".format(len(self.output)))
        self.output_names = ["$A_{PET}$", "$T_{PET}$", "$N_{PET}$", "$A_{CSF}$", "$T_{pCSF}$", "$T_{CSF}$",
                             "$T_{tCSF}$"]
        self.output_names_rest = ["$A_{m} Avg$", "$T_{m} Avg$", "$A_{o} Avg$", "$T_{o} Avg$", "$T_{p} Avg$"]
        self.colors = ["red", "green", "blue", "cyan", "orange", "purple", "brown", "gray", "olive"]
        self.y = None
        self.output = None
        self.output_spatial = None
        self.params = None
        self.starts_weights = None
        self.starts_weights_spatial = None
        self.diffusion_list = None
        self.tol = 1e-4
        # print("atol = rtol = {}".format(self.tol))

    def step(self, _params, _starts_weights, _diffusion_list):
        # if _params is not None:
        self.params = np.asarray(_params)
        self.starts_weights = np.asarray(_starts_weights)
        self.starts_weights_spatial = np.concatenate([
            np.repeat(self.starts_weights[0], 1 if i in [3, 8, 9] else self.n) for i in range(11)
        ])
        self.diffusion_list = _diffusion_list
        # else:
        # self.params = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
        # self.starts_weights = np.asarray([STARTS_WEIGHTS[i]["init"] for i in range(STARTS_NUM)])
        # print("Params & starts_weights are not given. Using the initial value instead to simulate ...")
        # self.y0 = self.y0 * self.starts_weights
        self.y0 = self.y0 * self.starts_weights_spatial
        self.y = odeint(self.pend, self.y0, self.t, rtol=self.tol, atol=self.tol)
        self.output = self.get_output()

    def step_spatial(self, _params, _starts_weights, _diffusion_list):
        # if _params is not None:
        self.params = np.asarray(_params)
        self.starts_weights = np.asarray(_starts_weights)
        self.starts_weights_spatial = np.concatenate([
            np.repeat(self.starts_weights[i], 1 if i in [3, 8, 9] else self.n) for i in range(11)
        ])
        self.diffusion_list = _diffusion_list
        # else:
        # self.params = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
        # self.starts_weights = np.asarray([STARTS_WEIGHTS[i]["init"] for i in range(STARTS_NUM)])
        # print("Params & starts_weights are not given. Using the initial value instead to simulate ...")
        # self.y0 = self.y0 * self.starts_weights
        self.y0 = self.y0 * self.starts_weights_spatial

        #debug 0619
        # Af_avg = np.mean(self.y0[self.n * 2: self.n * 3])
        # Tf_avg = np.mean(self.y0[self.n * 6 + 1: self.n * 7 + 1])
        # N_avg = np.mean(self.y0[self.n * 7 + 3: self.n * 8 + 3])
        # print(f"debug 0619 final start y0: {Af_avg}, {Tf_avg}, {N_avg}")
        # print(f"start weights: {self.starts_weights_spatial}")



        self.y = odeint(self.pend, self.y0, self.t, rtol=self.tol, atol=self.tol)
        self.output_spatial = self.get_output_spatial()

    def get_output_spatial(self):
        Af = self.y[:, self.n * 2: self.n * 3]
        Tf = self.y[:, self.n * 6 + 1: self.n * 7 + 1]
        N = self.y[:, self.n * 7 + 3: self.n * 8 + 3]
        N = 0.9 - (N - np.min(N)) / (np.max(N) - np.min(N)) * 0.1
        return [Af, Tf, N]

    def get_output(self):
        Am = self.y[:, 0: self.n]
        Ao = self.y[:, self.n: self.n * 2]
        Af = self.y[:, self.n * 2: self.n * 3]
        ACSF = self.y[:, self.n * 3: self.n * 3 + 1]
        Tm = self.y[:, self.n * 3 + 1: self.n * 4 + 1]
        Tp = self.y[:, self.n * 4 + 1: self.n * 5 + 1]
        To = self.y[:, self.n * 5 + 1: self.n * 6 + 1]
        Tf = self.y[:, self.n * 6 + 1: self.n * 7 + 1]
        TCSF = self.y[:, self.n * 7 + 1: self.n * 7 + 2]
        TpCSF = self.y[:, self.n * 7 + 2: self.n * 7 + 3]
        N = self.y[:, self.n * 7 + 3: self.n * 8 + 3]

        ACSF = np.expand_dims(ACSF[:, 0], axis=0)  # np.expand_dims(k_sA * np.sum(Am, axis=1), axis=0)
        TCSF = np.expand_dims(TCSF[:, 0], axis=0)  # np.expand_dims(k_sT * np.sum(Tm, axis=1), axis=0)
        TpCSF = np.expand_dims(TpCSF[:, 0], axis=0)  # np.expand_dims(k_sTp * np.sum(Tp, axis=1), axis=0)
        APET = np.expand_dims(np.mean(np.swapaxes(Af, 0, 1), axis=0), axis=0)
        TPET = np.expand_dims(np.mean(np.swapaxes(Tf, 0, 1), axis=0), axis=0)
        NPET = np.expand_dims(np.mean(np.swapaxes(N, 0, 1), axis=0), axis=0)

        NPET = 0.9 - (NPET - np.min(NPET)) / (np.max(NPET) - np.min(NPET)) * 0.1  # 200.0 - NPET


        TtCSF = TpCSF + TCSF

        Am_avg = np.expand_dims(np.mean(Am, axis=1), axis=0)
        Tm_avg = np.expand_dims(np.mean(Tm, axis=1), axis=0)
        Ao_avg = np.expand_dims(np.mean(Ao, axis=1), axis=0)
        To_avg = np.expand_dims(np.mean(To, axis=1), axis=0)
        Tp_avg = np.expand_dims(np.mean(Tp, axis=1), axis=0)

        # APET_average = np.expand_dims(np.mean(APET, axis=0), axis=0)
        # TPET_average = np.expand_dims(np.mean(TPET, axis=0), axis=0)
        # NPET_average = np.expand_dims(np.mean(NPET, axis=0), axis=0)
        # return [APET, TPET, NPET, ACSF, TpCSF, TCSF, TtCSF, Ao_sum, To_sum]
        return [APET, TPET, NPET, ACSF, TpCSF, TCSF, TtCSF, Am_avg, Tm_avg, Ao_avg, To_avg, Tp_avg]

    # def get_output_spatial(self):
    #     # Am = self.y[:, 0: self.n]
    #     # Ao = self.y[:, self.n: self.n * 2]
    #     Af = self.y[:, self.n * 2: self.n * 3]
    #     # ACSF = self.y[:, self.n * 3: self.n * 3 + 1]
    #     # Tm = self.y[:, self.n * 3 + 1: self.n * 4 + 1]
    #     # Tp = self.y[:, self.n * 4 + 1: self.n * 5 + 1]
    #     # To = self.y[:, self.n * 5 + 1: self.n * 6 + 1]
    #     Tf = self.y[:, self.n * 6 + 1: self.n * 7 + 1]
    #     # TCSF = self.y[:, self.n * 7 + 1: self.n * 7 + 2]
    #     # TpCSF = self.y[:, self.n * 7 + 2: self.n * 7 + 3]
    #     N = self.y[:, self.n * 7 + 3: self.n * 8 + 3]
    #     return Af, Tf, N

    def get_L(self, t):
        splits = [0, 3.0, 6.0, 8.0, 11.0, 12.0]  #adjust 9.0 to 8.0 now different form before
        for i in range(5):
            if splits[i] <= t <= splits[i + 1]:
                return self.Ls[i]
        if t > splits[-1]:
            return self.Ls[-1]
        assert splits[0] <= t <= splits[-1], "error in get_L: t = {} exceeded range!".format(t)

    def pend(self, y, t):
        # mt.time_start()
        self.L = self.get_L(t)

        Am = y[0: self.n]
        Ao = y[self.n: self.n * 2]
        Af = y[self.n * 2: self.n * 3]
        ACSF = y[self.n * 3: self.n * 3 + 1]
        Tm = y[self.n * 3 + 1: self.n * 4 + 1]
        Tp = y[self.n * 4 + 1: self.n * 5 + 1]
        To = y[self.n * 5 + 1: self.n * 6 + 1]
        Tf = y[self.n * 6 + 1: self.n * 7 + 1]
        TCSF = y[self.n * 7 + 1: self.n * 7 + 2]
        TpCSF = y[self.n * 7 + 2: self.n * 7 + 3]
        N = y[self.n * 7 + 3: self.n * 8 + 3]

        # k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2 \
        #     = iter(self.params)

        k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2, n_TA, n_cA, n_AT, n_cT, n_cTp, n_cTo, n_AN, n_TN, n_a1A, n_a2A, n_a1T, n_a2T, n_a1Tp, n_a2Tp, k_acsf, k_tcsf \
            = iter(self.params)
        # k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2, n_TA, n_cA, n_AT, n_cT, n_cTp, n_cTo, n_AN, n_TN, n_a1A, n_a2A, n_a1T, n_a2T, n_a1Tp, n_a2Tp, k_acsf, k_tcsf \
        #     = iter(self.params)
        #        k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2, n_TA, n_cA, n_AT, n_cT, n_cTp, n_cTo, n_AN, n_TN, n_a1A, n_a2A, n_a1T, n_a2T, n_a1Tp, K_cA \
        #            = iter(self.params)

        # n_TA = 2.0
        # n_cA = 4.0
        # n_AT = 1.0
        # n_cT = 1.0
        # n_cTp = 4.0
        # n_cTo = 1.0
        # n_AN = 2.0
        # n_TN = 2.0
        # n_a1A = 2.0
        # n_a2A = 1.0
        # n_a1T = 1.0
        # n_a2T = 2.0
        # n_a1Tp = 2.0

        # d_Am = 1.0  # 5.0
        # d_Ao = 1.0
        # d_Tm = 1.0  # 5.0
        # d_Tp = 1.0  # 5.0
        # d_To = 1.0
        d_Am, d_Ao, d_Tm, d_Tp, d_To = iter(self.diffusion_list)

        sum_func = np.sum
        matmul_func = my_matmul  # np.matmul
        offset = 1e-18

        Am_ = k_p1Am + k_p2Am * 1.0 / (
                numpy_safe_pow(K_mTA, n_TA) / numpy_safe_pow(To, n_TA) + 1.0) - k_dAm * Am - n_a1A * k_a1A * (
                  numpy_safe_pow(Am, n_a1A)) - n_a2A * k_a2A * Af * numpy_safe_pow(Am, n_a2A) + (
                      n_a1A + n_a2A) * k_diA * Ao - n_cA * k_cA * (
                  numpy_safe_pow(Am, n_cA)) * Ao - k_sA * Am + d_Am * matmul_func(self.L, Am)

        Ao_ = - k_dAo * Ao + k_a1A * numpy_safe_pow(Am, n_a1A) + k_a2A * Af * numpy_safe_pow(Am,
                                                                                             n_a2A) - k_diA * Ao - k_cA * numpy_safe_pow(
            Am, n_cA) * Ao + d_Ao * matmul_func(self.L,
                                                Ao)

        Af_ = k_cA * numpy_safe_pow(Am, n_cA) * Ao

        ACSF_ = k_sA * sum_func(Am) - k_yA * ACSF
        assert self.const_truth.params["option"] in ["option1", "option2"] or "option1" in self.const_truth.params["option"]
        if "option1" in self.const_truth.params["option"]:
            #####0222######
            Tm_ = k_pTm * Am ** 2 - k_dTm * Tm - (
                    k_ph1 + k_ph2 * Ao
            ) * Tm + k_deph * Tp - n_a1T * k_a1T * numpy_safe_pow(
                Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * 1.0 / (
                          1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp), n_a2T)) + (
                          n_a1T + n_a2T) * k_diT * To - n_cT * k_cT * numpy_safe_pow(
                Tm, n_cT) * (numpy_safe_pow(Tp, n_cTp)) * To - k_sT * Tm + d_Tm * matmul_func(self.L, Tm)

            Tp_ = -k_dTp * Tp + (
                    k_ph1 + k_ph2 * Ao) * Tm - k_deph * Tp - n_a1Tp * k_a1T * (
                      numpy_safe_pow(Tm, n_a1T)) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * 1.0 / (
                          1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp), n_a2T)) + (
                          n_a1Tp + n_a2T) * k_diT * To - n_cTp * k_cT * numpy_safe_pow(
                Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * To - k_sTp * Tp + d_Tp * matmul_func(self.L, Tp)

            # Tp_ = -k_dTp * Tp + (
            #         k_ph1 + k_ph2 * 1.0 / (numpy_safe_pow(K_mAT, n_AT) / numpy_safe_pow(Ao,
            #                                                                             n_AT) + 1.0)) * Tm - k_deph * Tp - n_a1Tp * k_a1T * (
            #           numpy_safe_pow(Tm, n_a1T)) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * 1.0 / (
            #               1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp), n_a2T)) + (
            #                   n_a1Tp + n_a2T) * k_diT * To - n_cTp * k_cT * numpy_safe_pow(
            #     Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * To - k_sTp * Tp + d_Tp * matmul_func(self.L, Tp)

            To_ = - k_dTo * To + k_a1T * numpy_safe_pow(Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) + k_a2T * Tf * 1.0 / (
                    1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp),
                                                                        n_a2T)) - k_diT * To - k_cT * numpy_safe_pow(Tm,
                                                                                                                     n_cT) * (
                      numpy_safe_pow(Tp, n_cTp)) * To + d_To * matmul_func(self.L, To)

        else:  # option2
            Tm_ = k_pTm - k_dTm * Tm - (k_ph1 + k_ph2 * 1.0 / (numpy_safe_pow(K_mAT, n_AT) / numpy_safe_pow(Ao,
                                                                                                            n_AT) + 1.0)) * Tm + k_deph * Tp - n_a1T * k_a1T * numpy_safe_pow(
                Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) - n_a2T * k_a2T * Tf * numpy_safe_pow(Tm,
                                                                                              n_a2T) * numpy_safe_pow(
                Tp, n_a2Tp) + (n_a1T + n_a2T) * k_diT * To - n_cT * k_cT * numpy_safe_pow(
                Tm, n_cT) * (numpy_safe_pow(Tp, n_cTp)) * To - k_sT * Tm + d_Tm * matmul_func(self.L, Tm)

            Tp_ = -k_dTp * Tp + (k_ph1 + k_ph2 * 1.0 / (numpy_safe_pow(K_mAT, n_AT) / numpy_safe_pow(Ao,
                                                                                                     n_AT) + 1.0)) * Tm - k_deph * Tp - n_a1Tp * k_a1T * (
                      numpy_safe_pow(Tm, n_a1T)) * numpy_safe_pow(Tp, n_a1Tp) - n_a2Tp * k_a2T * Tf * numpy_safe_pow(Tm,
                                                                                                                     n_a2T) * numpy_safe_pow(
                Tp, n_a2Tp) + (n_a1Tp + n_a2Tp) * k_diT * To - n_cTp * k_cT * numpy_safe_pow(
                Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * To - k_sTp * Tp + d_Tp * matmul_func(self.L, Tp)

            To_ = - k_dTo * To + k_a1T * numpy_safe_pow(Tm, n_a1T) * numpy_safe_pow(Tp, n_a1Tp) + k_a2T * Tf * 1.0 / (
                    1.0 + numpy_safe_pow(K_mT2, n_a2T) / numpy_safe_pow((Tm + Tp),
                                                                        n_a2T)) - k_diT * To - k_cT * numpy_safe_pow(Tm,
                                                                                                                     n_cT) * (
                      numpy_safe_pow(Tp, n_cTp)) * To + d_To * matmul_func(self.L, To)

        #         Tf_ = k_cT * numpy_safe_pow(Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * numpy_safe_pow(To, n_cTo)
        #
        #         TCSF_ = k_sT * sum_func((Tm ** 2) / (Tm ** 2 + k_tcsf ** 2)) - k_yT * TCSF  # v0118 sqrt !
        # #        TCSF_ = k_sT * sum_func(Tm**2) - k_yT * TCSF
        #         TpCSF_ = k_sTp * sum_func((Tp ** 2) / (Tp ** 2 + k_tcsf ** 2)) - k_yTp * TpCSF  # v0118 sqrt !
        #
        #         N_ = k_AN * 1.0 / (numpy_safe_pow(K_mAN, n_AN) / numpy_safe_pow((Ao + Af), n_AN) + 1.0) + k_TN * 1.0 / (
        #                     numpy_safe_pow(K_mTN, n_TN) / numpy_safe_pow((To + Tf), n_TN) + 1.0)

        Tf_ = k_cT * numpy_safe_pow(Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * numpy_safe_pow(To, n_cTo)

        TCSF_ = k_sT * sum_func(Tm) - k_yT * TCSF

        TpCSF_ = k_sTp * sum_func(Tp) - k_yTp * TpCSF

        N_ = k_AN * 1.0 / (numpy_safe_pow(K_mAN, n_AN) / numpy_safe_pow((Ao + Af), n_AN) + 1.0) + k_TN * 1.0 / (
                numpy_safe_pow(K_mTN, n_TN) / numpy_safe_pow((To + Tf), n_TN) + 1.0)
        #
        dy = np.concatenate([Am_, Ao_, Af_, ACSF_, Tm_, Tp_, To_, Tf_, TCSF_, TpCSF_, N_])
        # # print(dy.shape)
        # mt.time_end()
        return dy

    def draw(self, save_flag=True, time_string="test", given_loss=None):

        folder_path = "figure/{}/".format(time_string)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path_target = os.path.join(folder_path, "figure_target_{}.png".format(time_string))
        save_path_rest = os.path.join(folder_path, "figure_rest_{}.png".format(time_string))
        m = MultiSubplotDraw(row=3, col=3, fig_size=(24, 18), tight_layout_flag=True, show_flag=False,
                             save_flag=save_flag,
                             save_path=save_path_target, save_dpi=400)
        for i, (name, data, color, line_string) in enumerate(
                zip(self.output_names, self.output[:len(self.output_names)], self.colors[:len(self.output_names)],
                    self.lines)):
            ax = m.add_subplot(
                y_lists=data[:, :-int(3.0 / self.T_unit)] if self.const_truth.params["start"] == "fixed" else data,
                x_list=self.t,
                color_list=[color],
                line_style_list=["solid"],
                fig_title="{}{}".format(name, " (loss={})".format(
                    given_loss[i]) if given_loss is not None and i != 2 else ""),
                legend_list=[name],
                line_width=2,
            )
            # ax.set_ylim([np.min(data[0]), np.max(data[0])])

            if self.const_truth:
                x = self.const_truth.x[line_string]
                y = self.const_truth.y[line_string]
                # print(len(x), len(y))
                ax2 = ax.twinx()
                ax2.set_ylabel("truth points val", fontsize=15)
                ax2.scatter(x=x, y=y, s=100, facecolor="red", alpha=0.5, marker="o", edgecolors='black', linewidths=1,
                            zorder=10)
                # if line_string in ["NPET"]:
                #     ax2.scatter(x=x, y=y, s=100, facecolor="blue", alpha=0.5, marker="d",
                #                 edgecolors='black', linewidths=1,
                #                 zorder=10)
                # elif line_string in ["ACSF", "TpCSF", "TCSF", "TtCSF"]:
                #     ax2.scatter(x=x[[0, 2, 3, 4]], y=y[[0, 2, 3, 4]], s=100, facecolor="red", alpha=0.5, marker="o", edgecolors='black', linewidths=1,
                #                 zorder=10)
                #     ax2.scatter(x=x[1], y=y[1], s=100, facecolor="blue", alpha=0.5, marker="d", edgecolors='black', linewidths=1,
                #                 zorder=10)
                # else:
                #     ax2.scatter(x=x, y=y, s=100, facecolor="red", alpha=0.5, marker="o",
                #                 edgecolors='black', linewidths=1,
                #                 zorder=10)
                ax2.tick_params(axis='y', labelcolor="red", labelsize=15)
                ylim_bottom, ylim_top = ax2.get_ylim()
                if line_string in ["ACSF"]:
                    ax2.set_ylim([ylim_bottom, ylim_bottom + (ylim_top - ylim_bottom) / (
                                data[0][int(x[0] / self.T_unit)] - data[0][int(x[-1] / self.T_unit)]) * (
                                              data[0][0] - data[0][int(x[-1] / self.T_unit)])])
                elif line_string in ["APET", "TPET", "TpCSF", "TCSF", "TtCSF"]:
                    ax2.set_ylim([ylim_top - (ylim_top - ylim_bottom) / (
                                data[0][int(x[0] / self.T_unit)] - data[0][int(x[-1] / self.T_unit)]) * (
                                              data[0][0] - data[0][int(x[-1] / self.T_unit)]), ylim_top])

        m.add_subplot(
            y_lists=np.concatenate(self.output[:len(self.output_names)], axis=0),
            x_list=self.t,
            color_list=self.colors[:len(self.output_names)],
            line_style_list=["solid"] * len(self.output_names),
            fig_title="Seven Target Curves",
            legend_list=self.output_names,
            line_width=2,
        )
        m.draw()
        print("Save flag: {}. Target figure is saved to {}".format(save_flag, save_path_target))

        m = MultiSubplotDraw(row=2, col=3, fig_size=(24, 12), tight_layout_flag=True, show_flag=False,
                             save_flag=save_flag,
                             save_path=save_path_rest, save_dpi=400)
        for name, data, color in zip(self.output_names_rest, self.output[-len(self.output_names_rest):],
                                     self.colors[:len(self.output_names_rest)]):
            m.add_subplot(
                y_lists=data,
                x_list=self.t,
                color_list=[color],
                line_style_list=["solid"],
                fig_title=name,
                legend_list=[name],
                line_width=2,
            )
        m.add_subplot(
            y_lists=np.concatenate(self.output[-len(self.output_names_rest):], axis=0),
            x_list=self.t,
            color_list=self.colors[:len(self.output_names_rest)],
            line_style_list=["solid"] * len(self.output_names_rest),
            fig_title="Rest Curves",
            legend_list=self.output_names_rest,
            line_width=2,
        )
        # plt.suptitle("{} Class ODE Solution".format(self.class_name), fontsize=40)
        m.draw()
        print("Save flag: {}. Rest figure is saved to {}".format(save_flag, save_path_rest))


def f_csf_rate(x, thr=1.7052845384621318, tol=0.2, p=1.0):
    return max((x - thr * (1 + tol)) * p, (thr * (1 - tol) - x) * p, 0)


def rate_penalty(x, lb, ub, p=1.0):
    return max((x - ub) * p, (lb - x) * p, 0)


def loss_func(params, starts_weight, diffusion_list, ct):
    # print("calling loss_func..")
    truth = ADSolver("CN")
    truth.step(params, starts_weight, diffusion_list)
    targets = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]
    record = np.zeros(len(targets))
    for i, one_target in enumerate(targets):
        target_points = np.asarray(ct.y[one_target])
        if len(target_points) == 0:
            record[i] = 0.0
            continue
        t_fixed = np.asarray(ct.x[one_target])
        # if one_target in ["ACSF", "TpCSF", "TCSF", "TtCSF"]:  # skip the second points for all CSFs
        #     target_points = target_points[[0, 2, 3, 4]]
        #     t_fixed = t_fixed[[0, 2, 3, 4]]
        index_fixed = (t_fixed / truth.T_unit).astype(int)
        predict_points = np.asarray(truth.output[i][0][index_fixed])
        # print("target_points:", target_points.shape)
        # print("predict_points:", predict_points.shape)

        target_points_scaled = (target_points - np.min(target_points)) / (np.max(target_points) - np.min(target_points))
        # print("[loss_func] ({}) predict_points: {}".format(one_target, predict_points))

        if np.max(predict_points) - np.min(predict_points) <= 1e-15:
            predict_points_scaled = np.zeros(len(target_points_scaled))
        else:
            predict_points_scaled = (predict_points - np.min(predict_points)) / (
                        np.max(predict_points) - np.min(predict_points))
        # try:
        #     # assert 0.0 not in list(np.max(predict_points) - np.min(predict_points))
        #     print(predict_points - np.min(predict_points), np.max(predict_points) - np.min(predict_points))
        #     predict_points_scaled = (predict_points - np.min(predict_points)) / (np.max(predict_points) - np.min(predict_points))
        # except Exception as e:
        #     print(e)
        #     predict_points_scaled = np.zeros(len(target_points_scaled))

        # record[i] = np.mean(((predict_points - target_points) / target_points) ** 2)
        record[i] = np.mean((predict_points_scaled - target_points_scaled) ** 2)
    # record = record[[0, 1, 3, 4, 5, 6]]
    csf_rate = f_csf_rate(np.max(truth.output[3][0]) / np.max(truth.output[6][0]))
    return record, csf_rate  # remove NPET here


# def loss_func_spatial(params, starts_weight, diffusion_list, ct_spatial: ConstTruthSpatial, save_file_path, silent=True,
#                       save_flag=False, element_id=None):
#     ad = ADSolver("CN", ct_spatial)
#     ad.step_spatial(params, starts_weight, diffusion_list)
#     # output = ad.get_output_spatial()
#     # print([item.shape for item in output])
#     targets = ["APET", "TPET", "NPET"]
#     record_pattern_penalty = np.zeros(len(targets))
#     record_rate_penalty = np.zeros(len(targets))
#     record_rate = np.zeros(len(targets))
#     for i, one_target in enumerate(targets):
#         target_points = np.asarray(ct_spatial.y[one_target])
#         target_points_scaled = (target_points - np.min(target_points)) / (np.max(target_points) - np.min(target_points))
#         # print(target_points.shape)
#
#         t_fixed = np.asarray(ct_spatial.x[one_target])
#         #     # if one_target in ["ACSF", "TpCSF", "TCSF", "TtCSF"]:  # skip the second points for all CSFs
#         #     #     target_points = target_points[[0, 2, 3, 4]]
#         #     #     t_fixed = t_fixed[[0, 2, 3, 4]]
#         index_fixed = (t_fixed / ad.T_unit).astype(int)
#         predict_points = np.asarray(ad.output_spatial[i][index_fixed])
#         # if np.max(predict_points) - np.min(predict_points) <= 1e-15:
#         #     predict_points_scaled = np.zeros(len(target_points_scaled))
#         # else:
#         #     predict_points_scaled = (predict_points - np.min(predict_points)) / (
#         #                 np.max(predict_points) - np.min(predict_points))
#
#         if np.max(predict_points) - np.min(predict_points) <= 1e-15:
#             predict_points_scaled = np.zeros(predict_points.shape)
#         else:
#             predict_points_scaled = (predict_points - np.min(predict_points)) / (
#                         np.max(predict_points) - np.min(predict_points))
#
#         record_rate[i] = (np.max(predict_points) - np.min(predict_points)) / np.max(predict_points)
#         # print(record_rate)
#         record_rate_penalty[i] = rate_penalty(
#             record_rate[i],
#             ct_spatial.const_min_max_scale[one_target][0],
#             ct_spatial.const_min_max_scale[one_target][1],
#         )
#         record_pattern_penalty[i] = np.mean((predict_points_scaled - target_points_scaled) ** 2)
#     time_string = get_now_string()
#     if not silent:
#         print("time_string:", time_string)
#         print("record_rate:", record_rate)
#         print("record_rate_penalty:", record_rate_penalty)
#         print("record_pattern_penalty:", record_pattern_penalty)
#
#     loss = np.sum(record_pattern_penalty) + np.sum(record_rate_penalty)
#     with open(save_file_path, "a") as f:
#         f.write("{0},{1},{2},{3},{4:.9f},{5:.9f},{6:.9f},{7:.6e},{8:.6e},{9:.6e},{10:.6e},{11:.6e}\n".format(
#             element_id,
#             time_string,
#             ct_spatial.args.threshold,
#             ct_spatial.args.diffusion_unit_rate,
#             loss,
#             np.sum(record_pattern_penalty),
#             np.sum(record_rate_penalty),
#             diffusion_list[0],
#             diffusion_list[1],
#             diffusion_list[2],
#             diffusion_list[3],
#             diffusion_list[4],
#         ))
#     if save_flag:
#         folder_path = "./figure_spatial/{}/".format(time_string)
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#         np.save("{}/diffusion_list.npy".format(folder_path), np.asarray(diffusion_list))
#         np.save("{}/params.npy".format(folder_path), np.asarray(params))
#         np.save("{}/starts_weight.npy".format(folder_path), np.asarray(starts_weight))
#
#         default_colors = ["r", "g", "b"]
#         fig = plt.figure(figsize=(18, 18))
#         for i, one_target in enumerate(targets):
#             draw_data = ad.output_spatial[i]
#             draw_data = np.swapaxes(draw_data, 0, 1)
#             # print("{} shape: {}".format(one_target, draw_data.shape))
#             ax = fig.add_subplot(3, 1, i + 1)
#             for j in range(160):
#                 ax.plot(ad.t, draw_data[j].flatten(), c=default_colors[i], linewidth=1, alpha=0.5)
#             ax.set_title("{} ({})".format(one_target, time_string))
#         plt.savefig("{}/diffusion_ATN.png".format(folder_path), dpi=300)
#         plt.close()
#
#         fig = plt.figure(figsize=(8 * len(LABEL_LIST), 3 * len(targets)))
#
#         for i, one_target in enumerate(targets):
#             target_points = np.asarray(ct_spatial.y[one_target])
#             t_fixed = np.asarray(ct_spatial.x[one_target])
#             index_fixed = (t_fixed / ad.T_unit).astype(int)
#             predict_points = np.asarray(ad.output_spatial[i][index_fixed])
#             for j in range(len(LABEL_LIST)):
#                 ax = fig.add_subplot(len(targets), len(LABEL_LIST), i * len(LABEL_LIST) + j + 1)
#                 ax.plot(range(1, 161), predict_points[j, :], c="black", marker="o", markersize=1, linewidth=1)
#                 ax.tick_params(axis='y', labelcolor="black")
#                 ax.set_title("{}-{}".format(one_target, LABEL_LIST[j]))
#
#                 ax2 = ax.twinx()
#                 ax2.plot(range(1, 161), target_points[j, :], c="red", marker="o", markersize=1, linewidth=1)
#                 ax2.tick_params(axis='y', labelcolor="red")
#
#         plt.tight_layout()
#         plt.savefig("{}/diffusion.png".format(folder_path), dpi=300)
#         # plt.show()
#         plt.close()
#     return time_string, loss


class MiniTruth:
    def __init__(self, x_dict, y_dict):
        self.x = x_dict
        self.y = y_dict
        # print(f"MiniTruth: x shape = {self.x.shape}, y shape = {self.y.shape}")
        print("Mini Truth:")
        print({key: value.shape for key, value in self.x.items()})
        print({key: value.shape for key, value in self.y.items()})


def loss_func_spatial(params, starts_weight, diffusion_list, ct_spatial: ConstTruthSpatial, save_file_path, silent=True,
                      save_flag=False, element_id=None, output_flag=False):
    ad = ADSolver("CN", ct_spatial)

    # print(f"\n0619 debug: truth y0: {ad.y0}\n")

    ad.step_spatial(params, starts_weight, diffusion_list)
    # y0 = np.load("saves/y0.npy")[:480].reshape(3,160)

    # output = ad.get_output_spatial()
    # print([item.shape for item in output])
    targets = ["APET", "TPET", "NPET"]
    record_pattern_penalty = np.zeros(len(targets))
    record_rate_penalty = np.zeros(len(targets))
    record_rate = np.zeros(len(targets))
    # output_truth_dict = dict()
    # output_predict_dict = dict()
    output_predict_dict_origin_all = dict()
    output_truth_dict_origin = dict()
    output_predict_dict_origin = dict()
    for i, one_target in enumerate(targets):
        target_points = np.asarray(ct_spatial.y[one_target])
        # print("\ntarget point shape: %s", target_points.shape)
        target_points_scaled = (target_points - np.min(target_points)) / (np.max(target_points) - np.min(target_points))
        # print(target_points.shape)

        t_fixed = np.asarray(ct_spatial.x[one_target])
        # print("\nt_fixed shape: %s",t_fixed.shape)

        #     # if one_target in ["ACSF", "TpCSF", "TCSF", "TtCSF"]:  # skip the second points for all CSFs
        #     #     target_points = target_points[[0, 2, 3, 4]]
        #     #     t_fixed = t_fixed[[0, 2, 3, 4]]
        index_fixed = (t_fixed / ad.T_unit).astype(int)
        # print("\nindex_fixed shape: %s", index_fixed.shape)
        predict_points = np.asarray(ad.output_spatial[i][index_fixed])
        # print("\npredict points shape: %s", predict_points.shape)
        # if np.max(predict_points) - np.min(predict_points) <= 1e-15:
        #     predict_points_scaled = np.zeros(len(target_points_scaled))
        # else:
        #     predict_points_scaled = (predict_points - np.min(predict_points)) / (
        #                 np.max(predict_points) - np.min(predict_points))

        if np.max(predict_points) - np.min(predict_points) <= 1e-15:
            predict_points_scaled = np.zeros(predict_points.shape)
        else:
            predict_points_scaled = (predict_points - np.min(predict_points)) / (
                        np.max(predict_points) - np.min(predict_points))

        if output_flag:
            # output_truth_dict[one_target] = target_points_scaled
            # output_predict_dict[one_target] = predict_points_scaled
            output_predict_dict_origin_all[one_target] = np.asarray(ad.output_spatial[i])
            output_truth_dict_origin[one_target] = target_points
            output_predict_dict_origin[one_target] = predict_points

        record_rate[i] = (np.max(predict_points) - np.min(predict_points)) / np.max(predict_points)
        # print(record_rate)
        record_rate_penalty[i] = rate_penalty(
            record_rate[i],
            ct_spatial.const_min_max_scale[one_target][0],
            ct_spatial.const_min_max_scale[one_target][1],
        )
        record_pattern_penalty[i] = np.mean((predict_points_scaled - target_points_scaled) ** 2)
    time_string = get_now_string()
    if not silent:
        print("time_string:", time_string)
        print("record_rate:", record_rate)
        print("record_rate_penalty:", record_rate_penalty)
        print("record_pattern_penalty:", record_pattern_penalty)

    loss = np.sum(record_pattern_penalty) + np.sum(record_rate_penalty)
    with open(save_file_path, "a") as f:
        f.write("{0},{1},{2},{3},{4:.9f},{5:.9f},{6:.9f},{7:.6e},{8:.6e},{9:.6e},{10:.6e},{11:.6e}\n".format(
            element_id,
            time_string,
            ct_spatial.args.threshold,
            ct_spatial.args.diffusion_unit_rate,
            loss,
            np.sum(record_pattern_penalty),
            np.sum(record_rate_penalty),
            diffusion_list[0],
            diffusion_list[1],
            diffusion_list[2],
            diffusion_list[3],
            diffusion_list[4],
        ))
    if save_flag:
        folder_path = "./figure_spatial/{}/".format(time_string)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save("{}/diffusion_list.npy".format(folder_path), np.asarray(diffusion_list))
        np.save("{}/params.npy".format(folder_path), np.asarray(params))
        np.save("{}/starts_weight.npy".format(folder_path), np.asarray(starts_weight))

        default_colors = ["r", "g", "b"]
        fig = plt.figure(figsize=(18, 18))
        for i, one_target in enumerate(targets):
            draw_data = ad.output_spatial[i]
            draw_data = np.swapaxes(draw_data, 0, 1)
            # print("{} shape: {}".format(one_target, draw_data.shape))
            ax = fig.add_subplot(3, 1, i + 1)
            for j in range(160):
                ax.plot(ad.t, draw_data[j].flatten(), c=default_colors[i], linewidth=1, alpha=0.5)
            ax.set_title("{} ({})".format(one_target, time_string))
        plt.savefig("{}/diffusion_ATN.png".format(folder_path), dpi=300)
        plt.close()

        fig = plt.figure(figsize=(8 * len(LABEL_LIST), 3 * len(targets)))

        for i, one_target in enumerate(targets):
            target_points = np.asarray(ct_spatial.y[one_target])
            t_fixed = np.asarray(ct_spatial.x[one_target])
            index_fixed = (t_fixed / ad.T_unit).astype(int)
            predict_points = np.asarray(ad.output_spatial[i][index_fixed])
            for j in range(len(LABEL_LIST)):
                ax = fig.add_subplot(len(targets), len(LABEL_LIST), i * len(LABEL_LIST) + j + 1)
                ax.plot(range(1, 161), predict_points[j, :], c="black", marker="o", markersize=1, linewidth=1)
                ax.tick_params(axis='y', labelcolor="black")
                ax.set_title("{}-{}".format(one_target, LABEL_LIST[j]))

                ax2 = ax.twinx()
                ax2.plot(range(1, 161), target_points[j, :], c="red", marker="o", markersize=1, linewidth=1)
                ax2.tick_params(axis='y', labelcolor="red")

        plt.tight_layout()
        plt.savefig("{}/diffusion.png".format(folder_path), dpi=300)
        # plt.show()
        plt.close()

    if output_flag:
        # ct_truth = MiniTruth(ct_spatial.x, output_truth_dict)
        # ct_pred = MiniTruth(ct_spatial.x, output_predict_dict)
        ct_truth_origin_all = MiniTruth({one_key: np.linspace(0.0, 12.00, 1201) for one_key in ct_spatial.x}, output_predict_dict_origin_all)
        ct_truth_origin = MiniTruth(ct_spatial.x, output_truth_dict_origin)
        ct_pred_origin = MiniTruth(ct_spatial.x, output_predict_dict_origin)
        return ct_truth_origin_all, ct_truth_origin, ct_pred_origin

    return time_string, loss


def loss_func_spatial_match(params, starts_weight, diffusion_list, ct_spatial: ConstTruthSpatial, save_file_path, silent=True,
                      save_flag=False, element_id=None, output_flag=False, n_class=2):
    ad = ADSolver("CN", ct_spatial)

    # print(f"\n0619 debug: truth y0: {ad.y0}\n")

    ad.step_spatial(params, starts_weight, diffusion_list)
    # y0 = np.load("saves/y0.npy")[:480].reshape(3,160)

    # output = ad.get_output_spatial()
    # print([item.shape for item in output])
    targets = ["APET", "TPET", "NPET"] #I want to add the tau-csf in the target (didn't)

    record_mat_all = np.zeros([len(targets), n_class, n_class])
    record_mat_sum = np.zeros([n_class, n_class])
    record_rate_list = np.zeros(len(targets))

    # output_truth_dict = dict()
    # output_predict_dict = dict()
    output_predict_dict_origin_all = dict()
    output_truth_dict_origin = dict()
    output_predict_dict_origin = dict()
    for i, one_target in enumerate(targets):
        target_points = np.asarray(ct_spatial.y[one_target])

        t_fixed = np.asarray(ct_spatial.x[one_target])
        index_fixed = (t_fixed / ad.T_unit).astype(int)

        predict_points = np.asarray(ad.output_spatial[i][index_fixed])

        if np.max(predict_points) - np.min(predict_points) <= 1e-15:
            predict_points = np.zeros(predict_points.shape)

        target_points_flatten = target_points.flatten()
        target_points_flatten_class = convert_val_to_class(get_classification_threshold_list(n_class, target_points_flatten), target_points_flatten)
        predict_points_flatten = predict_points.flatten()
        predict_points_flatten_class = convert_val_to_class(get_classification_threshold_list(n_class, predict_points_flatten), predict_points_flatten)

        match_matrix = count_matrix(target_points_flatten_class, predict_points_flatten_class, n_class)
        record_mat_all[i] = match_matrix
        record_mat_sum += match_matrix
        record_rate_list[i] = np.trace(match_matrix) / np.sum(match_matrix)



        if output_flag:
            # output_truth_dict[one_target] = target_points_scaled
            # output_predict_dict[one_target] = predict_points_scaled
            output_predict_dict_origin_all[one_target] = np.asarray(ad.output_spatial[i])
            output_truth_dict_origin[one_target] = target_points
            output_predict_dict_origin[one_target] = predict_points


    time_string = get_now_string()


    loss = len(targets) - np.sum(record_rate_list)
    if save_flag:
        with open(save_file_path, "a") as f:
            f.write("{0},{1},{2},{3},{4:.9f},{5},{6:.6e},{7:.6e},{8:.6e},{9:.6e},{10:.6e}\n".format(
                element_id,
                time_string,
                ct_spatial.args.threshold,
                ct_spatial.args.diffusion_unit_rate,
                loss,
                str(record_rate_list).replace(",", ";"),
                diffusion_list[0],
                diffusion_list[1],
                diffusion_list[2],
                diffusion_list[3],
                diffusion_list[4],
            ))
    if save_flag:
        folder_path = "./figure_spatial/{}/".format(time_string)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save("{}/diffusion_list.npy".format(folder_path), np.asarray(diffusion_list))
        np.save("{}/params.npy".format(folder_path), np.asarray(params))
        np.save("{}/starts_weight.npy".format(folder_path), np.asarray(starts_weight))

        default_colors = ["r", "g", "b"]
        fig = plt.figure(figsize=(18, 18))
        for i, one_target in enumerate(targets):
            draw_data = ad.output_spatial[i]
            draw_data = np.swapaxes(draw_data, 0, 1)
            # print("{} shape: {}".format(one_target, draw_data.shape))
            ax = fig.add_subplot(3, 1, i + 1)
            for j in range(160):
                ax.plot(ad.t, draw_data[j].flatten(), c=default_colors[i], linewidth=1, alpha=0.5)
            ax.set_title("{} ({})".format(one_target, time_string))
        plt.savefig("{}/diffusion_ATN.png".format(folder_path), dpi=300)
        plt.close()

        fig = plt.figure(figsize=(8 * len(LABEL_LIST), 3 * len(targets)))

        for i, one_target in enumerate(targets):
            target_points = np.asarray(ct_spatial.y[one_target])
            t_fixed = np.asarray(ct_spatial.x[one_target])
            index_fixed = (t_fixed / ad.T_unit).astype(int)
            predict_points = np.asarray(ad.output_spatial[i][index_fixed])
            for j in range(len(LABEL_LIST)):
                ax = fig.add_subplot(len(targets), len(LABEL_LIST), i * len(LABEL_LIST) + j + 1)
                ax.plot(range(1, 161), predict_points[j, :], c="black", marker="o", markersize=1, linewidth=1)
                ax.tick_params(axis='y', labelcolor="black")
                ax.set_title("{}-{}".format(one_target, LABEL_LIST[j]))

                ax2 = ax.twinx()
                ax2.plot(range(1, 161), target_points[j, :], c="red", marker="o", markersize=1, linewidth=1)
                ax2.tick_params(axis='y', labelcolor="red")

        plt.tight_layout()
        plt.savefig("{}/diffusion.png".format(folder_path), dpi=300)
        # plt.show()
        plt.close()

    if output_flag:
        # ct_truth = MiniTruth(ct_spatial.x, output_truth_dict)
        # ct_pred = MiniTruth(ct_spatial.x, output_predict_dict)
        ct_truth_origin_all = MiniTruth({one_key: np.linspace(0.0, 12.00, 1201) for one_key in ct_spatial.x}, output_predict_dict_origin_all)
        ct_truth_origin = MiniTruth(ct_spatial.x, output_truth_dict_origin)
        ct_pred_origin = MiniTruth(ct_spatial.x, output_predict_dict_origin)
        return ct_truth_origin_all, ct_truth_origin, ct_pred_origin, record_mat_all, record_mat_sum, record_rate_list

    return time_string, loss


# MyTime is only for debugging
class MyTime:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.tmp = None

    def time_start(self):
        ts = time.time()
        # if self.count > 0:
        #     self.sum += (ts - self.tmp)
        # self.count += 1
        self.tmp = ts

    def time_end(self):
        ts = time.time()
        self.sum += (ts - self.tmp)
        self.count += 1
        self.tmp = None

    def print(self):
        print("count = {}; total time = {} s; avg time = {} s".format(self.count, self.sum, self.sum / self.count))


def run(params=None, starts=None, diffusion_list=None, time_string=None, silent_flag=False):
    if not silent_flag:
        if not time_string:
            time_string = get_now_string()
        print("Time String (as folder name): {}".format(time_string))

    class_name = "CN"
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, help="dataset strategy")
    # parser.add_argument("--start", type=str, help="start strategy")
    # parser.add_argument("--generation", type=int, help="generation")
    # parser.add_argument("--pop_size", type=int, help="pop_size")
    # parser.add_argument("--params", type=str, help="params file (in 'saves/')")
    # parser.add_argument("--diff_strategy", type=str, help="C / D")
    # opt = parser.parse_args()
    ct = ConstTruth(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/",
        dataset=opt.dataset,
        start=opt.start,
    )
    # given_params = np.load("saves/{}".format(opt.params))
    truth = ADSolver(class_name, ct)
    truth.step(params, starts, diffusion_list)
    loss, csf_rate_loss = loss_func(params, starts, diffusion_list, ct)
    if not silent_flag:
        print("loss: {}".format(sum(loss) + csf_rate_loss))
        print("loss parts: {} csf match loss: {}".format(list(loss), csf_rate_loss))
        truth.draw(time_string=time_string, given_loss=loss)
    return np.sum(loss) + csf_rate_loss


# def run_spatial(params=None, starts=None, diffusion_list=None, time_string=None, silent_flag=False):
#     if not silent_flag:
#         if not time_string:
#             time_string = get_now_string()
#         print("Time String (as folder name): {}".format(time_string))
#
#     class_name = "CN"
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--dataset", type=str, help="dataset strategy")
#     # parser.add_argument("--start", type=str, help="start strategy")
#     # parser.add_argument("--generation", type=int, help="generation")
#     # parser.add_argument("--pop_size", type=int, help="pop_size")
#     # parser.add_argument("--params", type=str, help="params file (in 'saves/')")
#     # parser.add_argument("--diff_strategy", type=str, help="C / D")
#     # opt = parser.parse_args()
#     ct = ConstTruth(
#         csf_folder_path="data/CSF/",
#         pet_folder_path="data/PET/",
#         dataset=opt.dataset,
#         start=opt.start,
#     )
#     # given_params = np.load("saves/{}".format(opt.params))
#     truth = ADSolver(class_name, ct)
#     truth.step(params, starts, diffusion_list)
#     loss, csf_rate_loss = loss_func(params, starts, diffusion_list, ct)
#     if not silent_flag:
#         print("loss: {}".format(sum(loss) + csf_rate_loss))
#         print("loss parts: {} csf match loss: {}".format(list(loss), csf_rate_loss))
#         truth.draw(time_string=time_string, given_loss=loss)
#     return np.sum(loss) + csf_rate_loss



def draw_spatial_truth(ct, save_path="test/test.png", n=3, draw_type="pred", ct_pred_all=None):
    line_labels = ["APET", "TPET", "NPET"]
    colors = ["pink", "lime", "cyan"]
    colors_average = ["red", "green", "blue"]
    titles = ["APET Truth", "TPET Truth", "NPET Truth"] if "truth" in save_path else ["APET Pred", "TPET Pred", "NPET Pred"]
    color_class = {
        2: ["orange", "red"],
        3: ["blue", "orange", "red"],
        4: ["blue", "green", "orange", "red"],
    }

    label_class = {
        2: ["Low", "High"],
        3: ["Low", "Medium", "High"],
        4: ["Low", "Medium Low", "Medium High", "High"],
    }

    color_list = color_class[n]
    label_list = label_class[n]

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    for i, one_label in enumerate(line_labels):
        ax = axes[i]
        color = colors[i]
        title = titles[i]
        xs = ct.x[one_label]

        data_list = ct.y[one_label].flatten()
        threshold_list = get_classification_threshold_list(n=n, data_list=data_list)
        data_label_list = convert_val_to_class(threshold_list, data_list)
        # data_labels = data_label_list.reshape(ct.y[one_label].shape)


        for j in range(160):
            if draw_type == "pred":
                ys_line = ct_pred_all.y[one_label][:, j].flatten()
                xs_line = ct_pred_all.x[one_label]
            else:
                ys_line = ct.y[one_label][:, j].flatten()
                xs_line = xs
            # color_plot = [color_list[item] for item in data_labels[:, j].flatten()]
            # label_plot = [label_list[item] for item in data_labels[:, j].flatten()]
            # ax.plot(xs, ys, c=color, markeredgecolor=color, markerfacecolor='white', linewidth=1.0, marker='o', markersize=6.0)
            ax.plot(xs_line, ys_line, c=color, zorder=1, linewidth=1)
            # ax.scatter(xs, ys, c=color_plot, label=label_plot, s=10)

        if draw_type == "pred":
            ax.plot(ct_pred_all.x[one_label], np.mean(ct_pred_all.y[one_label], axis=1), c=colors_average[i], zorder=2, linewidth=5, alpha=0.5,
                    linestyle="dashed")
        else:
            ax.plot(xs, np.mean(ct.y[one_label], axis=1), c=colors_average[i], zorder=2, linewidth=5, alpha=0.5, linestyle="dashed")

        x_flatten = np.repeat([ct.x[one_label]], ct.y[one_label].shape[1], axis=1).flatten()
        y_flatten = ct.y[one_label].flatten()
        label_flatten = data_label_list

        for one_key in range(n):
            x_list = x_flatten[label_flatten == one_key]
            y_list = y_flatten[label_flatten == one_key]
            ax.scatter(x_list, y_list, edgecolor=color_list[one_key], facecolor='white', label=f"{label_list[one_key]}: {len(x_list)} ({len(x_list) / len(x_flatten) * 100:.1f}%)", s=20, zorder=10, alpha=0.5)

        ax.legend()
        ax.set_title(title, fontsize=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"saved to {save_path}")


def get_classification_threshold_list(n, data_list):
    assert n in [2, 3, 4]
    data_list = np.asarray(data_list)
    sigma = np.std(data_list)
    mu = np.mean(data_list)
    if n == 2:
        return [mu, float('inf')]
    elif n == 3:
        return [mu - 0.5 * sigma, mu + 0.5 * sigma, float('inf')]
    else:
        return [mu - sigma, mu, mu + sigma, float('inf')]


def get_classification_threshold_list_match_split_list(n, data_list):
    data_list = np.asarray(data_list)
    thresholds = np.linspace(np.min(data_list) - 1e-10, np.max(data_list) + 1e-10, n)
    return [[item, float('inf')] for item in thresholds]


def convert_val_to_class(threshold_list, data_list):
    # n = len(threshold_list)
    output_list = []
    for item in data_list:
        for i, one_threshold in enumerate(threshold_list):
            if item < one_threshold:
                output_list.append(i)
                break
    return np.asarray(output_list)


def output_ct_to_txt_for_brain_surface(ct_truth, ct_pred, timestring):
    record_folder = f"brain_surface/{timestring}/"
    if not os.path.exists(record_folder):
        os.makedirs(record_folder)
    f_record = open(f"{record_folder}/record.csv", "w")
    f_record.write("n,type,line,min,max,mean,std,threshold_list,count_list,ratio_list\n")

    for n in [2, 3, 4]:
        save_type = "truth"
        ct = ct_truth
        save_folder = f"brain_surface/{timestring}/n={n}/{save_type}/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for one_key in ["APET", "TPET", "NPET"]:
            data_list = ct.y[one_key].flatten()
            threshold_list = get_classification_threshold_list(n=n, data_list=data_list)
            data_label_list = convert_val_to_class(threshold_list, data_list)
            data_labels = data_label_list.reshape(ct.y[one_key].shape)  # 5 * 160
            f_record.write("{0:d},{1},{2},{3:.4f},{4:.4f},{5:.4f},{6:.4f},{7},{8},{9}\n".format(
                n,
                save_type,
                one_key,
                np.min(data_list),
                np.max(data_list),
                np.mean(data_list),
                np.std(data_list),
                ";".join(["{0:.4e}".format(item) for item in threshold_list[:-1]]),
                ";".join(["{0:d}".format(np.count_nonzero(data_label_list == item)) for item in range(n)]),
                ";".join(["{0:.1f}%".format(np.count_nonzero(data_label_list == item) / len(data_label_list) * 100.0) for item in range(n)]),
            ))
            for i in range(5):
                txt_path = f"{save_folder}/{one_key}_{i}.txt"
                with open(txt_path, "w") as f_txt:
                    for item in data_labels[i]:
                        f_txt.write(f"{item:.2f}\n")

        save_type = "pred"
        ct = ct_pred
        save_folder = f"brain_surface/{timestring}/n={n}/{save_type}/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for one_key in ["APET", "TPET", "NPET"]:
            data_list = ct.y[one_key].flatten()
            threshold_list = get_classification_threshold_list(n=n, data_list=data_list)
            data_label_list = convert_val_to_class(threshold_list, data_list)
            data_labels = data_label_list.reshape(ct.y[one_key].shape)  # 5 * 160
            f_record.write("{0:d},{1},{2},{3:.4f},{4:.4f},{5:.4f},{6:.4f},{7},{8},{9}\n".format(
                n,
                save_type,
                one_key,
                np.min(data_list),
                np.max(data_list),
                np.mean(data_list),
                np.std(data_list),
                ";".join(["{0:.4e}".format(item) for item in threshold_list[:-1]]),
                ";".join(["{0:d}".format(np.count_nonzero(data_label_list == item)) for item in range(n)]),
                ";".join(
                    ["{0:.1f}%".format(np.count_nonzero(data_label_list == item) / len(data_label_list) * 100.0) for
                     item in range(n)]),
            ))
            for i in range(5):
                txt_path = f"{save_folder}/{one_key}_{i}.txt"
                with open(txt_path, "w") as f_txt:
                    for item in data_labels[i]:
                        f_txt.write(f"{item:.2f}\n")


def count_matrix(list1, list2, n):
    m = np.zeros((n, n), dtype=int)

    assert len(list1) == len(list2)
    for i in range(len(list1)):
        val1 = list1[i]
        val2 = list2[i]
        m[val1][val2] += 1
    return m

def print_params(arr):
    n_param = PARAM_NUM
    n_start = STARTS_NUM
    assert len(arr) == n_param + n_start
    print("Params:")
    for i in range(n_param):
        print(f"{PARAMS[i]['id']}\t{PARAMS[i]['name']}\t{arr[i]:.6e}")
    print("Starts:")
    for i in range(n_start):
        print(f"{STARTS_WEIGHTS[i]['id']}\t{STARTS_WEIGHTS[i]['name']}\t{arr[i + n_param]:.6e}")


def one_time_reduce_kan_ktn(src_path, dst_path):
    data = np.load(src_path)

    data[21] /= 0.01
    data[22] /= 0.01
    print_params(data)
    np.save(dst_path, data)


def get_min_max_rate(x, v_min, v_max):
    assert v_min <= x <= v_max and v_min < v_max
    return (x - v_min) / (v_max - v_min)


def one_time_draw_3D_match_result(ct_truth, ct_pred, timestring, n_split=100):
    record_folder = f"brain_surface_match/{timestring}/"
    if not os.path.exists(record_folder):
        os.makedirs(record_folder)


    for n in [2]:
        # record_shape = ct_truth.y["APET"].shape
        match_record = np.zeros([4, n_split, n_split])
        match_record_normal_position = np.zeros([3, 3])
        # average_match_record = np.zeros([n_split, n_split])

        save_folder = f"brain_surface_match/{timestring}/n={n}/"
        save_path_plot = os.path.join(save_folder, "match.png")
        save_path_npy = os.path.join(save_folder, "save.npy")
        save_path_normal_position_npy = os.path.join(save_folder, "normal_position_save.npy")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if os.path.exists(save_path_npy) and os.path.exists(save_path_normal_position_npy):
            print(f"{save_path_npy} and {save_path_normal_position_npy} found. Skip generating.")
            match_record = np.load(save_path_npy)
            match_record_normal_position = np.load(save_path_normal_position_npy)
        else:
            for j, one_key in enumerate(["APET", "TPET", "NPET"]):
                data_list_truth = ct_truth.y[one_key].flatten()
                data_list_pred = ct_pred.y[one_key].flatten()
                threshold_list_truth = get_classification_threshold_list_match_split_list(n_split, data_list_truth)
                threshold_list_pred = get_classification_threshold_list_match_split_list(n_split, data_list_pred)

                threshold_list_truth_normal = get_classification_threshold_list(2, data_list_truth)
                threshold_list_pred_normal = get_classification_threshold_list(2, data_list_pred)
                match_matrix_normal = count_matrix(convert_val_to_class(threshold_list_truth_normal, data_list_truth).flatten(), convert_val_to_class(threshold_list_pred_normal, data_list_pred).flatten(), 2)
                record_normal = np.trace(match_matrix_normal) / np.sum(match_matrix_normal)
                truth_mu_rate = get_min_max_rate(threshold_list_truth_normal[0], np.min(data_list_truth), np.max(data_list_truth))
                pred_mu_rate = get_min_max_rate(threshold_list_pred_normal[0], np.min(data_list_pred), np.max(data_list_pred))
                print(f"{one_key}: truth: {threshold_list_truth_normal[0]}, min = {np.min(data_list_truth)}, max = {np.max(data_list_truth)}, so rate = {truth_mu_rate}")
                print(f"{one_key}: pred: {threshold_list_pred_normal[0]}, min = {np.min(data_list_pred)}, max = {np.max(data_list_pred)}, so rate = {pred_mu_rate}")
                match_record_normal_position[j] = np.asarray([truth_mu_rate, pred_mu_rate, record_normal])
                print(np.asarray([truth_mu_rate, pred_mu_rate, record_normal]))
                for k, threshold_truth in tqdm(enumerate(threshold_list_truth), total=len(threshold_list_truth)):
                    for r, threshold_pred in enumerate(threshold_list_pred):
                        data_label_list_truth = convert_val_to_class(threshold_list_truth[k], data_list_truth).flatten()
                        data_label_list_pred = convert_val_to_class(threshold_list_pred[r], data_list_pred).flatten()
                        match_matrix = count_matrix(data_label_list_truth, data_label_list_pred, 2)
                        record = np.trace(match_matrix) / np.sum(match_matrix)
                        match_record[j][k][r] = record
            match_record[-1, :, :] = np.mean(match_record[:3], axis=0)
            np.save(save_path_npy, match_record)
            np.save(save_path_normal_position_npy, match_record_normal_position)

        draw_3d_surface_subplots(match_record, match_record_normal_position, save_path_plot)


def draw_3d_surface_subplots(data, data_normal, path):
    print(f"data shape: {data.shape}")
    print(f"data_normal: {data_normal}")

    x = np.arange(0, data.shape[2])
    x = np.asarray(x) / float(data.shape[2])
    y = np.arange(0, data.shape[1])
    y = np.asarray(y) / float(data.shape[1])
    Y, X = np.meshgrid(x, y)
    print(X)
    print(Y)
    print(X[55, 45], Y[55, 45], data[0][55, 45])
    title_names = [""] * 4#["APET", "TPET", "NPET", "Average"]

    # Create figure and axes
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=1.0)

    for i in range(data.shape[0]):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        Z = data[i]

        # ax.plot_surface(X, Y, data[i], cmap='viridis')
        # ax.plot_surface(X, Y, data[i], cmap='viridis')
        # ax.plot_wireframe(X, Y, data[i], linewidth=0.5, color='blue')

        if i != 3:
            yy = np.linspace(0, 1, 1000)
            zz = np.linspace(0, data_normal[i][2], 1000)
            yy, zz = np.meshgrid(yy, zz)
            xx = np.ones_like(yy) * data_normal[i][0]
            ax.plot_surface(xx, yy, zz, alpha=0.25, color="pink", zorder=1)

            xx = np.linspace(0, 1, 1000)
            zz = np.linspace(0, data_normal[i][2], 1000)
            xx, zz = np.meshgrid(xx, zz)
            yy = np.ones_like(xx) * data_normal[i][1]
            ax.plot_surface(xx, yy, zz, alpha=0.25, color="pink", zorder=1)

        if i != 3:
            plane_x = data_normal[i][0]
            idx_x = np.argmin(np.abs(x - plane_x))
            intersection_y = Y[idx_x, :]
            intersection_z = Z[idx_x, :]
            ax.plot([plane_x] * len(intersection_y), intersection_y, intersection_z, color='red', linewidth=1, alpha=0.8, zorder=7)

            plane_y = data_normal[i][1]
            idx_y = np.argmin(np.abs(y - plane_y))
            intersection_x = X[:, idx_y]
            intersection_z = Z[:, idx_y]
            ax.plot(intersection_x, [plane_y] * len(intersection_x), intersection_z, color='red', linewidth=1, alpha=0.8, zorder=7)

            print(Z[idx_x, idx_y])

            # ax.scatter(data_normal[i][0], data_normal[i][1], data_normal[i][2], s=20, facecolor='red',
            #            edgecolor='red', linewidth=1, zorder=10)
            # ax.text(data_normal[i][0], data_normal[i][1], data_normal[i][2] - 0.1, f'{data_normal[i][2] * 100:.1f}%',
            #         color='red', ha='left', zorder=5,
            #         path_effects=[path_effects.Stroke(linewidth=2, foreground='white'),
            #                       path_effects.Normal()])
            print(f"for i = {i}, texting {data_normal[i][2] * 100:.1f}%")

        ax.plot_surface(X, Y, data[i], cmap='viridis', alpha=0.5, zorder=3)
        ax.plot_wireframe(X, Y, data[i], linewidth=0.4, color='blue', alpha=0.7, zorder=3)




        ax.set_xlabel('Truth Threshold')
        ax.set_ylabel('Prediction Threshold')
        ax.set_zlabel('Match Score')
        ax.zaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.set_title(title_names[i])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.invert_xaxis()

        # Hide the surface
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Set the grid lines color
        ax.xaxis.line.set_color('black')
        ax.yaxis.line.set_color('black')
        ax.zaxis.line.set_color('black')

        ax.view_init(elev=40)

    # fig.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"3D match plot is saved to {path}")

def percent_formatter(x, pos):
    return f'{x*100:.0f}%'


if __name__ == "__main__":
    # old 230612
    # ct = ConstTruthSpatial(
    #     csf_folder_path="data/CSF/",
    #     pet_folder_path="data/PET/",
    #     # dataset="all"
    # )
    # save = np.load("saves/params_A2_2250.npy")
    # params = save[:46]
    # starts = save[-11:]
    # # from spatial_simulation import SPATIAL_DIFFUSION_CONST
    #
    # # diffusion_list = np.asarray([SPATIAL_DIFFUSION_CONST[i]["init"] for i in range(5)])
    # # diffusion_list = np.asarray([2.52e-03, 6.31e-06, 1.89e-04, 4.34e-04, 1.97e-06])
    # diffusion_list = np.asarray([3.15e-04, 2.52e-05, 1.54e-04, 5.64e-04, 1.97e-06])
    #
    # # print(save.shape)
    # loss_func_spatial(params, starts, diffusion_list, ct, False, True)

    # /old 230612


    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="n")
    parser.add_argument("--threshold", type=float, help="threshold")
    parser.add_argument("--diffusion_unit_rate", type=float, help="diffusion_unit_rate")
    opt = parser.parse_args()
    # ct = ConstTruthSpatial(
    #     csf_folder_path="data/CSF/",
    #     pet_folder_path="data/PET/",
    #     # dataset="all"
    #     args=opt,
    # )
    # 20230619_200950_834042: 1.0
    # 20230619_201113_923881: 1e4
    # !20230622_193403_226130*
    # !20230622_193407_389476*
    # !20230622_193411_110479*
    timestring_pairs = [["20230629_233108_721783", [2]]]#[["20230629_233108_721783", [2]], ["20230629_233112_710947", [2]], ["20230629_233116_626075", [2]]] # [["20230629_164606_360836", [2]], ["20230629_164614_745372", [2]], ["20230629_164635_457669", [2]]]  # [["20230622_193403_226130", [2]], ["20230622_193407_389476", [3]], ["20230622_193411_110479", [4]]]
    for one_pair in timestring_pairs:
        timestring = one_pair[0] #"20230619_195520_933683"#"20230619_185941_826305"# "20230619_185429_081843"#"20230619_173631_406302"#"20230619_172737_589445"#"20230619_172525_233765" # "20230619_170436_889472" #"20230619_165547_901048"#"20230619_163156_356005"#"20230619_161642_806991" #  "20230619_155848_438276"  # "20230614_225645_043020"  # "20230614_220438_197917"
        with open(f"simulation_output/{timestring}_pred_origin_all.pkl", "rb") as f:
            pred_ct_origin_all = pickle.load(f)
        with open(f"simulation_output/{timestring}_truth_origin.pkl", "rb") as f:
            truth_ct_origin = pickle.load(f)
        with open(f"simulation_output/{timestring}_pred_origin.pkl", "rb") as f:
            pred_ct_origin = pickle.load(f)
        with open(f"simulation_output/{timestring}_match_matrix.pkl", "rb") as f:
            match_matrix = pickle.load(f)
        print(f"n = {one_pair[1][0]}")
        rate_sum = 0.0
        for i in range(3):
            print(np.array2string(match_matrix[i], formatter={'int': '{:3d}'.format}))
            print(f"{np.trace(match_matrix[i])} / {np.sum(match_matrix[i])} = {np.trace(match_matrix[i]) / np.sum(match_matrix[i]) * 100:.4f} %")
            rate_sum += np.trace(match_matrix[i]) / np.sum(match_matrix[i])
        print(f"Average Accuracy: {rate_sum / 3}")
        print()


        # print(ct.y["APET"].shape)
        # print(ct.x)
        for n in [2, 3, 4]:#one_pair[1]:#[2, 3, 4]:
            # draw_spatial_truth(truth_ct_origin, f"test/output_{timestring}_truth_origin_n={n}.png", n=n, draw_type="truth")
            # draw_spatial_truth(pred_ct_origin, f"test/output_{timestring}_pred_origin_n={n}.png", n=n, draw_type="pred", ct_pred_all=pred_ct_origin_all)

            print(f"For n={n}:")
            line_labels = ["APET", "TPET", "NPET"]
            for i, one_label in enumerate(line_labels):
                data_list_truth = truth_ct_origin.y[one_label].flatten()
                data_list_pred = pred_ct_origin.y[one_label].flatten()

                threshold_list_truth = get_classification_threshold_list(n=n, data_list=data_list_truth)
                threshold_list_pred = get_classification_threshold_list(n=n, data_list=data_list_pred)

                data_label_list_truth = convert_val_to_class(threshold_list_truth, data_list_truth)
                data_label_list_pred = convert_val_to_class(threshold_list_pred, data_list_pred)

                match_matrix = count_matrix(data_label_list_truth, data_label_list_pred, n)
                record = np.trace(match_matrix) / np.sum(match_matrix)
                print("match_matrix:", match_matrix)
                print(f"score: {record}")

        # output_ct_to_txt_for_brain_surface(truth_ct_origin, pred_ct_origin, timestring)
        one_time_draw_3D_match_result(truth_ct_origin, pred_ct_origin, timestring)

    # print(convert_val_to_class([5,7,11,999,float("inf")], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]))

    # print(get_now_string())

    # record1 = loss_func(np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)]), ct)
    # print(record1)
    # print("hhhh")
#    params = np.load("saves/params_20221205_221101.npy")
#    params = np.load("saves/params_20221129_201557.npy")
#     params = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
#    params = np.load("saves/params_20221202_095940.npy")

# print(len(params))
# np.save("saves/params_default_46.npy", params)

# p0 = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
# record1 = loss_func(p0, ct)
# print(record1)
# run(p0)


# p = np.load("saves/params_20221103_090002.npy")
# record2 = loss_func(p, ct)
# print(record2)
# mt = MyTime()
# run(params)
# mt.print()