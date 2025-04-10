import numpy as np
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import os
import pickle
import json
from scipy.integrate import odeint

from const import PARAM_NUM, STARTS_NUM, PARAMS, STARTS_WEIGHTS, LABEL_LIST
from config import Config, Start

# from ode_truth import loss_func
from utils import ColorCandidate, MultiSubplotDraw, get_patient_wise_list

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
        assert "option" in params
        assert "tcsf_scaler" in params
        self.params = params
        csf_folder_path, pet_folder_path = params["csf_folder_path"], params["pet_folder_path"]
        label_list = LABEL_LIST  # [[0, 2, 3, 4]]  # skip the second nodes (SMC)
        self.class_num = len(label_list)
        self.lines = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]
        self.y = dict()
        self.x = dict()
        self.y_original = dict()
        self.x_original = dict()
        self.increase_rate = np.zeros(7)
        assert params["start"] in ["ranged", "fixed"] or "ranged" in params["start"]
        assert params["dataset"] in ["all", "chosen_0", "rebuild"]
        if "x" not in params:
            if params["start"] in ["ranged", "rebuild"] or "ranged" in params["start"]:
                self.x_all = np.asarray([3, 6, 9, 11, 12])
            else:
                self.x_all = np.asarray([0, 3, 6, 8, 9])
        else:
            self.x_all = np.asarray(params.get("x"))
        if params["dataset"] == "rebuild":
            with open("data/rebuild_truth.pkl", "rb") as f:
                rebuild_dic = pickle.load(f)
            for one_line in self.lines:
                if one_line in rebuild_dic:
                    self.y[one_line] = rebuild_dic[one_line]["y"]
                    self.x[one_line] = rebuild_dic[one_line]["x"]
                else:
                    self.y[one_line] = np.asarray([])
                    self.x[one_line] = np.asarray([])
            with open("data/plots.json", "r") as f:
                plot_json = json.load(f)
            for one_line in self.lines:
                self.y_original[one_line] = np.asarray(plot_json["truth_plot"][one_line]["y"])
                self.x_original[one_line] = np.asarray(plot_json["truth_plot"][one_line]["x"])
            return


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

            self.y["ACSF"] = self.y["ACSF"] + [csf_data[0]]
            self.y["TtCSF"] = self.y["TtCSF"] + [csf_data[1]]
            self.y["TpCSF"] = self.y["TpCSF"] + [csf_data[2]]
            self.y["TCSF"] = self.y["TCSF"] + [csf_data[1] - csf_data[2]]
        for i, one_key in enumerate(self.lines):
            self.y[one_key] = np.asarray(self.y[one_key])
            self.increase_rate[i] = (self.y[one_key][-1] - self.y[one_key][0]) / self.y[one_key][0]
        # print("increse rate: {}".format(self.increase_rate))
        # self.y["NPET"] = 2.0 - self.y["NPET"]  # 1.0 - (self.y["NPET"] - np.min(self.y["NPET"])) / (np.max(self.y["NPET"]) - np.min(self.y["NPET"]))

        if params["dataset"] == "chosen_0":
            for one_key in ["NPET"]:
                self.y[one_key] = self.y[one_key]  # [[]]
                self.x[one_key] = self.x[one_key]  # [[]]
            for one_key in ["ACSF", "TCSF", "TtCSF"]:
                self.y[one_key] = self.y[one_key]
                self.x[one_key] = self.x[one_key]
            for one_key in ["TpCSF"]:
                self.y[one_key] = self.y[one_key]
                self.x[one_key] = self.x[one_key]
        else:
            for one_key in ["NPET"]:
                self.y[one_key] = self.y[one_key]
                self.x[one_key] = self.x[one_key]
            pass

        self.pet_pickle_obj = None
        self.csf_pickle_obj = None


class TMPADSolver:
    def __init__(self, class_name, const_truth=None):
        self.n = 1  # Config.N_dim
        self.L = Config.L
        self.T = 12.01
        self.T_unit = 0.01
        self.t = np.linspace(0.0, self.T - self.T_unit, int(self.T / self.T_unit))  # expand time
        self.class_name = class_name
        self.const_truth = const_truth
        self.y0 = Start(class_name, tcsf_scaler=self.const_truth.params["tcsf_scaler"]).all

        self.lines = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]
        self.output_names = ["A-PET", "T-PET", "N-PET", "A-CSF", "Tp-CSF", "T-CSF",
                             "Tt-CSF"]
        self.output_names_rest = ["$A_{m} Avg$", "$T_{m} Avg$", "$A_{o} Avg$", "$T_{o} Avg$", "$T_{p} Avg$"]
        self.n_color = 12
        self.colors = ["blue"] * 20 #ColorCandidate().get_color_list(self.n_color, light_rate=0.5)
        self.y = None
        self.output = None
        self.params = None
        self.starts_weights = None
        self.truth_ylim = dict()
        self.predict_ylim = dict()
        self.tol = 1e-4


    def step(self, _params=None, _starts_weights=None):
        if _params is not None:
            self.params = np.asarray(_params)
            self.starts_weights = np.asarray(_starts_weights)
        else:
            self.params = np.asarray([PARAMS[i]["init"] for i in range(PARAM_NUM)])
            self.starts_weights = np.asarray([STARTS_WEIGHTS[i]["init"] for i in range(STARTS_NUM)])
            print("Params & starts_weights are not given. Using the initial value instead to simulate ...")
        self.y0 = self.y0 * self.starts_weights
        self.y = odeint(self.pend, self.y0, self.t, rtol=self.tol, atol=self.tol)
        self.output = self.get_output()

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
        NPET = 0.9 - (NPET - np.min(NPET)) / (np.max(NPET) - np.min(NPET)) * 0.1
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

    def pend(self, y, t):
        # mt.time_start()
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


        k_p1Am, k_p2Am, k_dAm, k_diA, k_cA, k_sA, k_dAo, k_yA, k_pTm, k_dTm, k_ph1, k_ph2, k_deph, k_diT, k_cT, k_sT, k_dTp, k_sTp, k_dTo, k_yT, k_yTp, k_AN, k_TN, k_a1A, k_a2A, k_a1T, k_a2T, K_mTA, K_mAT, K_mAN, K_mTN, K_mT2, K_mA2, n_TA, n_cA, n_AT, n_cT, n_cTp, n_cTo, n_AN, n_TN, n_a1A, n_a2A, n_a1T, n_a2T, n_a1Tp, n_a2Tp, k_acsf, k_tcsf \
            = iter(self.params)


        d_Am = 0.0  # 5.0
        d_Ao = 0.0
        d_Tm = 0.0  # 5.0
        d_Tp = 0.0  # 5.0
        d_To = 0.0

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
        if self.const_truth.params["option"] == "option1":
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


        Tf_ = k_cT * numpy_safe_pow(Tm, n_cT) * numpy_safe_pow(Tp, n_cTp) * numpy_safe_pow(To, n_cTo)

        TCSF_ = k_sT * sum_func(Tm) - k_yT * TCSF

        TpCSF_ = k_sTp * sum_func(Tp) - k_yTp * TpCSF

        N_ = k_AN * 1.0 / (numpy_safe_pow(K_mAN, n_AN) / numpy_safe_pow((Ao + Af), n_AN) + 1.0) + k_TN * 1.0 / (
                numpy_safe_pow(K_mTN, n_TN) / numpy_safe_pow((To + Tf), n_TN) + 1.0)
        dy = np.concatenate([Am_, Ao_, Af_, ACSF_, Tm_, Tp_, To_, Tf_, TCSF_, TpCSF_, N_])
        # # print(dy.shape)
        # mt.time_end()
        return dy

    def draw(self, opt, save_flag=True, time_string=None):
        folder_path = "figure/{0}{1}/".format(
            "{}_".format(opt.model_name) if opt.model_name != "none" else "",
            time_string)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_path_target = os.path.join(folder_path, "figure_target_{}.png".format(time_string))
        save_path_rest = os.path.join(folder_path, "figure_rest_{}.png".format(time_string))
        # m = MultiSubplotDraw(row=3, col=3, fig_size=(24, 18), tight_layout_flag=True, show_flag=False,
        #                      save_flag=save_flag,
        #                      save_path=save_path_target, save_dpi=200)

        plot_dic = {
            "APET": [0, 1, 2, 3, 4],
            "TPET": [0, 1, 2, 3, 4],
            "NPET": [0, 1, 2, 3, 4],
            "ACSF": [0, 2, 3, 4],
            "TpCSF": [0, 2, 4],
            "TCSF": [0, 2, 3, 4],
            "TtCSF": [0, 2, 3, 4],
        }

        for i, (name, data, color, line_string) in enumerate(zip(self.output_names, self.output[:len(self.output_names)], self.colors[:len(self.output_names)], self.lines)):
            y_lists = data[:, :-int(3.0 / self.T_unit)] if self.const_truth.params["start"] == "fixed" else data
            fig = plt.figure()
            ax2 = fig.add_subplot(1, 1, 1)
            ax = ax2.twinx()
            y_lists = y_lists.flatten()
            ax.plot(self.t, y_lists, linewidth=2, c="black", linestyle="solid", zorder=1)
            # ax.set_title(name, fontsize=20)
            save_path = save_path_target.replace(".png", "_{}.png".format(line_string))

            ax.set_yticks([])

            self.predict_ylim[self.lines[i]] = list(ax.get_ylim())

            if self.const_truth:
                x = self.const_truth.x[line_string]
                y = self.const_truth.y[line_string]

                for k in range(5):
                    ax2.scatter(x=x[k], y=y[k], s=100, facecolor='red' if k in plot_dic[line_string] else 'white', marker="o", edgecolors='red', linewidths=1,
                                zorder=2)
                # ax2.set_xlabel("time", fontsize=15)
                # ax2.set_ylabel("val", fontsize=15)
                if self.const_truth.params["dataset"] == "rebuild":
                    ax2.scatter(x=self.const_truth.x_original[line_string], y=self.const_truth.y_original[line_string], s=100, facecolor="none", marker="d", edgecolors='#00ff00', linewidths=3, zorder=2)


                ax2.tick_params(axis='y', labelcolor="black", labelsize=20)
                ax2.tick_params(axis='x', labelcolor="black", labelsize=20)
                # ax2.tick_params(axis='y', which='both', colors='blue')
                if "ranged" in self.const_truth.params["start"]:
                    ylim_bottom, ylim_top = ax2.get_ylim()
                    index_fixed = (x / self.T_unit).astype(int)
                    curve_data = data[0][index_fixed]
                    if line_string in ["ACSF", "NPET"]:
                        ax2.set_ylim([ylim_bottom, ylim_bottom + (ylim_top - ylim_bottom) / (np.max(curve_data) - np.min(curve_data)) * (np.max(data[0]) - np.min(curve_data))])
                        # ax2.set_ylim([ylim_bottom, ylim_bottom + (ylim_top - ylim_bottom) / (data[0][int(x[0] / self.T_unit)] - data[0][int(x[-1] / self.T_unit)]) * (data[0][0] - data[0][int(x[-1] / self.T_unit)])])
                    elif line_string in ["APET", "TPET", "TpCSF", "TCSF", "TtCSF"]:
                        ax2.set_ylim([ylim_top - (ylim_top - ylim_bottom) / (np.max(curve_data) - np.min(curve_data)) * (np.max(curve_data) - np.min(data[0])), ylim_top])
                        # ax2.set_ylim([ylim_top - (ylim_top - ylim_bottom) / (data[0][int(x[0] / self.T_unit)] - data[0][int(x[-1] / self.T_unit)]) * (data[0][0] - data[0][int(x[-1] / self.T_unit)]), ylim_top])
                self.truth_ylim[self.lines[i]] = list(ax2.get_ylim())
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

        print("Figure are saved to {}".format(folder_path))


def create_parameters_csv(param_full, csv_path):
    num_params = 49
    num_starting_weights = 11
    assert len(param_full) == num_params + num_starting_weights
    type_list = ["params"] * num_params + ["starting weights"] * num_starting_weights
    f = open(csv_path, "w")
    f.write("id,name,value,type,lower bound,upper bound,initial value\n")
    for i in range(60):
        if i < num_params:
            f.write("{},{},{},{},{},{},{}\n".format(
                i,
                PARAMS[i]["name"],
                param_full[i],
                type_list[i],
                PARAMS[i]["lb"],
                PARAMS[i]["ub"],
                PARAMS[i]["init"],
            ))
        else:
            f.write("{},{},{},{},{},{},{}\n".format(
                i,
                STARTS_WEIGHTS[i - num_params]["name"],
                param_full[i],
                type_list[i],
                STARTS_WEIGHTS[i - num_params]["lb"],
                STARTS_WEIGHTS[i - num_params]["ub"],
                STARTS_WEIGHTS[i - num_params]["init"],
            ))



def run(params=None, starts=None, time_string=None, opt=None):
    if not time_string:
        time_string = get_now_string()
    print("Time String (as folder name): {}".format(time_string))

    class_name = "CN"
    if opt is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str,  default="chosen_0", choices=["chosen_0", "all", "rebuild"], help="dataset strategy")
        parser.add_argument("--start", type=str, default="ranged", choices=["ranged", "ranged*", "fixed"], help="start strategy")   
        parser.add_argument("--generation", type=int, default=1000, help="generation, default: 1000")
        parser.add_argument("--pop_size", type=int, default=50, help="pop_size, default: 50")
        parser.add_argument("--model_name", default="none", type=str, help="model_name, can be any string")
        parser.add_argument("--option", type=str, default="option1", choices=["option1", "option1*", "option2"], help="option")
        parser.add_argument("--tcsf_scaler", type=float, default=0.3, help="tcsf_scaler, e.g., 0.3, 0.4, 0.5")
        parser.add_argument("--init_path", type=str, default=None, help="init_path")
        opt = parser.parse_args()
    ct = ConstTruth(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/",
        dataset=opt.dataset,
        start=opt.start,
        option=opt.option,
        tcsf_scaler=opt.tcsf_scaler,
    )
    truth = TMPADSolver(class_name, ct)
    truth.step(params, starts)
    # loss, csf_rate_loss = loss_func(params, starts, ct)
    # print("loss: {}".format(sum(loss) + csf_rate_loss))
    # print("loss parts: {} csf match loss: {}".format(list(loss), csf_rate_loss))
    truth.draw(opt, time_string=time_string)

    # csv_path = "figure/{}/{}.csv".format(time_string, time_string)
    # create_parameters_csv(np.concatenate([params, starts]), csv_path)
    return truth


if __name__ == "__main__":
    full_params = np.load("params/params_20230314_185400_648329.npy") 
    params = full_params[:PARAM_NUM]
    starts = full_params[-STARTS_NUM:]
    run(params, starts)