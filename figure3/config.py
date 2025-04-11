import numpy as np
import os

from const import LABEL_ID


class Config:
    def __init__(self, laplacian_threshold):
        self.N_dim = 160
        self.laplacian_threshold = laplacian_threshold
        # ["CN", "SMC", "EMCI", "LMCI", "AD"]
        self.Ls = [np.load("data/Laplacian_classes/network_laplacian_{}_{}.npy".format(one_class, self.laplacian_threshold))  for one_class in ["CN", "SMC", "EMCI", "LMCI", "AD"]]  ## 0619 debug


class Start:
    def __init__(self, class_name=None, pet_data_path="data/PET/", csf_data_path="data/CSF/"):
        # assert class_name in ["CN", "SMC", "EMCI", "LMCI", "AD"], "param class_name must in ['CN', 'SMC', 'EMCI', 'LMCI', 'AD'], but got \"{}\"!".format(class_name)
        assert class_name in ["CN"], "param class_name must in ['CN', 'SMC', 'EMCI', 'LMCI', 'AD'], but got \"{}\"!".format(class_name)
        self.class_name = class_name
        self.config = Config(0.1)
        np.random.seed(467)
        csf_data = np.load(os.path.join(csf_data_path, "CSF_{}.npy".format(self.class_name)))
        Am = np.random.uniform(1e-3, 5e-3, size=self.config.N_dim)
        Am_avg = np.mean(Am).reshape(1)
        Ao = np.random.uniform(0, 1e-3, size=self.config.N_dim)
        Ao_avg = np.mean(Ao).reshape(1)
        Af = np.load(os.path.join(pet_data_path, "PET-A_{}.npy".format(self.class_name)))
        Af = Af * 1e-3
        # randomize
        # Af = np.random.uniform(np.min(Af), np.max(Af), size=self.config.N_dim)
        # Af = Af * 1e-2  # 1e-4
        Af_avg = np.mean(Af).reshape(1)
        ACSF = np.expand_dims(csf_data[0], axis=0)  # 0.14 * np.ones(1)
        ACSF = ACSF * 1e-2
        Tm = np.random.uniform(1e-4, 3e-4, size=self.config.N_dim)  ##1020 TAU concentration in neuronal cells is around 2uM - AD26
        Tm_avg = np.mean(Tm).reshape(1)
        Tp = np.random.uniform(0, 1e-4, size=self.config.N_dim)
        Tp_avg = np.mean(Tp).reshape(1)
        To = np.random.uniform(0, 1e-4, size=self.config.N_dim)
        To_avg = np.mean(To).reshape(1)
        Tf = np.load(os.path.join(pet_data_path, "PET-T_{}.npy".format(self.class_name)))
        # randomize
        # Tf = np.random.uniform(np.min(Tf), np.max(Tf), size=self.config.N_dim)
        Tf = Tf * 2 * 1e-4 #Tf*2*1e-4
        Tf_avg = np.mean(Tf).reshape(1)
        TCSF = np.expand_dims(csf_data[1] - csf_data[2], axis=0)  # 0.19 * np.ones(1)
        TCSF = TCSF * 1e-5  # 1e-5
        TpCSF = np.expand_dims(csf_data[2], axis=0)  # 0.20 * np.ones(1)
        TpCSF = TpCSF * 1e-5  # 1e-5
        N = np.load(os.path.join(pet_data_path, "PET-N_{}.npy".format(self.class_name)))

        N = np.max(N) + np.min(N) - N

        # N = np.random.uniform(np.min(N), np.max(N), size=self.config.N_dim)
        N_avg = np.mean(N).reshape(1)

        self.all = np.concatenate([Am, Ao, Af, ACSF, Tm, Tp, To, Tf, TCSF, TpCSF, N])
        # self.all = np.concatenate([Am_avg, Ao_avg, Af_avg, ACSF, Tm_avg, Tp_avg, To_avg, Tf_avg, TCSF, TpCSF, N_avg])
        # print(f"debug 0619: start y0 = {np.concatenate([Am_avg, Ao_avg, Af_avg, ACSF, Tm_avg, Tp_avg, To_avg, Tf_avg, TCSF, TpCSF, N_avg])}")

# class Start:
#     Am = np.random.rand(Config.N_dim)  # 0.11 * np.ones([Config.N_dim])
#     Ao = np.random.rand(Config.N_dim)   # 0.12 * np.ones([Config.N_dim])
#     Af = np.random.rand(Config.N_dim)   # 0.13 * np.ones([Config.N_dim])
#     ACSF = np.random.rand(1)  # 0.14 * np.ones(1)
#     Tm = np.random.rand(Config.N_dim)   # 0.15 * np.ones([Config.N_dim])
#     Tp = np.random.rand(Config.N_dim)   # 0.16 * np.ones([Config.N_dim])
#     To = np.random.rand(Config.N_dim)   # 0.17 * np.ones([Config.N_dim])
#     Tf = np.random.rand(Config.N_dim)   # 0.18 * np.ones([Config.N_dim])
#     TCSF = np.random.rand(1)  # 0.19 * np.ones(1)
#     TpCSF = np.random.rand(1)  # 0.20 * np.ones(1)
#     N = np.random.rand(Config.N_dim)   # 0.21 * np.ones([Config.N_dim])
#     all = np.concatenate([Am, Ao, Af, ACSF, Tm, Tp, To, Tf, TCSF, TpCSF, N])

if __name__ == "__main__":
    Tf = np.load(os.path.join("data/PET/", "PET-T_{}.npy".format("CN")))
    Af = np.load(os.path.join("data/PET/", "PET-A_{}.npy".format("CN")))
    N = np.load(os.path.join("data/PET/", "PET-N_{}.npy".format("CN")))
    # Tf = np.mean(Tf).reshape(1)
    print("Af: {}".format(np.mean(Af)))
    print("Tf: {}".format(np.mean(Tf)))
    print("N: {}".format(np.mean(N)))
    csf_data = np.load(os.path.join("data/CSF/", "CSF_{}.npy".format("CN")))
    print("ACSF: {}".format(csf_data[0]))
    print("TpCSF: {}".format(csf_data[2]))
    print("TCSF: {}".format(csf_data[1] - csf_data[2]))
    # print("TtCSF: {}".format(csf_data[1]))



