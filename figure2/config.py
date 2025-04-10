import numpy as np
import os


class Config:
    N_dim = 160
    L = np.zeros([N_dim, N_dim])


class Start:
    def __init__(self, class_name=None, tcsf_scaler=1.0, pet_data_path="data/PET/", csf_data_path="data/CSF/"):
        assert class_name in ["CN", "SMC", "EMCI", "LMCI", "AD"], "param class_name must in ['CN', 'SMC', 'EMCI', 'LMCI', 'AD'], but got \"{}\"!".format(class_name)
        self.class_name = class_name
        csf_data = np.load(os.path.join(csf_data_path, "CSF_{}.npy".format(self.class_name)))
        Am = np.random.uniform(1e-3, 5e-3, size=Config.N_dim)
        Am_avg = np.mean(Am).reshape(1)
        Ao = np.random.uniform(0, 1e-3, size=Config.N_dim)
        Ao_avg = np.mean(Ao).reshape(1)
        Af = np.load(os.path.join(pet_data_path, "PET-A_{}.npy".format(self.class_name)))
        Af = Af * 1e-3
        Af_avg = np.mean(Af).reshape(1)
        ACSF = np.expand_dims(csf_data[0], axis=0)  # 0.14 * np.ones(1)
        ACSF = ACSF * 1e-2
        Tm = np.random.uniform(1e-4, 3e-4,
                               size=Config.N_dim)  ##1020 TAU concentration in neuronal cells is around 2uM - AD26
        Tm_avg = np.mean(Tm).reshape(1)
        Tp = np.random.uniform(0, 1e-4, size=Config.N_dim)
        Tp_avg = np.mean(Tp).reshape(1)
        To = np.random.uniform(0, 1e-4, size=Config.N_dim)
        To_avg = np.mean(To).reshape(1)
        Tf = np.load(os.path.join(pet_data_path, "PET-T_{}.npy".format(self.class_name)))
        Tf = Tf * 2 * 1e-4
        Tf_avg = np.mean(Tf).reshape(1)
        TCSF = np.expand_dims(csf_data[1] - csf_data[2], axis=0)  # 0.19 * np.ones(1)
        TCSF = TCSF * 1e-5  # 1e-5
        TpCSF = np.expand_dims(csf_data[2], axis=0)  # 0.20 * np.ones(1)
        TpCSF = TpCSF * 1e-5  # 1e-5
        N = np.load(os.path.join(pet_data_path, "PET-N_{}.npy".format(self.class_name)))
        N_avg = np.mean(N).reshape(1)
        self.all = np.concatenate([Am_avg, Ao_avg, Af_avg, ACSF, Tm_avg, Tp_avg, To_avg, Tf_avg, TCSF, TpCSF, N_avg])


if __name__ == "__main__":
    pass



