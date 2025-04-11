import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from ode_truth import ConstTruthSpatial, loss_func_spatial_match, draw_spatial_truth, get_now_string, ADSolver, \
    ConstTruth


# For Yang Xiao

def one_time_re_simulate():
    base_params = np.load("saves/params_20230314_185400_648329_modified_10.npy")
    new_params = base_params


    ##adjust the secretion params respectively for the abnormality graph figure3
    # new_params[5]=new_params[5]*3.5 #k_sA
    # new_params[17]=new_params[17]*3.5 #k_sTp
    abnor_graph_name = "test"

    assert len(new_params) == len(base_params)

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", default=0.5, type=float, help="threshold")
    parser.add_argument("--diffusion_unit_rate", default=1.0, type=float, help="diffusion_unit_rate")
    opt = parser.parse_args()
    ct = ConstTruthSpatial(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/",
        args=opt,
    )
    diffusion_list = np.asarray([1.942913e-02, 1.261440e-05, 1.068486e-04, 4.294134e-02,
                                 2.365200e-06])  # ["d_Am", "d_Ao", "d_Tm", "d_Tp", "d_To"]
    new_params[21] = new_params[21] * 50
    new_params[22] = new_params[22] * 150
    ct_pred_origin_all, ct_truth_origin, ct_pred_origin, record_mat_all, record_mat_sum, record_rate_list = loss_func_spatial_match(
        new_params[:49], new_params[49:60], diffusion_list, ct, "simulation_output/test.csv", silent=True,
        save_flag=False, element_id=-1, output_flag=True, n_class=2)
    print(f"pred structure - x: {ct_pred_origin.x}")
    print(f"pred structure - y: {ct_pred_origin.y}")
    print(type(ct_pred_origin.y))



    STAGE = np.array(["CN", "SMC", "EMCI", "LMCI", "AD"])
    Biomarkers = ["APET", "TPET", "NPET"]



    ad = ADSolver("CN", ct)
    ad.step_spatial(new_params[:49], new_params[49:60], diffusion_list)
    output = np.asarray(ad.get_output()).reshape([12, 1201])

    ct_seven_lines = ConstTruth(
        csf_folder_path="data/CSF/",
        pet_folder_path="data/PET/",
        dataset="all",
        start="ranged",
        tcsf_scaler=1.0,
    )
    # ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]

    #names = ["APET", "TPET", "NPET", "ACSF", "TpCSF", "TCSF", "TtCSF"]  # this is for resub
    names = ['PET Aβ', 'PET Tau', 'NPET', 'CSF Aβ', 'CSF p-Tau', 'TCSF', 'TtCSF'] #this is for other else
    data_points = [300, 600, 900, 1100, 1200]
    legend = ['k_sA*1', 'k_sTp*1']
    draw_all = [0, 1, 3, 4]
    plot_index = [0, 1, 4]
    reverse_plot_index = [3]

    sliced_output = output[:7]

    # Save pred values as a separate NumPy file for draw_all for figure3
    N_array=sliced_output
    for row_index in range(len(draw_all)):
        row_data = N_array[draw_all[row_index], :]
        filename = f"figure/Abnormality/test/{names[draw_all[row_index]]}_{abnor_graph_name}.npy"
        np.save(filename, row_data)







# this method is intend to generate a new biomarkers' Abnormality graph
def load_and_normalize_data(file_path, reverse=False, threshold=0.02):
    mean_values = np.load(file_path)
    min_value, max_value = min(mean_values), max(mean_values)
    normalized_data = [(val - min_value) / (max_value - min_value) for val in mean_values]

    if reverse:
        normalized_data = [1 - val for val in normalized_data]

    # Filter data points near 0
    filtered_data = [(t, val) for t, val in zip(range(len(normalized_data)), normalized_data) if val > threshold]

    return filtered_data


def plot_data(t_value, filtered_data, label, color, linestyle, linewidth):
    t_values, data = zip(*filtered_data)  # Unpack time and value
    plt.plot(t_values, data, color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def plot_data_trans(t_value, filtered_data, label, color, linestyle, linewidth, alpha):
    t_values, data = zip(*filtered_data)
    plt.plot(t_values, data, color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha)


if __name__ == "__main__":
    one_time_re_simulate()

    # draw new ab graph
    colors = {'PET Aβ': '#f55742', 'PET Tau': '#52AE5E', 'CSF Aβ': '#f55742',
              'CSF p-Tau': '#52AE5E'}  # Updated colors rbrb
    linestyles = {'PET Aβ': '-', 'PET Tau': '-', 'CSF Aβ': '--', 'CSF p-Tau': '--'}  # Dashed line styles
    reverse = {'PET Aβ': False, 'PET Tau': False, 'CSF Aβ': True, 'CSF p-Tau': False}
    t_values = np.arange(1201)
    biomarkers = ['PET Aβ', 'PET Tau', 'CSF Aβ', 'CSF p-Tau']

    # for biomarker in biomarkers:
    #     file_path = f'figure/Abnormality/{biomarker}_original.npy'
    #     normalized_data = load_and_normalize_data(file_path,reverse[biomarker])
    #     plot_data(t_values, normalized_data, biomarker, colors[biomarker], linestyles[biomarker], 3.5)

    # draw only ab lines graph
    ab_biomarkers = ['PET Aβ', 'CSF Aβ']
    ab_biomarkers_A = ['PET Aβ_A', 'CSF Aβ_A']
    sec_color_A = {0.25: '#F59542', 4: '#D84D3A', 1.5: '#f55742', 2: '#f55742', 2.5: '#f55742', 3: '#f55742',
                   3.5: '#f55742'}  # F57642
    secretion = [0.25, 4]
    secretion_trans = [1.5, 2, 2.5, 3, 3.5]
    for biomarker in ab_biomarkers:
        file_path = f'figure/Abnormality/{biomarker}_original.npy'
        normalized_data = load_and_normalize_data(file_path, reverse[biomarker])
        plot_data(t_values, normalized_data, biomarker, colors[biomarker], linestyles[biomarker], 2)

    for biomarkera, biomarker in zip(ab_biomarkers_A, ab_biomarkers):
        for value in secretion:
            print(biomarkera)
            file_path = f'figure/Abnormality/{biomarkera}{value}.npy'
            normalized_data = load_and_normalize_data(file_path, reverse[biomarker])
            plot_data(t_values, normalized_data, biomarker, sec_color_A[value], linestyles[biomarker], 2)

    # add trans lines between
    for biomarkera, biomarker in zip(ab_biomarkers_A, ab_biomarkers):
        for value in secretion_trans:
            file_path = f'figure/Abnormality/{biomarkera}{value}.npy'
            normalized_data = load_and_normalize_data(file_path, reverse[biomarker])
            plot_data_trans(t_values, normalized_data, biomarker, sec_color_A[value], linestyles[biomarker], 1, 0.3)

    plt.xticks([])
    plt.yticks([])
    plt.show()

    # draw only tau lines graph
    tau_biomarkers = ['PET Tau', 'CSF p-Tau']
    tau_biomarkers_T = ['PET Tau_Tp', 'CSF p-Tau_Tp']
    sec_color_T = {0.25: '#6AE77B', 4: '#28562E', 1.5: '#52AE5E', 2: '#52AE5E', 2.5: '#52AE5E', 3: '#52AE5E',
                   3.5: '#52AE5E'}
    secretion = [0.25, 4]
    secretion_trans = [1.5, 2, 2.5, 3, 3.5]
    for biomarker in tau_biomarkers:
        file_path = f'figure/Abnormality/{biomarker}_original.npy'
        normalized_data = load_and_normalize_data(file_path, reverse[biomarker])
        plot_data(t_values, normalized_data, biomarker, colors[biomarker], linestyles[biomarker], 2)

    for biomarkera, biomarker in zip(tau_biomarkers_T, tau_biomarkers):
        for value in secretion:
            print(biomarkera)
            file_path = f'figure/Abnormality/{biomarkera}{value}.npy'
            normalized_data = load_and_normalize_data(file_path, reverse[biomarker])
            plot_data(t_values, normalized_data, biomarker, sec_color_T[value], linestyles[biomarker], 2)

    # add trans lines between
    for biomarkera, biomarker in zip(tau_biomarkers_T, tau_biomarkers):
        for value in secretion_trans:
            file_path = f'figure/Abnormality/{biomarkera}{value}.npy'
            normalized_data = load_and_normalize_data(file_path, reverse[biomarker])
            plot_data_trans(t_values, normalized_data, biomarker, sec_color_T[value], linestyles[biomarker], 1, 0.3)

    plt.xticks([])
    plt.yticks([])
    plt.show()



