import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress, t
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.ticker import MaxNLocator

#figure5
def build_node(color_array, size_array, save_path, length=160):
    assert len(size_array) == len(color_array) == length
    df_dress = pd.read_excel("data/destriux_160.xlsx")
    with open(save_path, "w") as f:
        for i in range(length):
            f.write("{} {} {} {} {} {}\n".format(
                str(df_dress["x"][i]),
                str(df_dress["y"][i]),
                str(df_dress["z"][i]),
                str(color_array[i]),
                str(size_array[i]),
                str(df_dress["node"][i]),
            ))


def plot_with_color_partial_tau(arrays1, arrays2, indices):
    # Concatenate all array1s and array2s
    concatenated_array1 = np.concatenate(arrays1)
    concatenated_array2 = np.concatenate(arrays2)

    # Ensure the arrays are of the correct length
    assert len(concatenated_array1) == len(concatenated_array2) == 160 * len(indices)

    # Calculate the regression
    slope, intercept, r_value, _, _ = linregress(concatenated_array1, concatenated_array2)

    r_squared = r_value ** 2
    print(f"tau R² = {r_squared:.4f}")

    # Predicted values from the regression
    predicted = slope * concatenated_array1 + intercept

    # Plot each pair using a single color and give data points a border, with triangle markers
    for array1, array2 in zip(arrays1, arrays2):
        plt.scatter(array1, array2, c='#52ccaf', edgecolors='none', marker='^', s=100)  # Use '^' for triangle markers

    # Plot the regression line
    plt.plot(concatenated_array1, predicted, color='black', linewidth=2)

    # Adjust legend handling here as needed

    # Adjusting axis and layout
    ax = plt.gca()  # Get the current axis
    ax.xaxis.set_major_locator(MaxNLocator(5))
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.show()






def plot_with_color_partial_amy(arrays1, arrays2, indices):
    # Concatenate all array1s and array2s
    concatenated_array1 = np.concatenate(arrays1)
    concatenated_array2 = np.concatenate(arrays2)

    # Ensure the arrays are of the correct length
    assert len(concatenated_array1) == len(concatenated_array2) == 160 * len(indices)

    # Calculate the regression
    slope, intercept, r_value, _, _ = linregress(concatenated_array1, concatenated_array2)

    # Calculate R^2
    r_squared = r_value ** 2
    print(f"amyloid R² = {r_squared:.4f}")

    # Predicted values from the regression
    predicted = slope * concatenated_array1 + intercept

    # Plot each pair using a single color and give data points a border, with triangle markers
    for array1, array2 in zip(arrays1, arrays2):
        plt.scatter(array1, array2, c='#eb4034', edgecolors='none', marker='^', s=100)  # Use '^' for triangle markers

    # Plot the regression line
    plt.plot(concatenated_array1, predicted, color='black', linewidth=2)

    # Adjust legend handling here as needed

    # Adjusting axis and layout
    ax = plt.gca()  # Get the current axis
    ax.xaxis.set_major_locator(MaxNLocator(5))
    plt.tick_params(axis='both', which='major', labelsize=16)

    plt.show()






def plot_separate_brain_regions_CI(arrays1, arrays2, color_file, indices, confidence=0.95):
    # Concatenate all array1s and array2s
    concatenated_array1 = np.concatenate(arrays1)
    concatenated_array2 = np.concatenate(arrays2)

    # Load brain region colors and repeat to match the concatenated arrays
    brain_region_colors = np.loadtxt(color_file)
    repeated_colors = np.tile(brain_region_colors, len(indices))

    # Ensure the arrays and color data are of the correct length
    assert len(concatenated_array1) == len(concatenated_array2) == len(repeated_colors)

    # Define distinct color values for brain regions
    distinct_colors = [4.4, 3.5, 2.7, 1.6, 0.5]
    dict = {4.4: "Frontal lobe", 3.5: "Insula", 2.7: "Temporal and occipital lobes", 1.6: "Parietal lobe", 0.5: "Limbic lobe"}
    color_tolerance = 0.1  # Define a tolerance since we are working with floats

    # Filter and plot data for each brain region
    for region_value in distinct_colors:
        # Filter data points that belong to the current region within the tolerance
        region_mask = np.isclose(repeated_colors, region_value, atol=color_tolerance)
        region_x = concatenated_array1[region_mask]
        region_y = concatenated_array2[region_mask]

        # Calculate the regression for the filtered data
        regression_results = linregress(region_x, region_y)
        regional_slope, regional_intercept, regional_r_value, _, regional_std_err = regression_results

        # Define the degrees of freedom as the number of observations minus the number of parameters
        df = len(region_x) - 2

        # The t value for the desired confidence level
        t_val = t.ppf((1 + confidence) / 2, df)

        # The margin of error for the slope
        slope_error = t_val * regional_std_err

        # Calculate the standard error of the estimate (regression standard error)
        se_y = np.sqrt(np.sum((region_y - (regional_slope * region_x + regional_intercept))**2) / df)

        # Calculate the confidence intervals for the regression line
        ci = se_y * t_val * np.sqrt(1/len(region_x) + (region_x - np.mean(region_x))**2 / np.sum((region_x - np.mean(region_x))**2))
        ci_upper = regional_slope * region_x + regional_intercept + ci
        ci_lower = regional_slope * region_x + regional_intercept - ci

        # Plot each region in a separate graph
        plt.figure()
        #plt.xlabel('TPET', fontsize=16)
        #plt.ylabel('Neurodegeneration', fontsize=16)

        # Set color for points based on the brain region color
        norm = plt.Normalize(vmin=0, vmax=5)
        colors = plt.cm.jet(norm(repeated_colors))
        sc = plt.scatter(region_x, region_y, color=colors[region_mask], label=f'Brain Region: {dict[region_value]}')

        # Plot the regression line for the region
        plt.plot(region_x, regional_slope * region_x + regional_intercept, color='black')

        sorted_indices = np.argsort(region_x)
        sorted_region_x = region_x[sorted_indices]
        sorted_region_y = region_y[sorted_indices]

        # Calculate the regression values with the sorted x values
        sorted_regression_values = regional_slope * sorted_region_x + regional_intercept

        # Calculate the confidence intervals using the sorted x values
        sorted_ci_upper = sorted_regression_values + ci[sorted_indices]
        sorted_ci_lower = sorted_regression_values - ci[sorted_indices]

        # Plot the confidence interval as a shaded area using the sorted x values
        plt.fill_between(sorted_region_x, sorted_ci_lower, sorted_ci_upper, color='gray', alpha=0.2)

        # Create a legend with R^2 and a separate line for R^2
        # legend_text = f'{dict[region_value]}'
        legend_text = f'    '
        # plt.legend([legend_text, f'R^2 = {regional_r_value ** 2:.4f}'], loc='upper left', fontsize='16')
        plt.legend([legend_text], loc='upper left', fontsize='16', frameon=False)

        ax = plt.gca()  # Get the current axis
        ax.xaxis.set_major_locator(MaxNLocator(5))

        plt.tick_params(axis='both', which='major', labelsize=16)

        plt.show()

        # Print the R^2 value for the region
        print(f"Brain Region: {dict[region_value]} - R^2 value: {regional_r_value ** 2:.4f}")


def N_size_new(input_file, first_array, row_index):
    file_path = input_file
    data = np.load(file_path)
    modified_data = (first_array - data) * 10 * 5
    print(modified_data)
    filename = f"NPET_pred/new_param/NPET_effect_{row_index}.npy"
    np.save(filename, modified_data)
    return modified_data


if __name__ == "__main__":
    # part B-F
    # for APET color: #eb4034
    # for TPET color: #52ccaf
    # Load only the arrays with indices 0, 1 for A and T with respect to FDG_PET(Early stage)
    indices = [0,1]
    arrays1 = [np.load(f'NPET_pred/new_param/TPET_{i}.npy') for i in indices]
    arrays2 = [np.load(f'NPET_pred/new_param/NPET_{i}.npy') for i in indices]
    arrays3 = [np.load(f'NPET_pred/new_param/APET_{i}.npy') for i in indices]
    plot_with_color_partial_tau(arrays1, arrays2, indices)
    plot_with_color_partial_amy(arrays3, arrays2, indices)



    ####################################################
    #extract the points for different brain regions
    # now using plt
    indices = [1, 2, 3, 4]
    arrays1 = [np.load(f'NPET_pred/new_param/TPET_{i}.npy') for i in indices]
    arrays2 = [np.load(f'NPET_pred/new_param/NPET_effect_{i}.npy') for i in indices]
    color_file = 'color.txt'
    plot_separate_brain_regions_CI(arrays1, arrays2, color_file, indices)


    # part A
    example_color_array = np.load(f'color.npy')

    first_array = np.load(f"NPET_pred/new_param/NPET_0.npy")
    for row_index in range(5):
        file_path = f"NPET_pred/new_param/NPET_{row_index}.npy"
        N_size_array = N_size_new(file_path, first_array, row_index)
        build_node(example_color_array, N_size_array, "output/N_{}.node".format(row_index))


