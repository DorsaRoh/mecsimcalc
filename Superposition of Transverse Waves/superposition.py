# Code for the app

import numpy as np
import matplotlib.pyplot as plt
import mecsimcalc as msc


def main(inputs):
    t = np.linspace(0, 2*np.pi, 1000)  # Time array from 0 to 2π

    # Extract input values from the input dictionary
    amplitude1 = inputs['amplitude_1']
    frequency1 = inputs['frequency_1']
    amplitude2 = inputs['amplitude_2']
    frequency2 = inputs['frequency_2']

    # computing each wave's y values
    wave1 = amplitude1 * np.sin(2*np.pi*frequency1*t) # wave 1's y values
    wave2 = amplitude2 * np.sin(2*np.pi*frequency2*t) # wave 2's y values
    superposition = wave1 + wave2 # y values of resulting function (superposition)

    # Plot the x (t) and y (superposition) values with a label "superposition"
    plt.plot(t, superposition, label="Superposition")

    # Construct the title, labels and legend
    equation = f"y = {amplitude1} * sin({frequency1} * x) + y = {amplitude2} * sin({frequency2} * x)"
    plt.title(equation)
    plt.xlabel("Time")
    plt.ylabel("Superposition of Waves")
    plt.legend()

    # Get plot image and download link using mecsimcalc's print_plot function
    img, download = msc.print_plot(plt, download=True)

    return {
        "plot": img,
        "equation": equation,
        "download": download,
    }
