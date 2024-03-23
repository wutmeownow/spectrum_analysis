## -----Import modules----- ##
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import scipy.constants as const
import numpy as np
from tabulate import tabulate
import sys
import os
## from https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
# from Tkinter import Tk     # from tkinter import Tk for Python 3.x
# from tkinter.filedialog import askopenfilename




## -----Functions----- ##
# define wiens law formula
# returns temperature from peak wavelength
# L is wavelength in nm
def wiens_law(L):
    return const.Wien/(L*10**-9)

# define inverse wiens law formula
# returns peak wavelength from temperature
# T is temperature in kelvin
def inv_wiens_law(T):
    return (const.Wien/(T))/10**-9

# define black body curve formula
# T is black body temperature
def bb_curve(x, T, A):
    x_m = x*10**-9 # convert to m
    return A*(2 * const.h * const.c**2)/(x_m**5 * (np.exp(const.h*const.c/(x_m*const.k*T)) - 1))

# percent error function
def per_err(exp, actual):
    return ((exp - actual)/actual)*100




## -----Main----- ##
if __name__ == "__main__":
    # take in filename, actual temperature - compute actual peak wavelength
    filename = str(sys.argv[1])
    temp_act = float(sys.argv[2])
    lmax_act = inv_wiens_law(temp_act)

    ## Read in Data ##
    # Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    # filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    directory = os.path.join("..","Spectra","Final Spectra")
    file_path = os.path.join(directory,filename)
    out_dir = os.path.join(directory,filename.split('.', 1)[0] + "results")
    # print(out_dir)

    # # Specify the delimiter, in this case, it's tab
    # delimiter = '  '

    # Read the .dat file into a Pandas DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True, names=["lamA", "intensity"])
    # df = pd.read_csv(file_path, sep=delimiter, engine="python")
    # print(df.columns)
    print("Read", file_path)
    df["nanometers"] = df["lamA"]/10.
    x = df['nanometers']
    y = df['intensity']
    # y = df[f'f_g0iii']

    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(x,y, linestyle='-')
    plt.xlabel('Wavelength (nm)')
    x_max = int(np.max(x))
    x_min = int(np.min(x))
    custom_ticks = np.linspace((x_min//100 - 1)*100, (x_max//100 + 1)*100, 2*((x_max//100 + 1)-(x_min//100 - 1))+1)
    plt.xticks(custom_ticks)
    plt.grid()
    plt.show(block=False)

    # ask for lower and upper bounds for crop
    lower = float(input("Enter lower bound of main peak\n"))
    upper = float(input("Enter upper bound of main peak\n"))
    print(f"Bounds: {lower, upper}")
    plt.close()

    
    ## do BB fit
    # initial guess for T and A
    bb_init = (temp_act,np.max(y))

    # Use curve_fit to fit the data with black body curve
    params_bb, covariance_bb = curve_fit(bb_curve, x, y, p0=bb_init)
    T_sigma = np.sqrt(covariance_bb[0][0]) # T std dev
    
    # BB Params
    T_fit = params_bb[0]
    A_fit = params_bb[1]
    
    print("Black Body Fit:")
    print(f"  T = {T_fit:.2f}")
    print(f"  Std Dev = {T_sigma:.2f}")

    

    # Generate fitted curve using the fitted parameters
    wl = np.linspace(100,1200,1100*10, dtype=np.double)
    bb_fit = bb_curve(wl, T_fit, A_fit)

    # take weighted average of wavelengths to find peak
    # create a boolean mask
    b_mask = x > lower
    b_mask2 = x < upper
    x_cropped = x[b_mask]
    x_cropped = x_cropped[b_mask2]
    y_cropped = y[b_mask]
    y_cropped = y_cropped[b_mask2]
    
    cropped_weighted_max = np.sum(x_cropped * y_cropped)/np.sum(y_cropped)
    l_max_bb = wl[bb_fit.argmax()]

    # get temp for bb fit and weighted avg
    # temp_bb = wiens_law(l_max_bb)
    temp_wavg = wiens_law(cropped_weighted_max)
    
    # get percent errors
    perr_bb = per_err(T_fit,temp_act)
    perr_wavg = per_err(temp_wavg,temp_act)

    table = [
        ["Black Body", f"{l_max_bb:.2f}", f"{T_fit:.2f}({T_sigma:.2f})", f"{perr_bb:.2f}"],
        ["Cropped Weighted Avg", f"{cropped_weighted_max:.2f}", f"{temp_wavg:.2f}",f"{perr_wavg:.2f}"],
        ["Actual", f"{lmax_act:.2f}" ,f"{temp_act:.2f}",""]
    ]

    # Define the headers for the table
    headers = ["Fit", "Peak Wavelength", "Temperature", "Percent Error"]

    # Print the table
    print(tabulate(table, headers, tablefmt="fancy_grid"))

    # Plot the original data and the fitted curve
    plt.scatter(x, y, label='Spectrum',s=1)
    plt.plot(wl, bb_fit, label='Black Body Fit', color='green')
    plt.axvline(cropped_weighted_max, color='red', label="Cropped Max")
    plt.legend()
    plt.xlabel('wavelength (nm)')
    
    # save plot and results table
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir,"plot.png"))
    with open(os.path.join(out_dir,"results.txt"), 'w') as outputfile:
        outputfile.write(tabulate(table, headers))

    # show plot
    plt.show()
