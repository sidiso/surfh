import os

os.environ["pandeia_refdata"] = "./pandeia_refdata"

import numpy as np
import matplotlib.pyplot as plt
from pandeia.engine.instrument_factory import InstrumentFactory
from pandeia.engine.calc_utils import build_default_calc
from matplotlib import cm
from datetime import date


def pce(show=True, write=True):
    """MRS Photon-to-electron conversion efficiency (PCE) estimator.
    Utility that fetches the PCE numerical values from the pandeia
    package, plots a graph of the PCE and saves it into a single
    text file.

    Parameters:
        show (bool, optional): Whether to plot the PCE. Defaults to True.
        write (bool, optional): Whether to save the PCE into a text file. Defaults to True.
    """
    # update matplotlib font
    plt.rcParams.update({"font.size": 6})

    # Photon conversion efficiency plotter
    channels = ["ch1", "ch2", "ch3", "ch4"]
    bands = ["short", "medium", "long"]

    config = {
        "instrument": {
            "aperture": "ch1",
            "disperser": "short",
            "filter": None,
            "instrument": "miri",
            "mode": "mrs",
        }
    }

    # Initialize wavelengths axis
    wave = np.arange(4.5, 30, 0.001)

    # Initialize color palette
    colors = iter(cm.rainbow(np.linspace(0, 1, len(channels) * len(bands))))

    # Initialize the list of efficiencies (Four channels, three bands)
    efficiencies = []

    _, ax = plt.subplots(1, 1, figsize=(10, 2), dpi=500)

    # Loop over the channels and bands
    for c in channels:
        for b in bands:
            color = next(colors)
            config["instrument"]["aperture"] = c
            config["instrument"]["disperser"] = b
            instrument_factory = InstrumentFactory(config=config)
            efficiency = instrument_factory.get_total_eff(wave)
            efficiencies.append(efficiency)

            ax.plot(
                wave, efficiency, color=color, linestyle="-", linewidth=0.2, alpha=0.7
            )
            ax.plot(
                [],
                [],
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.7,
                label=f"{c}{b}",
            )
            ax.fill_between(wave, efficiency, color=color, alpha=0.7)

    ax.set_ylim(0, 0.22)
    ax.set_xlim(4.5, 30)
    ax.legend(ncol=4, fontsize=5)
    ax.set_ylabel("Photon-to-electron conversion efficiency")
    ax.set_xlabel("Wavelength (microns)")

    if write:
        # Initialize jwst_mrs_pce.dat output file
        output = open("./src/pce_estimator/jwst_mrs_pce.dat", "w")
        separator = "!____________________________________________________________________________________________________\n"
        output.write(separator)
        output.write("!\n")
        output.write("! PCE (Photon-to-electron conversion efficiency), JWST MRS\n")
        output.write(separator)
        output.write("!\n")
        output.write("! Date : " + str(date.today()) + "\n")
        output.write(
            "! Pierre Dell'Ova, IAS IDOC (pierre.dellova@universite-paris-saclay.fr)\n"
        )
        output.write(separator)
        output.write("!\n")
        output.write("! TABULATED QUANTITIES \n")
        output.write("!\n")
        output.write(
            "!   0 .... WAVE ............ Wavelength .......................................... (micron)\n"
        )
        output.write(
            "!   1 .... PCE-ch1short .... Photon conversion efficiency of channel ch1short ....\n"
        )
        output.write(
            "!   2 .... PCE-ch1medium ... Photon conversion efficiency of channel ch1medium ...\n"
        )
        output.write(
            "!   3 .... PCE-ch1long ..... Photon conversion efficiency of channel ch1long .....\n"
        )
        output.write(
            "!   4 .... PCE-ch2short .... Photon conversion efficiency of channel ch2short ....\n"
        )
        output.write(
            "!   5 .... PCE-ch2medium ... Photon conversion efficiency of channel ch2medium ...\n"
        )
        output.write(
            "!   6 .... PCE-ch2long ..... Photon conversion efficiency of channel ch2long .....\n"
        )
        output.write(
            "!   7 .... PCE-ch3short .... Photon conversion efficiency of channel ch3short ....\n"
        )
        output.write(
            "!   8 .... PCE-ch3medium ... Photon conversion efficiency of channel ch3medium ...\n"
        )
        output.write(
            "!   9 .... PCE-ch3long ..... Photon conversion efficiency of channel ch3long .....\n"
        )
        output.write(
            "!   10 ... PCE-ch4short .... Photon conversion efficiency of channel ch4short ....\n"
        )
        output.write(
            "!   11 ... PCE-ch4medium ... Photon conversion efficiency of channel ch4medium ...\n"
        )
        output.write(
            "!   12 ... PCE-ch4long ..... Photon conversion efficiency of channel ch5long .....\n"
        )
        output.write(separator + "\n")

        # Write the efficiencies into jwst_mrs_pce.dat (one column per band)
        for k in range(len(wave)):
            line = "{:.3f}".format(wave[k]) + " "
            for i in range(len(efficiencies)):
                line += "{:.5f}".format(efficiencies[i][k]) + " "
            line += "\n"
            output.write(line)

    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    pce()
