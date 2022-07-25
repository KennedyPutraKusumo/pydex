from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # criterion vs n_lp unstitched
    results = pd.DataFrame(
        data={
            "Number of Samples": [125, 250, 500, 1000, 2000, 4000],
            "Criterion Value": [-15.66823769, -15.49277973, -15.23026215, -15.37208581, -15.28538668, -15.30971747]
        }
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        results["Number of Samples"],
        results["Criterion Value"],
        marker="o",
        c="tab:blue",
    )
    ax.set_xlabel("Number of Samples $N_s$")
    ax.set_ylabel("D-optimal Criterion")
    fig.tight_layout()
    fig.savefig("criterion_vs_n_lp_unstitched", dpi=360)

    # criterion vs n_scr unstitched
    results = pd.DataFrame(
        data={
            "Number of Scenarios": [100, 200, 400, 1000, 2000, 4000],
            "Criterion Value": [-14.35758988, -15.083275641395712, -15.355716439581718, -15.37208581, -15.33288422, -15.34657555],
        }
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        results["Number of Scenarios"],
        results["Criterion Value"],
        marker="o",
        c="tab:blue",
    )
    ax.set_xlabel("Number of Scenarios $N_{mc}$")
    ax.set_ylabel("D-optimal Criterion")
    fig.tight_layout()
    fig.savefig("criterion_vs_n_scr_unstitched", dpi=360)

    # criterion vs n_scr stitched
    results = pd.DataFrame(
        data={
            "Number of Scenarios": [100, 200, 400, 800, 1000, 2000, 4000],
            "Criterion Value": [-14.96485188, -14.35758959, -14.35758959, -14.35758959, -14.35758959, -14.35758959, -14.35758959],
        }
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        results["Number of Scenarios"],
        results["Criterion Value"],
        marker="o",
        c="tab:blue",
    )
    ax.set_xlabel("Number of Scenarios $N_{mc}$")
    ax.set_ylabel("D-optimal Criterion")
    fig.tight_layout()
    fig.savefig("criterion_vs_n_scr_stitched", dpi=360)

    # criterion vs n_lp stitched
    results = pd.DataFrame(
        data={
            "Number of Samples": [125, 379, 883, 1883, 3885, 7973],
            "Criterion Value": [-15.66823769, -15.39009379, -15.17827072, -15.15813478, -15.1339971, -15.13328019]
        }
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        results["Number of Samples"],
        results["Criterion Value"],
        marker="o",
        c="tab:blue",
    )
    ax.set_xlabel("Number of Samples $N_s$")
    ax.set_ylabel("D-optimal Criterion")
    fig.tight_layout()
    fig.savefig("criterion_vs_n_lp_stitched", dpi=360)
    plt.show()
