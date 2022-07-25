from pandas_datareader import data as pdr
from matplotlib import pyplot as plt
import pickle


def compute_var(returns, confidence_level=.05):
    return returns.quantile(confidence_level, axis=0, interpolation='higher')


def compute_cvar(returns, confidence_level=.05):
    var = compute_var(returns, confidence_level)

    return returns[returns.lt(var, axis=1)].mean()

with open("var_vs_cvar_data.pkl", "rb") as file:
    pdf_data = pickle.load(file)

var = compute_var(pdf_data)
cvar = compute_cvar(pdf_data)

axes = pdf_data.plot.density()
fig = axes.get_figure()
figsize = (4.5, 3.5)
fig.set_size_inches(figsize)
pdf1_c = "tab:red"
pdf2_c = "tab:blue"
var_c = "tab:green"
lines = axes.get_lines()
lines[0].set_color(pdf1_c)
lines[1].set_color(pdf2_c)
axes.legend()
axes.axvline(
    x=cvar["pdf1"],
    ymin=0,
    ymax=1,
    c=pdf1_c,
)
axes.axvline(
    x=cvar["pdf2"],
    ymin=0,
    ymax=1,
    c=pdf2_c,
)
axes.axvline(
    x=var["pdf1"],
    ymin=0,
    ymax=1,
    c=var_c,
)
axes.annotate(
    text="VaR",
    xy=(var["pdf1"], 10),
    xytext=(0.03, 10),
    arrowprops={
        "facecolor": var_c,
        "shrink": 0.05,
    }
)
axes.annotate(
    text="CVaR pdf1",
    xy=(cvar["pdf1"], 30),
    xytext=(-0.08, 10),
    arrowprops={
        "facecolor": pdf1_c,
        "shrink": 0.05,
    }
)
axes.annotate(
    text="CVaR pdf2",
    xy=(cvar["pdf2"], 40),
    xytext=(-0.09, 30),
    arrowprops={
        "facecolor": pdf2_c,
        "shrink": 0.05,
    }
)
axes.set_xlim([-0.10,  0.06])
axes.set_ylim([ 0.00, 70.00])
axes.set_yticks([])
axes.set_xticks([])
axes.set_xlabel("Experimental Information")
fig.tight_layout()
fig.savefig("var_vs_cvar.png", dpi=360, quality=95)
plt.show()
