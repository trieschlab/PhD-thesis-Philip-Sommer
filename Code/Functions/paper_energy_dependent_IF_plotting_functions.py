# paper plotting functions

# import predefined functions
import Functions.plotting_functions as pf
import Functions.analysis_functions as af

import os
import io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from PIL import Image
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
    
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns



def set_paper_style():
    # set global matplotlib style for paper-ready, square figures

    mpl.rcParams.update({
        "figure.figsize": (5.0, 5.0),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.transparent": True,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"], # "CMU Serif"
    })
    # costum letter sizes
    panelletterfontsize=15
    
    return panelletterfontsize
    
def OLD_set_paper_style():
    # set global matplotlib style for paper-ready, square figures

    mpl.rcParams.update({
        "figure.figsize": (5.0, 5.0),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.transparent": True,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"], # "CMU Serif"
    })
    # costum letter sizes
    panelletterfontsize=15
    
    return panelletterfontsize


def set_thesis_style():
    # set global matplotlib style for thesis figures (slightly larger text)

    mpl.rcParams.update({
        "figure.figsize": (5.0, 5.0),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.transparent": True,
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"], # "CMU Serif"
    })
    # costum letter sizes
    panelletterfontsize=14
    
    return panelletterfontsize

############################ helper functions ############################

def panel_letter(ax, letter, size=15, dx=-0.02, dy=0.02):
    fig = ax.figure
    bbox = ax.get_position()  # in figure coordinates
    fig.text(bbox.x0 + dx, bbox.y1 + dy, letter, fontsize=size, fontweight='bold', ha='left', va='bottom')

def render_3Dfig_to_ax(fig, ax, scale=2, trim_pad=1):
    # render a plotly figure to an axis object
    # input
    # ax is the given axis
    # fig is the given fig
    # scale is the scale to import the figure
    # trim_pad is the frame distance to cut the figure
    
    
    # tight layout & transparent background
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict( xaxis=dict(showbackground=False, visible=True), yaxis=dict(showbackground=False, visible=True), zaxis=dict(showbackground=False, visible=True)))

    # render png in memory
    png_bytes = pio.to_image(fig, format="png", scale=scale)
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    # crop transparent borders
    alpha = im.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        L, T, R, B = bbox
        L = max(0, L - trim_pad)
        T = max(0, T - trim_pad)
        R = min(im.width,  R + trim_pad)
        B = min(im.height, B + trim_pad)
        im = im.crop((L, T, R, B))
        
    # show figure
    ax.imshow(np.asarray(im), interpolation='none')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.axis('off')

############################ plotting functions ############################

# Fig. energy-dependent IF
def fig_energy_IF_CTR_FR(time, rho_E_smooth, r_Ca_smooth, w_syn_e_dyn, V_rest_dyn, R_m_dyn, rho_E_target, r_target, r_Ca_var_smooth=None, r_var_target=None, var_CV_mode="var", y_mode_V_rest=None, exp_w=None, exp_V_rest=None, exp_R_m=None, downsample_factor=1, colors=['black', 'red'], fontsizes={'panelletterfontsize': 15}, figsize=(7,4), savename="4_results/fig_energy_IF_CTR_FR"):
    # create energy-dependent integrate-and-fire model summary figure for CTR and FR
    # input
    # time is a list of simulation-time arrays in s for CTR and FR
    # rho_E_smooth is a list of smoothed energy-balance arrays for CTR and FR
    # r_Ca_smooth is a list of calcium-based firing-rate estimate arrays in Hz for CTR and FR
    # w_syn_e_dyn is a list of mean excitatory synaptic-weight arrays in nS for CTR and FR
    # V_rest_dyn is a list of dynamic resting-potential arrays in mV for CTR and FR
    # R_m_dyn is a list of dynamic membrane-resistance arrays in MOhm for CTR and FR
    # rho_E_target is the target normalized energy-balance value
    # r_target is the target firing rate in Hz
    # r_Ca_var_smooth is an optional list of calcium-rate variance arrays in Hz^2 for CTR and FR
    # r_var_target is an optional target calcium-rate variance in Hz^2
    # var_CV_mode is "var" for variance plotting or otherwise for CV^2 plotting
    # y_mode_V_rest is "blocked" for fixed y-scale of V_rest
    # exp_w is an optional list of experimentally measured relative synaptic weights
    # exp_V_rest is an optional list of experimentally measured resting potentials in mV
    # exp_R_m is an optional list of experimentally measured membrane resistances in MOhm
    # downsample_factor keeps every nth sample for faster plotting
    # colors are the colors for CTR and FR
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figure size
    # savename sets the save name of the file
    # output
    # fig is the created matplotlib figure
    # ax is the dictionary of created matplotlib axes
    
    labels = ["CTR", "FR"]
    color_CTR, color_FR = colors[0], colors[1]
    colors = [color_CTR, color_FR]
    plot_variance_panel = r_Ca_var_smooth is not None

    def as_pair(x, name):
        # check whether input is a pair of CTR and FR arrays
        # input
        # x is a list or tuple containing two arrays
        # name is the name of the checked variable
        # output
        # x_pair is a list containing two numpy arrays
        
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"{name} must be a list or tuple with two entries: [CTR, FR].")
        x_pair = [np.asarray(x[0]), np.asarray(x[1])]
        
        return x_pair

    def downsample_func(x):
        # downsample an array by keeping every nth sample
        # input
        # x is an array
        # output
        # x_downsampled is the downsampled array
        
        if downsample_factor is None or downsample_factor <= 1:
            x_downsampled = x
        else:
            x_downsampled = x[::downsample_factor]
        
        return x_downsampled
    
    def add_exp_right_axis_ticks(axis, exp_values, colors, value_format="{:.1f}"):
        # add experimental values as right-side y-axis ticks
        # input
        # axis is the matplotlib axis
        # exp_values is a list of experimental values for CTR and FR
        # colors are the colors for color_CTR and color_FR
        # value_format determines the number formatting
        # output
        # ax_exp is the added right-side axis
        
        if exp_values is None:
            return None
    
        if not isinstance(exp_values, (list, tuple, np.ndarray)) or len(exp_values) != 2:
            raise ValueError("exp_values must be None or a list/tuple with two entries: [CTR, FR].")
    
        ax_exp = axis.twinx()
        ax_exp.set_ylim(axis.get_ylim())
    
        ax_exp.set_yticks(exp_values)
        ax_exp.set_yticklabels([value_format.format(v) for v in exp_values])
    
        ax_exp.tick_params(axis="y", which="major", right=True, labelright=True, left=False, labelleft=False, length=3.5, width=0.8, direction="out", pad=3)
    
        ax_exp.spines["right"].set_visible(True)
        ax_exp.spines["right"].set_linewidth(1.0)
        ax_exp.spines["left"].set_visible(False)
        ax_exp.spines["top"].set_visible(False)
        ax_exp.spines["bottom"].set_visible(False)
        ax_exp.patch.set_visible(False)
    
        for j, tick in enumerate(ax_exp.yaxis.get_major_ticks()):
            tick.tick2line.set_color(colors[j])
            tick.label2.set_color(colors[j])
            tick.label2.set_fontsize(axis.yaxis.get_ticklabels()[0].get_fontsize())
    
        return ax_exp

    time = as_pair(time, "time")
    rho_E_smooth = as_pair(rho_E_smooth, "rho_E_smooth")
    r_Ca_smooth = as_pair(r_Ca_smooth, "r_Ca_smooth")
    w_syn_e_dyn = as_pair(w_syn_e_dyn, "w_syn_e_dyn")
    V_rest_dyn = as_pair(V_rest_dyn, "V_rest_dyn")
    R_m_dyn = as_pair(R_m_dyn, "R_m_dyn")

    if plot_variance_panel is True:
        r_Ca_var_smooth = as_pair(r_Ca_var_smooth, "r_Ca_var_smooth")

    for i in range(2):
        lengths = [len(time[i]), len(rho_E_smooth[i]), len(r_Ca_smooth[i]), len(w_syn_e_dyn[i]), len(V_rest_dyn[i]), len(R_m_dyn[i])]

        if plot_variance_panel is True:
            lengths.append(len(r_Ca_var_smooth[i]))

        n = min(lengths)

        time[i] = downsample_func(time[i][:n])
        rho_E_smooth[i] = downsample_func(rho_E_smooth[i][:n])
        r_Ca_smooth[i] = downsample_func(r_Ca_smooth[i][:n])
        w_syn_e_dyn[i] = downsample_func(w_syn_e_dyn[i][:n])
        V_rest_dyn[i] = downsample_func(V_rest_dyn[i][:n])
        R_m_dyn[i] = downsample_func(R_m_dyn[i][:n])

        if plot_variance_panel is True:
            r_Ca_var_smooth[i] = downsample_func(r_Ca_var_smooth[i][:n])

    if plot_variance_panel is True:
        mosaic = [["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"], ["E", "E"], ["F", "F"]]
        figsize = (figsize[0], figsize[1]/5*6)
        panel_energy = "A"
        panel_rate = "B"
        panel_var = "C"
        panel_w = "D"
        panel_v = "E"
        panel_r = "F"
    else:
        mosaic = [["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"], ["E", "E"]]
        figsize = figsize
        panel_energy = "A"
        panel_rate = "B"
        panel_w = "C"
        panel_v = "D"
        panel_r = "E"
        
        
    # normalize time to relative time from 0 to 1
    time_all = np.concatenate((time[0], time[1]))
    time_min = np.nanmin(time_all)
    time_max = np.nanmax(time_all)

    if time_max > time_min:
        time = [(time[i] - time_min) / (time_max - time_min) for i in range(2)]
    else:
        time = [np.zeros_like(time[i], dtype=float) for i in range(2)]
        
    fig, ax = plt.subplot_mosaic(mosaic, figsize=figsize, sharex=True, gridspec_kw={"hspace": 0.6})

    # A
    # energy balance
    rho_E_all = np.concatenate([rho_E_smooth[0], rho_E_smooth[1]])
    rho_E_min = np.nanmin(rho_E_all)
    rho_E_max = np.nanmax(rho_E_all)
    rho_E_ylim_min = rho_E_min - 0.1
    rho_E_ylim_max = rho_E_max + 0.1
    x_text = min(np.nanmin(time[0]), np.nanmin(time[1]))

    for i in range(2):
        ax[panel_energy].plot(time[i], rho_E_smooth[i], label=labels[i], color=colors[i])
        
    ax[panel_energy].axhline(rho_E_target, color="gray", linestyle="--", linewidth=1, label="target")
    ax[panel_energy].set_ylim(rho_E_ylim_min, rho_E_ylim_max)
    ax[panel_energy].text(x_text, rho_E_max + 0.05, r"$\rho_E > 1$: energy excess regime", ha="left", va="center")
    ax[panel_energy].text(x_text, rho_E_min - 0.05, r"$\rho_E < 1$: energy scarcity regime", ha="left", va="center")
    ax[panel_energy].set_ylabel(r"$\rho_E$")
    ax[panel_energy].set_title(r"Energy balance $\rho_E(t)$")
    ax[panel_energy].legend(loc="center right", facecolor='white', edgecolor='none', framealpha=1.0)
    panel_letter(ax[panel_energy], "A", dx=-0.02, dy=0.0, size=fontsizes['panelletterfontsize'])

    # B
    # calcium-based firing-rate estimate
    for i in range(2):
        ax[panel_rate].plot(time[i], r_Ca_smooth[i], label=labels[i], color=colors[i]) 

    ax[panel_rate].axhline(r_target, color="gray", linestyle="--", linewidth=1)
    ax[panel_rate].set_ylabel(r"$r$ / Hz")
    ax[panel_rate].set_title(r"Firing-rate estimate $r(t)$")
    panel_letter(ax[panel_rate], "B", dx=-0.02, dy=0.0, size=fontsizes['panelletterfontsize'])

    # C
    # optional calcium-rate variance panel
    if plot_variance_panel is True:
        if var_CV_mode == "var":
            label_y = r"$\sigma^2_{r}$ / Hz$^2$"
            title_var = r"Firing-rate variance estimate $\sigma^2_{r}(t)$"
        else:
            label_y = r"$\mathrm{CV}^2_{r}$"
            title_var = r"Firing-rate CV$^2$ estimate"

        for i in range(2):
            ax[panel_var].plot(time[i], r_Ca_var_smooth[i], label=labels[i], color=colors[i])

        if r_var_target is not None:
            ax[panel_var].axhline(r_var_target, color="gray", linestyle="--", linewidth=1)

        ax[panel_var].set_ylabel(label_y)
        ax[panel_var].set_title(title_var)
        panel_letter(ax[panel_var], "C", dx=-0.02, dy=0.0, size=fontsizes['panelletterfontsize'])

    # C or D
    # synaptic scaling factor
    exp_w_ticks = None
    if exp_w is not None:
        w_ref = np.asarray(w_syn_e_dyn[0])[np.isfinite(w_syn_e_dyn[0])][-1]
        exp_w_ticks = [exp_w[0] * w_ref, exp_w[1] * w_ref]
        
    for i in range(2):
        ax[panel_w].plot(time[i], w_syn_e_dyn[i], label=labels[i], color=colors[i])

    add_exp_right_axis_ticks(axis=ax[panel_w], exp_values=exp_w_ticks, colors=colors, value_format="{:.2f}")

    ax[panel_w].set_ylabel(r"$\langle w_{\mathrm{syn},e} \rangle$ / nS")
    ax[panel_w].set_title(r"Mean excitatory synaptic weight $\langle w_{\mathrm{syn},e} \rangle(t)$")
    
    if plot_variance_panel is True:
        panel_letter(ax[panel_w], "D", dx=-0.02, dy=0.0, size=fontsizes['panelletterfontsize'])
    else:
        panel_letter(ax[panel_w], "C", dx=-0.02, dy=0.0, size=fontsizes['panelletterfontsize'])

    # D or E
    # dynamic resting potential
    for i in range(2):
        ax[panel_v].plot(time[i], V_rest_dyn[i], label=labels[i], color=colors[i])
    
    if y_mode_V_rest == "blocked":
        ax[panel_v].set_ylim(-73, -65)

    add_exp_right_axis_ticks(axis=ax[panel_v], exp_values=exp_V_rest, colors=colors, value_format="{:.0f}")

    if exp_V_rest is not None or exp_R_m is not None or exp_w is not None:
        ax[panel_w].text(1.011, 1.03, "exp.", transform=ax[panel_w].transAxes, ha="left", va="bottom", clip_on=False)
    
    ax[panel_v].set_ylabel(r"$V_{\mathrm{rest}}$ / mV")
    ax[panel_v].set_title(r"Resting potential $V_{\mathrm{rest}}(t)$")
    
    if plot_variance_panel is True:
        panel_letter(ax[panel_v], "E", dx=-0.02, dy=0.0, size=fontsizes['panelletterfontsize'])
    else:
        panel_letter(ax[panel_v], "D", dx=-0.02, dy=0.0, size=fontsizes['panelletterfontsize'])
    
    # E or F
    # dynamic membrane resistance
    for i in range(2):
        ax[panel_r].plot(time[i], R_m_dyn[i], label=labels[i], color=colors[i])

    add_exp_right_axis_ticks(axis=ax[panel_r], exp_values=exp_R_m, colors=colors, value_format="{:.0f}")
    
    ax[panel_r].set_xlabel("relative time")
    ax[panel_r].set_ylabel(r"$R_m$ / M$\Omega$")
    ax[panel_r].set_title(r"Membrane resistance $R_m(t)$")
    
    if plot_variance_panel is True:
        panel_letter(ax[panel_r], "F", dx=-0.02, dy=0.0, size=fontsizes['panelletterfontsize'])
    else:
        panel_letter(ax[panel_r], "E", dx=-0.02, dy=0.0, size=fontsizes['panelletterfontsize'])

    # format axes
    for key in ax:
        ax[key].spines['top'].set_visible(False)
        ax[key].spines['right'].set_visible(False)
        ax[key].set_xticks([0, 0.5, 1])
        ax[key].set_xticklabels(["0", "0.5", "1"])
        #ax[key].xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax[key].yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))

    if savename is not False:
        mpl.rcParams['pdf.compression'] = 0
        savepath = f"../Figures/paper_energy_dependent_IF/{savename}.pdf"
        fig.savefig(savepath, bbox_inches='tight', transparent=False)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=600, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.tiff'), dpi=600, bbox_inches='tight', transparent=False, pil_kwargs={"compression": "tiff_lzw"})

    plt.show()

    return fig, ax

# Fig. synapse- vs. excitability-first model

# Fig. synapse- vs. excitability-first model
def fig_synapse_vs_excitability_first(time_FR_PD=None, time_FR_PD_reverse=None, E_cons_dyn_FR_PD=None, E_cons_dyn_FR_PD_reverse=None, results_mean_FR_trajectory_synapse_first=None, results_mean_FR_trajectory_excitability_first=None, results_FR_synapse_first=None, results_FR_excitability_first=None, E_target_ATP=1.6, downsample_factor=100, colors=("red", "darkred"), labels=("FR synapse-first", "FR excitability-first"), markers=("o", "D"), fontsizes={"panelletterfontsize": 15}, figsize=(7,1.5), savename_mode=True):
    # creates a three-panel figure comparing the synapse-first and excitability-first models
    # input
    # time_FR_PD is the synapse-first simulation time array in s
    # time_FR_PD_reverse is the excitability-first simulation time array in s
    # E_cons_dyn_FR_PD is the normalized synapse-first consumed energy array
    # E_cons_dyn_FR_PD_reverse is the normalized excitability-first consumed energy array
    # results_mean_FR_trajectory_synapse_first is the averaged results dictionary for the synapse-first trajectory
    # results_mean_FR_trajectory_excitability_first is the averaged results dictionary for the excitability-first trajectory
    # results_FR_trajectory_synapse_first is the full results dictionary for the synapse-first trajectory
    # results_FR_trajectory_excitability_first is the full results dictionary for the excitability-first trajectory
    # E_target_ATP is the energy normalization factor in units of 1e9 ATP/s
    # downsample_factor keeps every nth sample for faster plotting
    # colors contains the colors of the synapse-first and excitability-first trajectories
    # labels contains the labels of the synapse-first and excitability-first trajectories
    # markers contains the marker shapes of the synapse-first and excitability-first trajectories
    # fontsizes is a dictionary containing the used font sizes
    # figsize determines the figure size
    # savename_mode determines whether the figure is saved under ../Figures/PhD_thesis/
    # output
    # fig is the created matplotlib figure
    # ax is a dictionary of created matplotlib axes

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.25, 1.0], wspace=0.5)

    R_m = 92700000
    V_Na = 50e-3
    V_K = -100e-3
    V_h = -43e-3
    alpha = 0.05

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # A
    pf.plot_E_cons_synapse_vs_excitability_first(time_FR_PD, time_FR_PD_reverse, E_cons_dyn_FR_PD, E_cons_dyn_FR_PD_reverse, E_target_ATP=E_target_ATP, downsample_factor=downsample_factor, colors=colors, labels=labels, ax=axA)
    panel_letter(axA, "A", size=fontsizes["panelletterfontsize"])

    # B
    #pf.plot_energy_vs_OSI_two_FR_trajectories(results_mean_FR_trajectory_synapse_first, results_mean_FR_trajectory_excitability_first, colors=colors, labels=labels, markers=markers, show_legend=False, ax=axB)
    pf.plot_OSI_per_energy_two_FR_trajectories(results_mean_FR_trajectory_synapse_first, results_mean_FR_trajectory_excitability_first, results_FR_synapse_first, results_FR_excitability_first, OSI_per_energy="OSI_per_energy", colors=colors, labels=labels, markers=("o", "D"), show_legend=False, ax=axB) # , s=42, linewidth=1.4
    panel_letter(axB, "B", size=fontsizes["panelletterfontsize"])
    
    # C
    pf.plot_new_E_RP_vs_V_RP(R_m, V_K, V_Na, V_h, alpha, ax=axC)
    panel_letter(axC, "C", size=fontsizes["panelletterfontsize"])

    for axis in [axA, axB, axC]:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        #axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))

    axB.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axC.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    
    axA.set_xlim(-0.05, 1.05)
    axA.set_xticks([0, 0.5, 1], labels=["0", "0.5", "1"])

    if savename_mode is True:
        mpl.rcParams["pdf.compression"] = 0
        savepath = "../Figures/paper_energy_dependent_IF/3_results/fig_synapse_vs_excitability_first.pdf"
        fig.savefig(savepath, bbox_inches="tight", transparent=False)
        fig.savefig(savepath.replace(".pdf", ".png"), dpi=600, bbox_inches="tight", transparent=True)
        fig.savefig(savepath.replace(".pdf", ".tiff"), dpi=600, bbox_inches="tight", transparent=False, pil_kwargs={"compression": "tiff_lzw"})

    plt.show()

    ax = {"A": axA, "B": axB, "C": axC}

    return fig, ax


# Fig. natural scenes
def fig_natural_scenes(firing_statistics, colors=['black', 'red'], fontsizes={'panelletterfontsize': 15}, figsize=(7,2.3), savename_mode=True):
    # create natural scenes firing statistics figure
    # input
    # firing_statistics is a dictionary containing firing rate, ISI variance and CV_ISI values for CTR and FR
    # colors are the colors for color_CTR and color_FR
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    # output
    # fig is the created matplotlib figure
    # stats_df is a pandas dataframe containing the statistical results for all plotted quantities
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.45)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # get data
    if "firing_rate_movie_CTR" in firing_statistics:
        firing_rate_movie_CTR = firing_statistics["firing_rate_movie_CTR"]
        firing_rate_movie_FR = firing_statistics["firing_rate_movie_FR"]
    else:
        firing_rate_movie_CTR = firing_statistics["mean_firing_rate_movie_CTR"]
        firing_rate_movie_FR = firing_statistics["mean_firing_rate_movie_FR"]

    if "ISI_var_movie_CTR" in firing_statistics:
        ISI_var_movie_CTR = firing_statistics["ISI_var_movie_CTR"]
        ISI_var_movie_FR = firing_statistics["ISI_var_movie_FR"]
    else:
        ISI_var_movie_CTR = firing_statistics["mean_ISI_var_movie_CTR"]
        ISI_var_movie_FR = firing_statistics["mean_ISI_var_movie_FR"]

    if "CV_ISI_movie_CTR" in firing_statistics:
        CV_ISI_movie_CTR = firing_statistics["CV_ISI_movie_CTR"]
        CV_ISI_movie_FR = firing_statistics["CV_ISI_movie_FR"]
    else:
        CV_ISI_movie_CTR = firing_statistics["mean_CV_ISI_movie_CTR"]
        CV_ISI_movie_FR = firing_statistics["mean_CV_ISI_movie_FR"]

    # A
    stat_A = pf.plot_boxplot(firing_rate_movie_CTR, firing_rate_movie_FR, 'firing rate (Hz)', 'Firing rate', colors=colors, ax=axA)
    stat_A["metric"] = "firing_rate_movie"
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])

    # B
    stat_B = pf.plot_boxplot(ISI_var_movie_CTR, ISI_var_movie_FR, 'ISI variance (s$^2$)', 'ISI variance', colors=colors, ax=axB)
    stat_B["metric"] = "ISI_var_movie"
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])

    # C
    stat_C = pf.plot_boxplot(CV_ISI_movie_CTR, CV_ISI_movie_FR, '$CV_{\mathrm{ISI}}$', 'ISI irregularity', colors=colors, ax=axC)
    stat_C["metric"] = "CV_ISI_movie"
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])


    if savename_mode is True:
        mpl.rcParams['pdf.compression'] = 0
        savepath="../Figures/paper_energy_dependent_IF/3_results/fig_natural_scenes_variance.pdf"
        fig.savefig(savepath, bbox_inches='tight', transparent=False)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=600, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.tiff'), dpi=600, bbox_inches='tight', transparent=False, pil_kwargs={"compression": "tiff_lzw"})

    plt.show()

    return fig

# Fig. 3D fitted plane
def fig_3D_fitted_plane(fig, results_mean_spiking_trials, value_key='OSI_per_energy', lower_threshold=0.2, upper_threshold=None, use_translated_w_scale=True, plane_resolution=20, plane_opacity=0.9, print_results=True, name='OSI plane fit', figsize=(7,10), fontsizes={'panelletterfontsize': 15}, savename_mode=True):
    # create 3D figure with fitted plane added to existing Plotly figure
    # input
    # fig is an existing Plotly figure containing the 3D parameter-space plot
    # results_mean_spiking_trials is a dictionary containing simulation results for different R_m, E_L and w_scale combinations
    # value_key is the key of the result value used for thresholding before plane fitting
    # lower_threshold is the lower threshold for selecting points used for plane fitting
    # upper_threshold is the upper threshold for selecting points used for plane fitting
    # use_translated_w_scale decides whether w_scale is translated into mean excitatory synaptic weights
    # plane_resolution is the number of grid points used for plotting the fitted plane
    # plane_opacity is the opacity of the fitted plane
    # print_results decides whether the fitted plane parameters are printed
    # name is the name of the fitted plane trace
    # figsize determines the figure size for saving in inches
    # fontsizes is a dictionary of used font sizes
    # savename_mode decides whether the figure is saved or not
    # output
    # fig is the Plotly figure with the fitted plane added
    # plane_fit is a dictionary containing the plane parameters, fit quality and fitted data points
    
    fig, plane_fit = pf.add_plane_fit_to_fig(fig, results_mean_spiking_trials, value_key=value_key, lower_threshold=lower_threshold, upper_threshold=upper_threshold, use_translated_w_scale=use_translated_w_scale, plane_resolution=plane_resolution, plane_opacity=plane_opacity, print_results=print_results, name=name)
    fig.show()
    
    if savename_mode is True:
        savepath="../Figures/paper_energy_dependent_IF/5_supplements/fig_3D_fitted_plane.html"
        fig.write_html(savepath)
        fig.write_image(savepath.replace('.html', '.pdf'), scale=3)
        fig.write_image(savepath.replace('.html', '.png'), scale=3)
        #fig.write_image(savepath.replace('.html', '.svg'), width=int(figsize[0] * 100), height=int(figsize[1] * 100), scale=3)
    
    return fig, plane_fit

