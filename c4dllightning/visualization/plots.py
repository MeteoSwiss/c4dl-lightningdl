from datetime import datetime, timedelta
import os
import string

from matplotlib import colors, gridspec, patches, pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

from ..analysis import evaluation


def plot_calibration(p, occurrence_rate, names, colors_linestyles=None):
    fig = plt.figure()
    ax = fig.add_subplot()

    for model in occurrence_rate:
        if colors_linestyles is not None:
            (c, ls) = colors_linestyles[model]
        else:
            c = ls = None
        ax.plot(p, occurrence_rate[model], label=names[model],
            color=c, linestyle=ls)

    ax.plot([0,1], [0,1], color=(0.4,0.4,0.4), linestyle=":", label="_nolegend_")
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.03), ncol=4)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Occurrence rate")

    return fig


def plot_pr_curve(conf_matrix, names, colors_linestyles=None, show_auc=False):
    precision = {}
    recall = {}
    labels = {}
    for (k,cm) in conf_matrix.items():
        precision[k] = evaluation.precision(cm)
        recall[k] = evaluation.recall(cm)
        if show_auc:
            auc = evaluation.pr_area_under_curve(cm)
            labels[k] = f"{names[k]} (AUC: {auc:.3f})"
        else:
            labels[k] = names[k]

    fig = plt.figure()
    ax = fig.add_subplot()

    for model in precision:
        print(model, recall[model][-1], recall[model][0], precision[model][-1], precision[model][0])
        if colors_linestyles is not None:
            (c, ls) = colors_linestyles[model]
        else:
            c = ls = None
        ax.plot(recall[model], precision[model], label=labels[model],
            color=c, linestyle=ls)
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    return fig


def plot_roc_curve(conf_matrix, names, colors_linestyles=None, show_auc=False):
    tpr = {}
    fpr = {}
    labels = {}
    for (k,cm) in conf_matrix.items():
        ((tp, fn), (fp, tn)) = cm
        tpr[k] = tp / (tp + fn)
        fpr[k] = fp / (fp + tn)
        if show_auc:
            auc = evaluation.roc_area_under_curve(cm)
            labels[k] = f"{names[k]} (AUC: {auc:.3f})"
        else:
            labels[k] = names[k]

    fig = plt.figure(figsize=())
    ax = fig.add_subplot()

    for model in tpr:
        print(model, fpr[model][-1], fpr[model][0], tpr[model][-1], tpr[model][0])
        if colors_linestyles is not None:
            (c, ls) = colors_linestyles[model]
        else:
            c = ls = None
        ax.plot(fpr[model], tpr[model], label=labels[model],
            color=c, linestyle=ls)
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    return fig


metric_names = {
    "CSI": "Critical Success Index",
    "HSS": "Heidke Skill Score",
    "PSS": "Peirce Skill Score",
    "ETS": "Equitable Threat Score",
}
metric_funcs = {
    "CSI": evaluation.intersection_over_union,
    "HSS": evaluation.heidke_skill_score,
    "PSS": evaluation.peirce_skill_score,
    "ETS": evaluation.equitable_threat_score,
    "ROC AUC": evaluation.roc_area_under_curve,
    "PR AUC": evaluation.pr_area_under_curve,
    "POD": evaluation.recall,
    "FAR": evaluation.false_alarm_ratio
}

def plot_threshold_metric_curve(thresholds, conf_matrix, names, metric,
    colors_linestyles=None, fig=None, ax=None, legend=True,
    xlabel=True, show_best=False
):
    metric_scores = {}
    labels = {}
    for (k,cm) in conf_matrix.items():
        metric_scores[k] = metric_funcs[metric](cm)
        if show_best:      
            best = metric_scores[k].max()        
            labels[k] = f"{names[k]} (Best: {best:.3f})"
        else:
            labels[k] = names[k]

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    for (model, score) in metric_scores.items():
        if colors_linestyles is not None:
            (c, ls) = colors_linestyles[model]
        else:
            c = ls = None
        ax.plot(thresholds, score, label=labels[model],
            color=c, linestyle=ls)
    
    ylim = ax.get_ylim()
    ax.set_ylim((0,ylim[1]))
    ax.set_xlim((0,1))
    if legend:
        ax.legend()
    if xlabel:
        ax.set_xlabel("Threshold")
    ax.set_ylabel(metric_names[metric])

    return (fig, ax)


def plot_metric_leadtime(conf_matrix_lt, names, metrics=("CSI", "PSS"),
    colors_linestyles=None, fig=None, legend=True, dt_minutes=5
):
    if fig is None:
        fig = plt.figure(figsize=(6,6))
    
    leadtime = None
    
    for (i,metric) in enumerate(metrics):
        ax = fig.add_subplot(len(metrics), 1, i+1)            

        metric_scores = {}
        for (k,cm) in conf_matrix_lt.items():
            metric_scores[k] = metric_funcs[metric](cm).max(axis=0)
            if leadtime is None:
                leadtime = np.arange(1,len(metric_scores[k])+1) * dt_minutes

        for (model, score) in metric_scores.items():
            if colors_linestyles is not None:
                (c, ls) = colors_linestyles[model]
            else:
                c = ls = None
            ax.plot(leadtime, score, label=names[model],
                color=c, linestyle=ls)

        ax.text(
            0.01, 0.975,
            f"({string.ascii_lowercase[i]})",
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes
        )

        ax.set_xlim((0, leadtime[-1]))
        ylim = ax.get_ylim()
        ax.set_ylim((0, ylim[1]))
        ax.tick_params(right=True)

        if i == len(metrics)-1:
            ax.set_xlabel("Lead time [min]")
            ax.legend(loc='lower left')
        ax.set_ylabel(metric_names[metric])

    return fig


def plot_frame(ax, frame, norm=None):
    im = ax.imshow(frame.astype(np.float32), norm=norm)
    ax.tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)
    return im


def plot_model_examples(X, Y, models, shown_inputs=(0,25,12,9),
    input_timesteps=(-4,-1), output_timesteps=(0,2,5,11),
    batch_member=0, interval_mins=5,
    input_names=("Rain rate", "Lightning", "HRV", "CTH"),
    future_input_names=("CAPE-MU",),
    min_p=0.025, plot_scale=256
):
    num_timesteps = len(input_timesteps)+len(output_timesteps)
    gs_rows = 2 * max(len(models),len(shown_inputs))
    gs_cols = num_timesteps
    width_ratios = (
        [0.1, 0.19] +
        [1]*len(input_timesteps) +
        [0.1] +
        [1]*len(output_timesteps) +
        [0.19, 0.1]
    )
    gs = gridspec.GridSpec(gs_rows, gs_cols+5, wspace=0.02, hspace=0.05,
        width_ratios=width_ratios)
    batch = [x[batch_member:batch_member+1,...] for x in X]
    obs = [y[batch_member:batch_member+1,...] for y in Y]

    fig = plt.figure(figsize=(gs_cols*1.5, gs_rows/2*1.5))

    # plot inputs
    input_transforms = {
        "Rain rate": lambda x: 10**(x*0.528-0.051),
        "Lightning": lambda x: x,
        "POH": lambda x: x,
        "HRV": lambda x: x*100,
        "CTH": lambda x: x*2.810+5.260,
        "CAPE-MU": lambda x: x*200
    }
    input_norm = {
        "Rain rate": colors.LogNorm(0.01, 100, clip=True),
        "Lightning": colors.Normalize(0, 1),
        "POH": colors.Normalize(0, 1),
        "HRV": colors.Normalize(0,100),
        "CTH": colors.Normalize(0,12),
        "CAPE-MU": colors.Normalize(0,2000)
    }
    input_ticks = {
        "Rain rate": [0.1, 1, 10, 100],
        "Lightning": [0, 0.5, 1],
        "POH": [0, 0.5, 1],
        "HRV": [0, 25, 50, 75],
        "CTH": [0, 5, 10],
        "CAPE-MU": [500, 1000, 1500, 2000],
    }
    row0 = gs_rows//2 - len(shown_inputs)
    for (i,k) in enumerate(shown_inputs):
        row = row0 + 2*i        
        ip = batch[k][0,input_timesteps,:,:,0]
        ip = input_transforms[input_names[i]](ip)
        norm = input_norm[input_names[i]]
        for m in range(len(input_timesteps)):
            col = m+2
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, ip[m,:,:], norm=norm)
            if i == 0:
                iv = (input_timesteps[m]+1) * interval_mins
                ax.set_title(f"${iv}\\,\\mathrm{{min}}$")
            if m == 0:
                ax.set_ylabel(input_names[i])
                cax = fig.add_subplot(gs[row:row+2,0])                
                cb = plt.colorbar(im, cax=cax)
                cb.set_ticks(input_ticks[input_names[i]])
                cax.yaxis.set_ticks_position('left')

    # plot outputs
    row0 = 0
    future_input_ind = 0
    norm_log = colors.LogNorm(min_p,1,clip=True)
    for (i,model) in enumerate(models):
        if model == "obs":
            Y_pred = obs[0]
            norm = norm_log
            label = "Observed"
        elif isinstance(model, str) and model.startswith("input-future"):
            var_ind = int(model.split("-")[-1])
            Y_pred = batch[var_ind]
            input_name = future_input_names[future_input_ind]
            Y_pred = input_transforms[input_name](Y_pred)
            norm = input_norm[input_name]
            future_input_ind += 1
            label = input_name
        else:
            Y_pred = model.predict(batch)
            norm = norm_log
            label = "Forecast"
        row = row0 + 2*i
        op = Y_pred[0,output_timesteps,:,:,0]        
        for m in range(len(output_timesteps)):
            col = m + len(input_timesteps) + 3
            ax = fig.add_subplot(gs[row:row+2,col])
            im = plot_frame(ax, op[m,:,:], norm=norm)
            if i==0:
                iv = (output_timesteps[m]+1) * interval_mins
                ax.set_title(f"$+{iv}\\,\\mathrm{{min}}$")
            if m == len(output_timesteps)-1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(label)
                if i == len(models)-1:
                    scalebar = AnchoredSizeBar(ax.transData,
                           op.shape[1],
                           f'{plot_scale} km',
                           'lower center', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=1,
                           bbox_transform=ax.transAxes,
                           bbox_to_anchor=(0.5,-0.27)
                    )
                    ax.add_artist(scalebar)

        if i==len(models)-1:
            r0 = row0 + 2*len(future_input_names)
            r1 = r0 + 4
            cax = fig.add_subplot(gs[r0:r1,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cb.set_ticklabels([min_p, 0.05, 0.1, 0.2, 0.5, 1])
            cax.set_xlabel("$p$", fontsize=12)
        elif i<len(future_input_names):
            cax = fig.add_subplot(gs[row:row+2,-1])            
            cb = plt.colorbar(im, cax=cax)
            cb.set_ticks(input_ticks[input_name])

    return fig
