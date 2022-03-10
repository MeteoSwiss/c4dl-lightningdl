from itertools import product
import os
import string

from matplotlib import pyplot as plt
import numpy as np

from c4dllightning.visualization import plots


loss_names = {
    "BFL2": "FL $\gamma=2$",
    "BFL1": "FL $\gamma=1$",
    "WFL2": "WFL $\gamma=2$",
    "WFL1": "WFL $\gamma=1$",
    "WCE": "WCE",
    "BCE": "CE",
    "IOU": "CSI",
}
color_fl2 = "#0072B2"
color_fl1 = "#56B4E9"
color_ce = "#009E73"
color_iou = "#E69F00"
color_per = "#009E73"
color_lag = "#E69F00"
loss_colors_linestyles = {
    "BFL2": (color_fl2, "-"),
    "BFL1": (color_fl1, "-"),
    "WFL2": (color_fl2, "--"),
    "WFL1": (color_fl1, "--"),
    "WCE": (color_ce, "--"),
    "BCE": (color_ce, "-"),
    "IOU": (color_iou, "-")
}
loss_calibration_files = {
    "BFL2": "calibration-lightning_dropout_weightdecay_noclassweight.npy",
    "BFL1": "calibration-lightning_dropout_weightdecay_noclassweight_gamma1.npy",
    "BCE": "calibration-lightning_bce.npy",
    "WFL2": "calibration-lightning_baseline1.npy",
    "WFL1": "calibration-lightning_gamma1.npy",
    "WCE": "calibration-lightning_wce.npy",    
    "IOU": "calibration-lightning_iou.npy"
}
loss_conf_matrix_files = {
    "BFL2": "conf_matrix-lightning_dropout_weightdecay_noclassweight.npy",
    "BFL1": "conf_matrix-lightning_dropout_weightdecay_noclassweight_gamma1.npy",
    "BCE": "conf_matrix-lightning_bce.npy",
    "WFL2": "conf_matrix-lightning_baseline1.npy",
    "WFL1": "conf_matrix-lightning_gamma1.npy",
    "WCE": "conf_matrix-lightning_wce.npy",    
    "IOU": "conf_matrix-lightning_iou.npy"
}


def calibration_by_loss(out_file=None, dataset='valid'):
    occurrence_rate = {
        k: np.load(os.path.join("../results", dataset, fn))
        for (k,fn) in loss_calibration_files.items()
    }
    nbins = len(list(occurrence_rate.values())[0])
    p = np.linspace(0,1,nbins+1)
    p = 0.5 * (p[:-1] + p[1:])

    fig = plots.plot_calibration(p, occurrence_rate, loss_names,
        loss_colors_linestyles)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def pr_curve_by_loss(
    curve="PR", out_file=None, dataset='valid',
    losses=("BFL2", "BCE", "WFL2", "WCE")
):
    conf_matrix = {
        k: np.load(os.path.join("../results", dataset, fn))
        for (k,fn) in loss_conf_matrix_files.items()
        if k in losses
    }

    if curve == "PR":
        fig = plots.plot_pr_curve(conf_matrix, loss_names, loss_colors_linestyles)
    elif curve == "ROC":
        fig = plots.plot_roc_curve(conf_matrix, loss_names, loss_colors_linestyles)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def metric_curves_by_loss(**kwargs):
    metric_curves(loss_conf_matrix_files, loss_names,
        loss_colors_linestyles, **kwargs)


def leadtime_metrics(metrics=("CSI", "PSS"), out_file=None, dataset='test'):
    conf_matrix_files = {
        "FL2": "conf_matrix_leadtime-ensemble_dropout_weightdecay_noclassweight.npy",
        "LAG": "conf_matrix_leadtime-lightning_lagrangian.npy",
        "PER": "conf_matrix_leadtime-lightning_persistence.npy"
    }
    names = {
        "FL2": "Model",
        "LAG": "Lagrangian Pers.",
        "PER": "Eulerian Pers."
    }
    colors_linestyles = {
        ("FL2"): (color_fl2, "-"),
        ("LAG"): (color_lag, "--"),
        ("PER"): (color_per, "-.")
    }
    conf_matrix_lt = {
        k: np.load(os.path.join("../results/", dataset, fn))
        for (k,fn) in conf_matrix_files.items()
    }

    fig = plots.plot_metric_leadtime(conf_matrix_lt, names,
        colors_linestyles=colors_linestyles)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


ensemble_conf_matrix_files = {
    "M1": "conf_matrix-lightning_dropout_weightdecay_noclassweight.npy",
    "M2": "conf_matrix-lightning_dropout_weightdecay_noclassweight2.npy",
    "M3": "conf_matrix-lightning_dropout_weightdecay_noclassweight3.npy",
    "ENS": "conf_matrix-ensemble_dropout_weightdecay_noclassweight.npy",
}
color_ind = "#0072B2"
color_ensemble = "#009E73"
ensemble_colors_linestyles = {
    "M1": (color_ind, "--"),
    "M2": (color_ind, "--"),
    "M3": (color_ind, "--"),
    "ENS": (color_ensemble, "-"),
}
ensemble_names = {
    "M1": "Individual",
    "M2": "_nolegend_",
    "M3": "_nolegend_",
    "ENS": "Ensemble"
}

def metric_curves_dropout_ensemble(**kwargs):
    metric_curves(ensemble_conf_matrix_files, ensemble_names,
        ensemble_colors_linestyles, **kwargs)


def metric_curves(
    conf_matrix_files, names, colors_linestyles,
    metrics=("CSI", "PSS"), out_file=None,
    dataset="valid"
):
    conf_matrix = {
        k: np.load(os.path.join("../results/", dataset, fn))
        for (k,fn) in conf_matrix_files.items()
    }
    thresholds = np.arange(0, 1.0001, 0.001)

    fig = plt.figure(figsize=(6,6))
    for (i,metric) in enumerate(metrics):
        ax = fig.add_subplot(len(metrics), 1, i+1)
        plots.plot_threshold_metric_curve(thresholds,
            conf_matrix, names, metric,
            colors_linestyles=colors_linestyles,
            fig=fig, ax=ax, legend=False, xlabel=(i==len(metrics)-1)
        )
        if i==0:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5,1.03), ncol=4)
        ax.text(
            0.01, 0.975,
            f"({string.ascii_lowercase[i]})",
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes
        )

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)


def plot_examples(
    batch_gen, model, batch_number=13,
    batch_member=30, out_file=None, 
    shown_inputs=("RZC", "occurrence-8-10", "HRV", "ctth-alti"),
    plot_kwargs=None
):
    if plot_kwargs is None:
        plot_kwargs = {}

    names = batch_gen.pred_names_past
    shown_inputs = [names.index(ip) for ip in shown_inputs]

    (X,Y) = batch_gen.batch(batch_number, dataset='test')
    fig = plots.plot_model_examples(X, Y, ["obs", model],
        batch_member=batch_member, shown_inputs=shown_inputs,
        **plot_kwargs)

    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight', dpi=200)
        plt.close(fig)


def plot_all_examples(batch_gen, model,
    shown_inputs=("RZC", "occurrence-8-10", "HRV", "ctth-alti")
):
    samples = ((12, 28), (5, 37), (37, 30))
    for (i,(bn, bm)) in enumerate(samples):
        plot_examples(batch_gen, model,
            batch_number=bn, batch_member=bm, 
            shown_inputs=shown_inputs,
            out_file=f"../figures/model-example-{i}.pdf")


def plot_random_examples(batch_gen, model,
    shown_inputs=("RZC", "occurrence-8-10", "HRV", "ctth-alti"),
    random_seed=1234, num_samples=32
):
    num_batches = len(batch_gen.time_coords['test']) // \
        batch_gen.batch_size
    batch_size = batch_gen.batch_size
    rnd = np.random.RandomState(seed=random_seed)
    all_samples = list(product(range(num_batches), range(batch_size)))
    ind = rnd.choice(len(all_samples), num_samples, replace=False)
    samples = [all_samples[i] for i in ind]

    for (i,(bn, bm)) in enumerate(samples):
        plot_examples(batch_gen, model,
            batch_number=bn, batch_member=bm, 
            shown_inputs=shown_inputs,
            out_file=f"../figures/random-example/random-example-{i:02d}.pdf")


def model_metrics_table(
    metrics=("$T$", "POD", "FAR", "CSI", "ETS", "HSS", "PSS", "ROC AUC", "PR AUC"),
    threshold_metric="CSI",
):
    lines = []
    model_names = {
        "FL2": "FL $\gamma=2$ ensemble",
        "LAG": "Lagrangian pers.",
        "PER": "Eulerian pers.",
    }
    def print_line(items, model=None):
        items = [
            i if isinstance(i,str) else f"${i:.3f}$"
            for i in items
        ]
        items = [model_names.get(model, "")] + items

        line = " & ".join(items) + " \\\\"
        lines.append(line)

    lines.append("\\begin{tabular}{" + "l" + "c"*len(metrics) + "}")
    lines.append("\\topline")
    print_line(metrics)
    lines.append("\\midline")

    conf_matrix_files = {
        "FL2": "conf_matrix-ensemble_dropout_weightdecay_noclassweight.npy",
        "LAG": "conf_matrix-lightning_lagrangian.npy",
        "PER": "conf_matrix-lightning_persistence.npy"
    }
    conf_matrix_test = {
        m: np.load(os.path.join("../results/test", fn))
        for (m,fn) in conf_matrix_files.items()
    }
    conf_matrix_valid = {
        m: np.load(os.path.join("../results/valid", fn))
        for (m,fn) in conf_matrix_files.items()
    }
    thresholds = np.arange(0, 1.0001, 0.001)
    thresh_index = {}

    for (model,cm) in conf_matrix_test.items():
        scores = []
        for metric in metrics:
            if metric == "$T$":
                metric_func = plots.metric_funcs[threshold_metric]
                score = metric_func(conf_matrix_valid[model])
                thresh_index[model] = score.argmax()
                if model in ("LAG", "PER"):
                    scores.append("---")
                else:
                    scores.append(thresholds[thresh_index[model]])
                    
            elif metric.endswith("AUC"):
                if model in ("LAG", "PER"):
                    scores.append("---")
                else:
                    score = plots.metric_funcs[metric](cm)
                    scores.append(score)

            else:
                score = plots.metric_funcs[metric](cm)
                index = thresh_index[model]
                scores.append(score[index])
        
        print_line(scores, model=model)

    lines.append("\\botline")
    lines.append("\\end{tabular}")

    print("\n".join(lines)+"\n")
    return lines
