try:
    from pathlib import Path

    import hydra
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yaml
    from lib.lib_trainer import *
    from lib.lib_data import *
    from pytorch_lightning import Trainer, seed_everything
except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")

def get_checkpoint_name(checkpoints_path: Path):
    best_loss = sorted([
        model
        for model in checkpoints_path.iterdir()
        if str(model.stem).startswith("best_loss")
    ])

    return str(best_loss[0])

def r2_score(y_true,y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mean_y = np.mean(y_true)
    SSR = np.sum((y_pred - y_true) ** 2)
    SST = np.sum((y_true - mean_y) ** 2)
    return 1 - SSR / SST

def plot_fit(y,y_hat,dpath,target):
    min = np.min([np.min(y), np.min(y_hat)])
    max = np.max([np.max(y), np.max(y_hat)])

    errors = np.abs(np.array(y_hat) - np.array(y))
    # plt.rcParams['font.size']=12

    colormap = plt.get_cmap('plasma')
    color = colormap(np.mean(errors))
    fig = plt.figure(figsize=(8, 8))
    
    outer = gridspec.GridSpec(2, 1, height_ratios = [19,1],
                                left=0.1, right=0.9, bottom=0.1, top=0.9,hspace=0.2
                              )
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 2,
                                           subplot_spec = outer[0],
                                           height_ratios=(.1,1,7),
                                           width_ratios=(7, 1),
                                           wspace=0.01, hspace=0.01)
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[1], width_ratios=(7, 1),wspace=0.01)

    ax_title = fig.add_subplot(gs1[0,0])
    ax_title.set_title(f"{target} - R2 = {r2_score(y,y_hat):.3f}")
    ax_title.axis('off')

    ax_plot = fig.add_subplot(gs1[2, 0])
    sc = ax_plot.scatter(y_hat,y,marker='.', c=errors,cmap=colormap)
    ax_plot.plot([min, max],[min, max],color=color)
    ax_plot.set_xlabel("Predictions")
    ax_plot.set_ylabel("Targets")

    ax_histx = fig.add_subplot(gs1[1, 0], sharex=ax_plot)
    ax_histy = fig.add_subplot(gs1[2, 1], sharey=ax_plot)

    ax_histx.axis('off')
    ax_histy.axis('off')
    ax_histx.set_ylim(0,1)
    ax_histy.set_xlim(0,1)
    ax_histx.hist(y_hat,bins=1000,align='mid', fill=True, color=color,density=True)
    ax_histy.hist(y,bins=1000,orientation='horizontal',align='mid', fill=True, color=color,density=True)

    ax_cbar = fig.add_subplot(gs2[0,0])
    cbar = fig.colorbar(sc, cax=ax_cbar, orientation='horizontal')
    cbar.set_label("Absolute Error (eV/atom)")
    plt.savefig(str(dpath))
    return fig

def write_csv_results(y: list,
                      y_hat: list,
                      names: list,
                      dpath: Path,
                      target: str,):
    df = pd.DataFrame()
    df["id"] = names
    df[f"{target}_real"] = y
    df[f"{target}_predicted"] = y_hat
    df[f"{target}_error_relative"] = (
        np.abs(np.array(y_hat) - np.array(y)) / np.abs(np.array(y)) * 100.0
    )

    df = df.sort_values(by=f"{target}_error_relative", ascending=False)

    df.to_csv(dpath)

@hydra.main(version_base="1.2", config_path="config", config_name="train")
def main(cfg):
    seed_everything(42, workers=True)

    if cfg.predict.specify_checkpoint:
        checkpoints = cfg.predict.ckpt_path
    else:
        checkpoints = get_checkpoint_name(Path(cfg.train.dpath))

    model = PL_EGAT(cfg)

    dataloaders = PLGraphDataLoader(cfg)

    trainer = Trainer(
        deterministic=cfg.deterministic,
        accelerator="gpu",
        devices=1,
        callbacks=[model.get_progressbar()],
    )

    trainer.test(
        model,
        dataloaders,
        ckpt_path=checkpoints,
    )

    print("Maximum % error = {:.5f}%".format(np.max(model.errors)))
    print("Mean % error = {:.5f}%".format(np.mean(model.errors)))
    print("STD % error = {:.5f}%\n".format(np.std(model.errors)))

    performance = {
        "Maximum % error": float(np.max(model.errors)),
        "Mean % error": float(np.mean(model.errors)),
        "STD % error": float(np.std(model.errors)),
    }

    with open(
        str(
            Path(cfg.train.dpath).joinpath(
                f"{cfg.target}_prediction_results.yaml",
            )
        ),
        "w",
    ) as outfile:
        yaml.dump(performance, outfile)

    write_csv_results(
        y=model.plot_y,
        y_hat=model.plot_y_hat,
        names=model.sample_ids,
        dpath=Path(cfg.train.dpath).joinpath(f"{cfg.target}_prediction_results.csv"),
        target=cfg.target,
    )

    plot_fit(
        y=model.plot_y,
        y_hat=model.plot_y_hat,
        dpath=Path(cfg.train.dpath).joinpath(f"{cfg.target}_fit.png"),
        target=cfg.target,
    )

if __name__ == "__main__":
    main()
