from pathlib import Path

import pandas as pd

import src.typing as ty
from src import MODEL_ROOTS
from src.tools import TableFormatter
from src.utils.io import load_yaml


def get_models(root: Path,
               exp: str,
               dataset: str,
               ckpt: str = 'last',
               mode: str = '*',
               res: str = 'results',
               models: ty.N[list[str]] = None,
               tag: str = '') -> tuple[dict[str, list[Path]], list[str]]:
    """Find all models and files associated with a particular experiment.
    NOTE: Parameters can use regex expressions, but overlapping names will be combined together. Use at your own risk.

    Found model names can be adjusted using the MODEL_TAGS dictionary.

    :param root: (Path) Root directory to search for models.
    :param exp: (str) Experiment name.
    :param dataset: (str) Evaluation dataset name.
    :param ckpt: (str) Checkpoint mode to retrieve. {last, best, *}
    :param mode: (str) Depth alignment mode to retrieve. {metric, median, lsqr, *}
    :param res: (str) Results directory name.
    :param models: (None|list[str]) List of models to retrieve. (Default: All models will be retrieved)
    :param tag: (str) Tag to append to model names. Include '_' to make more legible.
    :return: (
        eval_files: (dict[str, list[Path]]) Mapping from model names to all found files.
        models: (list[str]) List of model names found.
    )
    """
    if isinstance(models, str): models = models.split()
    fname = f'{dataset}_{ckpt}_{mode}.yaml'

    if not models:
        fs = sorted(root.glob(f'{exp}/**/{res}/{fname}'))
        models = sorted({file.parents[2].stem for file in fs})

    print('Evaluation Models:', models)
    eval_files = {m+tag: sorted(root.glob(f'{exp}/{m}/**/{res}/{fname}')) for m in models}
    eval_files = {k: v for k, v in eval_files.items() if v}
    models = list(eval_files)
    return eval_files, models


def load_dfs(files: dict[str, list[Path]]) -> pd.DataFrame:
    """Load dict of YAML files into a single dataframe.

    :param files: (dict[str, list[Path]]) List of files for each model.
    :return: (DataFrame) Loaded dataframe, index based on the model key and a potential item number.
    """
    dfs = [pd.json_normalize(load_yaml(f)) for fs in files.values() for f in fs]
    df = pd.concat(dfs)

    # Add multi-index based on model and item number, since we don't have mean metrics.
    models = [f'{k}' for k, fs in files.items() for _ in fs]
    df.index = pd.MultiIndex.from_product([models, dfs[0].index], names=['Model', 'Item'])
    return df


def filter_df(df: pd.DataFrame) -> tuple[pd.DataFrame, ty.S[int]]:
    """Preprocess dataframe to include only AbsRel and (F-Score or delta) metrics."""
    metrics, metric_type = ['AbsRel'], [-1]

    delta, delta_legacy = '$\delta_{.25}$', '$\delta < 1.25$'
    f, f_legacy = 'F-Score (10)', 'F-Score'

    # Case where all models used the legacy metric names --> Rename
    if f_legacy in df and f not in df:
        df = df.rename(columns={f_legacy: f})

    if delta_legacy in df and delta not in df:
        df[delta] = 100*df[delta_legacy]
        df = df.drop(columns=[delta_legacy])

    # Append metrics.
    if f in df:
        metrics.append(f)
        metric_type.append(+1)

        # Case where only some models used the legacy metric names --> Combine columns
        if f_legacy in df: df[f] = df[f].fillna(0) + df[f_legacy].fillna(0)

    elif delta in df:
        metrics.append(delta)
        metric_type.append(+1)

        # Case where only some models used the legacy metric names --> Combine columns
        if delta_legacy in df: df[delta] = df[delta].fillna(0) + 100*df[delta_legacy].fillna(0)

    df = df[metrics]
    df = df.rename(columns={'AbsRel': 'Rel', f: 'F'})
    return df, metric_type


def get_df_mean(df: pd.DataFrame, models: ty.S[str],  name: str = 'Mean') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the average metrics and stddev across all model seeds."""
    df2 = df.groupby(level=0)
    df_mean = df2.agg('mean').reindex(models)
    df_std = df2.agg('std').reindex(models)

    df_std.columns.name = 'StdDev'
    df_mean.columns.name = name
    return df_mean, df_std


def add_multitask_metrics(
        df: pd.DataFrame,
        metric_types: ty.S[int],
        ref_idx: int = None) -> tuple[pd.DataFrame, ty.S[int]]:
    """Prepend multi-task metrics computed across all metrics."""
    rel = compute_rel_improvement(df, metric_types, ref=ref_idx)
    df.insert(0, ('MT', '\\%'), rel)
    metric_types.insert(0, 1)

    rank = compute_mean_rank(df, metric_types)
    df.insert(0, ('MT', 'Rank'), rank)
    metric_types.insert(0, -1)
    return df, metric_types


def compute_rel_improvement(df: pd.DataFrame, metric_types: ty.S[int], ref: int = 0) -> pd.Series:
    """Compute average relative improvement w.r.t. a reference row index.

    :param df: (DataFrame) Input dataframe.
    :param metric_types: (list[int]) Metric type for each metric. {+1: Higher is better, -1: Lower is better}
    :param ref: (int) Reference row index to compute relative improvement w.r.t. (Default: 0)
    :return: (DataFrame) Computed relative improvement.
    """
    df2 = (df*metric_types)
    rel = (df2 - df2.iloc[ref])/df2.iloc[ref]

    rel = (100*rel*metric_types).mean(axis=1)
    return rel


def compute_mean_rank(df: pd.DataFrame, metric_types: ty.S[int]) -> pd.Series:
    """Compute the average ranking position across all metrics for each model.

    :param df: (DataFrame) Input dataframe.
    :param metric_types: (list[int]) Metric type for each metric. {+1: Higher is better, -1: Lower is better}
    :return: (DataFrame) Computed average ranking.
    """
    ranks = (df*metric_types).rank(axis=0, ascending=False).mean(axis=1)
    return ranks


def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    root = MODEL_ROOTS[-1]

    splits = [
        # 'kitti_eigen_zhou',
        # 'syns_val',
        # 'mc',
        'kitti_eigen_benchmark',
        'mc',
        'ddad',
        'diode_outdoor',
        'sintel',
        'syns_test',
        'diode_indoors',
        'nyud',
        'tum',
    ]

    ref = 0
    dfs, stds, metric_types = [], [], []

    for split in splits:
        fs, ms = list(zip(*[
            get_models(root, exp='kbr', dataset=split, mode='lsqr', ckpt='last', models='base fwd no_ar_aug no_learn_K no_rand_supp none'),

            get_models(root, exp='benchmark', dataset=split, res='results', mode='lsqr', ckpt='best', models='garg monodepth2_MS diffnet_MS hrdepth_MS'),
            get_models(root, exp='benchmark', dataset=split, res='results', mode='stereo', ckpt='best', models='garg monodepth2_MS diffnet_MS hrdepth_MS'),
            get_models(root, exp='midas', dataset=split, mode='lsqr', ckpt='best', models='MiDaS DPT_Large DPT_BEiT_L_512'),
            get_models(root, exp='newcrfs', dataset=split, mode='lsqr', ckpt='best'),
        ]))

        files = {}
        for f in fs: files |= f
        models = [i for m in ms for i in m]

        df = load_dfs(files)
        df, metric_type = filter_df(df)

        df_mean, df_std = get_df_mean(df, models, name=split)
        dfs.append(df_mean)
        stds.append(df_std)
        metric_types.extend(metric_type)

    for d in dfs: d.columns = pd.MultiIndex.from_product([[d.columns.name], d.columns], names=['dataset', 'metrics'])
    df = pd.concat(dfs, axis=1)
    df, metric_types = add_multitask_metrics(df, metric_types, ref_idx=ref)
    print(TableFormatter.from_df(df, metrics=metric_types).to_latex(precision=2))

    for d in stds: d.columns = pd.MultiIndex.from_product([[d.columns.name], d.columns], names=['dataset', 'metrics'])
    std = pd.concat(stds, axis=1)
    print(TableFormatter.from_df(std, metrics=-1).to_latex(precision=2))


if __name__ == '__main__':
    main()
