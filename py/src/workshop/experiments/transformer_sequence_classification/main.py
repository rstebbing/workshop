##########################################
# File: main.py                          #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

r"""Usage:

python -m workshop.experiments.transformer_sequence_classification all
"""
import argparse
import itertools
import shlex
import sys
import typing as t
from collections import defaultdict
from enum import Enum
from functools import cached_property
from pathlib import Path

import numpy as np
from devtools import debug, sformat

from workshop.ext import pydantic as p

from .experiment import Config, Dataset, Experiment
from .figures import Figures
from .model import EmbeddingsInitStrategy


class ConfigSpec(p.BaseModel):
    model_seeds: t.Optional[t.Iterable[int]] = None
    num_model_seeds: int = 8

    overrides: p._Overrides = None

    @cached_property
    def model_seeds_(self) -> t.List[int]:
        model_seeds = self.model_seeds
        if model_seeds is None:
            model_seeds = range(self.num_model_seeds)

        model_seeds = list(model_seeds)

        return model_seeds


class RunSpec(p.BaseModel):
    name: str
    configs: t.List[ConfigSpec]
    eval_strings: t.List[str] = ["a", "b", "c", "ab"]


RUN_SPECS = [
    RunSpec(
        name="walkthrough",
        configs=[
            ConfigSpec(
                # `model_seed=5` is an example that generalizes correctly.
                model_seeds=[5],
            ),
        ],
        eval_strings=["aac", "baac"],
    ),
    RunSpec(
        name="generalization_errors",
        configs=[
            ConfigSpec(
                # `model_seed=2` is an example that exhibits both false negatives and false positives.
                model_seeds=[2],
            ),
        ],
    ),
    RunSpec(
        name="attend_cls",
        configs=[
            ConfigSpec(
                # `model_seed=8` is an example that generalizes correctly.
                model_seeds=[8],
                overrides=[
                    (("network", "no_attend_cls"), False),
                ],
            ),
        ],
        eval_strings=["aac", "baac"],
    ),
    RunSpec(
        name="all_fixed_length_sequences",
        configs=[
            ConfigSpec(
                num_model_seeds=2,
                overrides=[
                    # Using `AllFixedLengthSequences` does *not* generalize as well.
                    (("experiment", "train_dataset"), Dataset.all_fixed_length_sequences),
                ],
            ),
        ],
    ),
    RunSpec(
        name="all_initializations",
        configs=[
            ConfigSpec(),
        ],
    ),
    RunSpec(
        name="increased_dimensionality",
        configs=[
            ConfigSpec(
                overrides=[
                    (("network", "hidden_size"), 16),
                ],
            ),
        ],
    ),
    RunSpec(
        name="lower_magnitude_initializations",
        configs=[
            ConfigSpec(
                overrides=[
                    (("network", "hidden_size"), 16),
                    (("network", "embeddings_init_strategy"), EmbeddingsInitStrategy.linear_like),
                    (("network", "attention_init_strategy"), "output_fan_out"),
                    (("network", "feed_forward_init_strategy"), "output_fan_out"),
                ],
            ),
        ],
    ),
    RunSpec(
        name="default_embeddings_initialization",
        configs=[
            ConfigSpec(
                overrides=[
                    (("network", "hidden_size"), 16),
                    (("network", "attention_init_strategy"), "output_fan_out"),
                    (("network", "feed_forward_init_strategy"), "output_fan_out"),
                ],
            ),
        ],
    ),
    RunSpec(
        name="increased_num_attention_heads",
        configs=[
            ConfigSpec(
                overrides=[
                    (("network", "hidden_size"), 16),
                    (("network", "num_attention_heads"), 16),
                    (("network", "embeddings_init_strategy"), EmbeddingsInitStrategy.linear_like),
                    (("network", "attention_init_strategy"), "output_fan_out"),
                    (("network", "feed_forward_init_strategy"), "output_fan_out"),
                ],
            ),
        ],
    ),
]


RUN_SPEC_NAMES = [x.name for x in RUN_SPECS]
NAME_TO_RUN_SPEC = {x.name: x for x in RUN_SPECS}
assert len(NAME_TO_RUN_SPEC) == len(RUN_SPECS)


def main_all(args):
    for run_spec in RUN_SPECS:
        # fmt: off
        command = [
            "run", run_spec.name,
            "--directory", args.directory,
        ]
        if args.no_visualize:
            command.extend([
                "--no-visualize",
            ])
        # fmt: on

        exec_main(command, verbose=args.verbose, force=args.force, bold=True)


def main_run(args):
    run_spec = NAME_TO_RUN_SPEC[args.name]

    directory: Path = args.directory / args.name

    configs: t.List[Config] = []
    for config_spec in run_spec.configs:
        for model_seed in config_spec.model_seeds_:
            config = Config()

            # By default the `"CLS"` token is *not* attended to. This ensures the attention value
            # for every attention head is the weighted sum of attention values for *just* the
            # tokens (e.g. `"a"`, `"b"` and/or `"c"`) *without* any bias (which arises from
            # including the `"CLS"` token). Eliminating the potential for the bias prevents models
            # being fitted where one or more attention heads learn to attend to the *absence* of
            # one or more tokens. This occurs rarely, but preventing it entirely simplifies the
            # exposition. This is a bit unconventional and likely not something that should be done
            # with multi-layer models in general.
            config.network.no_attend_cls = True

            # By default layer normalization is *not* used because the optimization is more stable
            # without it for small `hidden_size`.
            config.network.layer_norm_eps = 0.0

            config.apply_overrides(config_spec.overrides)

            config.experiment.model_seed = model_seed

            configs.append(config)

    num_configs = len(configs)
    assert num_configs < 100
    directory_number_format = "{:02d}"

    base_name_to_number = defaultdict()
    base_name_to_number.default_factory = lambda: len(base_name_to_number) + 1

    directories = []
    for c in configs:
        base_name = build_base_name(c, exclude=[("experiment", "model_seed")])
        number = base_name_to_number[base_name]
        number_str = directory_number_format.format(number)
        fragments = [number_str]
        if base_name:
            fragments.append(base_name)
        name = "__".join(fragments)
        d = directory / name / f"model_seed={c.experiment.model_seed}"
        directories.append(d)

        command = ["init", "--directory", d]
        command.extend(build_init_command_args(c))
        exec_main(command, verbose=args.verbose)

        # fmt: off
        command = [
            "train",
            "--directory", d,
            "--max-num-incorrect-strings", 10,
        ]
        # fmt: on

        exec_main(command, verbose=args.verbose, force=args.force)

        figures_directory = d / "figures"

        experiment = Experiment(directory=d)

        for individual_figures, epoch in [(False, None), (True, None), (True, 0)]:
            figures_directory_ = figures_directory
            if individual_figures:
                figures_directory_ = figures_directory_ / (
                    f"epoch={epoch}" if epoch is not None else "min_validation_loss_epoch"
                )

            command: t.List[t.Any] = ["eval"]

            command.extend(run_spec.eval_strings)

            if incorrect_strings := experiment.test_result_.incorrect_strings:
                command.append(incorrect_strings[0])
                if len(incorrect_strings) > 1:
                    command.append(incorrect_strings[-1])

            # fmt: off
            command.extend([
                "--directory", d,
                "--max-num-incorrect-strings", 10,
            ])

            if not args.no_visualize:
                command.extend([
                    "--visualize",
                    "--figures-directory", figures_directory_,
                ])

                if individual_figures:
                    command.extend([
                        "--individual-figures",
                        "--include-transformed-embeddings",
                    ])
                    if epoch is not None:
                        command.extend([
                            "--epoch", epoch,
                        ])
                else:
                    command.extend([
                        "--epoch", -1,
                    ])
            # fmt: on

            exec_main(command, verbose=args.verbose, force=args.force)

    experiments = [Experiment(directory=d) for d in directories]

    test_confusion_matrices = []
    for d, g in itertools.groupby(sorted(experiments, key=lambda e: e.directory), key=lambda e: e.directory.parent):
        g = list(g)
        confusion_matrices = [(e.directory.name, e.test_result_.confusion_matrix) for e in g]
        test_confusion_matrices.append((str(d), confusion_matrices))
    debug(test_confusion_matrices)


def build_base_name(config: Config, *, exclude: t.Optional[t.Container[p._ModelPath]] = None):
    fragments = []
    for (_, shortest_suffix), value in iter_non_default_config_values(config, exclude=exclude):
        name = "_".join(shortest_suffix)
        fragment = f"{name}={value}"
        fragments.append(fragment)

    base_name = "__".join(fragments)

    return base_name


def build_init_command_args(config: Config):
    for (path, _), value in iter_non_default_config_values(config):
        fragments = [name.replace("_", "-") for name in path]
        slug = "-".join(fragments)
        arg = f"--{slug}"
        yield arg

        if value is not True:
            yield value


_CONFIG_PATHS_SHORTEST_SUFFIXES = p.get_model_paths_shortest_suffixes(model_class=Config)
_DEFAULT_CONFIG = Config()
_DEFAULT_VALUES = {path: p.get_model_value(_DEFAULT_CONFIG, path) for path, _ in _CONFIG_PATHS_SHORTEST_SUFFIXES}


def iter_non_default_config_values(
    config: Config, *, exclude: t.Optional[t.Container[p._ModelPath]] = None, enum_values: bool = True
):
    for path, shortest_suffix in _CONFIG_PATHS_SHORTEST_SUFFIXES:
        if exclude is not None and path in exclude:
            continue

        value = p.get_model_value(config, path)
        default_value = _DEFAULT_VALUES[path]
        if value != default_value:
            if enum_values and isinstance(value, Enum):
                value = value.value

            yield (path, shortest_suffix), value


def main_init(args):
    if not args.directory.exists():
        args.directory.mkdir(parents=True)

    experiment = Experiment(directory=args.directory, verbose=args.verbose)
    debug(experiment)

    config_overrides = p.get_model_overrides(args, model_class=Config)
    experiment.init_config(force=args.force, overrides=config_overrides)
    debug(experiment.config_)


def main_train(args):
    experiment = Experiment(directory=args.directory, verbose=args.verbose)
    debug(experiment)

    experiment.train(force=args.force)
    debug(experiment.train_result_)

    experiment.test(force=args.force)
    test_result = experiment.test_result_.copy()
    test_result.incorrect_strings = experiment.test_result_.incorrect_strings[: args.max_num_incorrect_strings]
    debug(test_result)


def main_eval(args):
    experiment = Experiment(directory=args.directory, verbose=args.verbose)
    debug(experiment)

    test_result = experiment.test_result_.copy()
    test_result.incorrect_strings = experiment.test_result_.incorrect_strings[: args.max_num_incorrect_strings]
    debug(test_result)

    if args.epoch is None:
        epochs = [experiment.train_result_.min_validation_loss_epoch]
    elif args.epoch >= 0:
        epochs: t.List[int] = [args.epoch]
    else:
        epochs = list(range(experiment.train_result_.num_epochs_))

    eval_results_ = []
    for i, epoch in enumerate(epochs):
        model = experiment.load_model(epoch)
        eval_results = experiment.eval_(args.strings, model=model)
        eval_results_.append(eval_results)

        if i >= len(epochs) - 1:
            output_probabilities = [eval_result.output_probability for eval_result in eval_results]
            debug(list(zip(args.strings, output_probabilities)))

    if args.visualize:
        from matplotlib import pyplot as plt
        from tqdm import tqdm

        build_figure_path = None  # pyright: ignore
        if args.figures_directory is not None:
            max_epoch = experiment.train_result_.num_epochs_
            num_digits = int(np.ceil(np.log10(max_epoch)))
            figure_stem_format = f"{{epoch:0{num_digits}d}}"

            def build_figure_path(*, epoch: int, num_figs: int, i: int) -> Path:
                figure_stem = figure_stem_format.format(epoch=epoch)
                if num_figs > 1:
                    figure_stem = f"{figure_stem}-{i}"
                figure_name = f"{figure_stem}.png"
                figure_path = args.figures_directory / figure_name

                return figure_path

        figures = Figures(
            experiment=experiment,
            epochs=epochs,
            eval_results=eval_results_,
            individual_figures=args.individual_figures,
            two_1d_attention_heads_to_2d=args.two_1d_attention_heads_to_2d,
            include_transformed_embeddings=args.include_transformed_embeddings,
        )

        required_epochs = epochs
        if not args.force and build_figure_path is not None:
            next_required_epochs: t.List[int] = []
            for epoch, num_figs in figures.plan():
                all_figures_already_saved = True
                for i in range(num_figs):
                    figure_path = build_figure_path(epoch=epoch, num_figs=num_figs, i=i)
                    if not figure_path.is_file():
                        all_figures_already_saved = False
                        break

                if not all_figures_already_saved:
                    next_required_epochs.append(epoch)

            required_epochs = next_required_epochs

        if not required_epochs:
            return

        epochs_figs = figures.plot(epochs=required_epochs)

        for epoch, figs in tqdm(epochs_figs, desc="Figures", total=len(required_epochs), position=0):
            figure_indices = args.figure_indices
            if figure_indices is None:
                figure_indices = range(len(figs))
            else:
                unused_figs = [fig for i, fig in enumerate(figs) if i not in figure_indices]
                for fig in unused_figs:
                    plt.close(fig)

                figs = [figs[i] for i in figure_indices]

            indexed_figs = list(zip(figure_indices, figs))

            if build_figure_path is not None:
                num_figs = len(indexed_figs)

                for i, fig in tqdm(indexed_figs, desc=f"Figures ({epoch = })", position=1, leave=False):
                    if args.individual_figures:
                        fig.subplots_adjust(left=0.10, bottom=0.06, right=0.95, top=0.95)
                        fig.set_size_inches(8, 8)
                    else:
                        fig.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.93, wspace=0.2, hspace=0.2)
                        fig.set_size_inches(30, 15)

                    figure_path = build_figure_path(epoch=epoch, num_figs=num_figs, i=i)
                    figure_path.parent.mkdir(parents=True, exist_ok=True)

                    fig.savefig(figure_path, dpi=180)
                    plt.close(fig)
            else:
                plt.show()


def add_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("-d", "--directory", type=Path, required=True)


_PACKAGE = sys.modules[__name__].__package__


def exec_main(command, *, verbose=True, force=False, bold=False):
    main_argv = []
    if not verbose:
        main_argv.append("--quiet")
    if force:
        main_argv.append("--force")

    main_argv.extend([str(x) for x in command if x is not None])

    styles = []
    if bold:
        styles.extend([sformat.blue, sformat.bold])

    print(sformat(f"+ {shlex.join(['python', '-m', _PACKAGE] + main_argv)}", *styles), file=sys.stderr, flush=True)

    main(main_argv)


DEFAULT_DIRECTORY = Path("data/experiments/transformer_sequence_classification")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quiet", dest="verbose", action="store_false")
    parser.add_argument("-f", "--force", action="store_true")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    parser_all = subparsers.add_parser("all")
    parser_all.add_argument("-d", "--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser_all.add_argument("-nv", "--no-visualize", action="store_true")
    parser_all.set_defaults(main=main_all)

    parser_run = subparsers.add_parser("run")
    parser_run.add_argument("name", choices=RUN_SPEC_NAMES)
    parser_run.add_argument("-d", "--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser_run.add_argument("-nv", "--no-visualize", action="store_true")
    parser_run.set_defaults(main=main_run)

    parser_init = subparsers.add_parser("init")
    add_experiment_arguments(parser_init)
    p.add_model_arguments(parser_init, model_class=Config)
    parser_init.set_defaults(main=main_init)

    parser_train = subparsers.add_parser("train")
    add_experiment_arguments(parser_train)
    parser_train.add_argument("-n", "--max-num-incorrect-strings", type=int)
    parser_train.set_defaults(main=main_train)

    parser_eval = subparsers.add_parser("eval")
    add_experiment_arguments(parser_eval)
    parser_eval.add_argument("strings", nargs="+")
    parser_eval.add_argument("-e", "--epoch", type=int)
    parser_eval.add_argument("-n", "--max-num-incorrect-strings", type=int, default=0)
    parser_eval.add_argument("-v", "--visualize", action="store_true")
    parser_eval.add_argument("-if", "--individual-figures", action="store_true")
    parser_eval.add_argument("-t", "--two-1d-attention-heads-to-2d", action="store_true")
    parser_eval.add_argument("-ite", "--include-transformed-embeddings", action="store_true")
    parser_eval.add_argument("-fd", "--figures-directory", type=Path)
    parser_eval.add_argument("-fi", "--figure-indices", nargs="*", type=int)
    parser_eval.set_defaults(main=main_eval)

    args = parser.parse_args(args=argv)

    args.argv = argv

    args.main(args)


if __name__ == "__main__":
    main()
