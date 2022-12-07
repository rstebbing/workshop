##########################################
# File: figures.py                       #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import abc
import itertools
import textwrap
import typing as t
from collections import defaultdict
from functools import cached_property

import numpy as np
from sklearn.decomposition import PCA

from workshop.ext import pydantic as p

from .experiment import EvalResult, Experiment


class Figures(p.BaseModel):
    experiment: Experiment
    epochs: t.List[int]
    eval_results: t.List[t.List[EvalResult]]

    individual_figures: bool = False
    two_1d_attention_heads_to_2d: bool = False
    include_transformed_embeddings: bool = False

    def __post_init__(self):
        if len(self.epochs) != len(self.eval_results):
            raise ValueError(
                textwrap.dedent(
                    f"""\
                    len(self.epochs) != len(self.eval_results)
                    {len(self.epochs) = }
                    {len(self.eval_results) = }"""
                )
            )

    def plan(self):
        for epoch, figure in zip(self.epochs, self.figures_):
            num_figs = figure.plan()

            yield epoch, num_figs

    def plot(self, *, epochs: t.Optional[t.Iterable[int]] = None):
        transform_epoch = self.experiment.train_result_.min_validation_loss_epoch
        if transform_epoch not in self.epochs:
            transform_epoch = None

        transform_points = defaultdict(list)
        limits_points = defaultdict(list)
        for figure in self.figures_:
            for plotter_key, plotter in zip(figure.plotter_keys_, figure.plotters_):
                if plotter is not None:
                    if (points := plotter.points()) is not None and points.shape[0] > 0:
                        if transform_epoch is None or figure.epoch == transform_epoch:
                            domain_key, _ = plotter_key
                            transform_points[domain_key].append(points)

                        limits_points[plotter_key].append(points)

        transforms = {}
        for domain_key, points_ in transform_points.items():
            assert points_ and all(points.ndim == 2 and points.shape[0] > 0 for points in points_)

            unique_num_dim = {points.shape[1] for points in points_}
            assert len(unique_num_dim) == 1
            num_dim = next(iter(unique_num_dim))

            transform: t.Optional[Transform] = None
            if num_dim > 3:
                unique_points = sorted(set(map(tuple, np.vstack(points_).tolist())))

                points = np.array(unique_points, dtype=points_[0].dtype)
                assert points.ndim == 2 and points.shape[1] == num_dim

                # `points` at the origin are assumed to be points corresponding to the "PAD" token,
                # and should not influence the dimensionality reduction.
                points = points[(points != 0.0).any(axis=1)]

                pca = PCA(n_components=3)
                transformed_points = pca.fit_transform(points)

                reconstructed_points = transformed_points @ pca.components_ + pca.mean_
                reconstruction_error = ((points - reconstructed_points) ** 2).sum(axis=1).mean(axis=0).item()

                if points.shape[0] == 4 and reconstruction_error >= 1e-8:
                    raise ValueError(f"reconstruction_error not near zero\n{reconstruction_error = }")

                transform = Transform(pca=pca, reconstruction_error=reconstruction_error)

            transforms[domain_key] = transform

        limits = {}
        for plotter_key, points_ in limits_points.items():
            domain_key, _ = plotter_key

            points = np.vstack(points_)
            if (transform := transforms[domain_key]) is not None:
                points = transform.transform(points)

            plotter_limits = np.vstack([points.min(axis=0), points.max(axis=0)]).T

            # Increase the dimensions across each axis by 10% (`1.10` below).
            dims = plotter_limits[:, 1] - plotter_limits[:, 0]
            means = plotter_limits.mean(axis=1)
            delta = (1.10 * 0.5) * dims
            plotter_limits[:, 0] = means - delta
            plotter_limits[:, 1] = means + delta

            np.floor(plotter_limits[:, 0], out=plotter_limits[:, 0])
            np.ceil(plotter_limits[:, 1], out=plotter_limits[:, 1])

            # If the minimum and maximum are identical for one or more dimensions, then reduce the minimum
            # and increase the maximum so that their difference is equal to the maximum difference in
            # other dimensions.
            min_equals_max = plotter_limits[:, 0] == plotter_limits[:, 1]
            if (i := np.flatnonzero(min_equals_max)).size > 0:
                j = np.flatnonzero(~min_equals_max)
                dim = (plotter_limits[j, 1] - plotter_limits[j, 0]).max() if j.size > 0 else 1.0
                plotter_limits[i, 0] -= 0.5 * dim
                plotter_limits[i, 1] += 0.5 * dim

            limits[plotter_key] = plotter_limits

        for epoch, figure in zip(self.epochs, self.figures_):
            if epochs is not None and epoch not in epochs:
                continue

            figs = figure.plot(transforms=transforms, limits=limits)

            yield epoch, figs

    @cached_property
    def figures_(self):
        figures: t.List[Figure] = []
        for epoch, eval_results in zip(self.epochs, self.eval_results):
            figure = Figure(
                experiment=self.experiment,
                epoch=epoch,
                eval_results=eval_results,
                individual_figures=self.individual_figures,
                two_1d_attention_heads_to_2d=self.two_1d_attention_heads_to_2d,
                include_transformed_embeddings=self.include_transformed_embeddings,
            )
            figures.append(figure)

        return figures


class Transform(p.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    pca: PCA
    reconstruction_error: float

    def transform(self, points: np.ndarray):
        transformed_points = self.pca.transform(points)

        return transformed_points


PlotterKey = t.Tuple[t.Tuple[str, int], t.Tuple[str, int]]


class Figure(p.BaseModel):
    experiment: Experiment
    epoch: int
    eval_results: t.List[EvalResult]

    individual_figures: bool = False
    two_1d_attention_heads_to_2d: bool = False
    include_transformed_embeddings: bool = False

    @cached_property
    def num_rows_(self):
        num_rows = max(2, len(self.plotters_) // 8 + 1) if len(self.plotters_) > 1 else 1

        return num_rows

    @cached_property
    def num_cols_(self):
        num_cols = int(np.ceil(len(self.plotters_) / self.num_rows_))

        return num_cols

    def plan(self):
        num_figs = sum(plotter is not None for plotter in self.plotters_) if self.individual_figures else 1

        return num_figs

    def plot(self, *, transforms=None, limits=None):
        if transforms is None:
            transforms = {}
        if limits is None:
            limits = {}

        from matplotlib import pyplot as plt

        plt.style.use("seaborn-v0_8")

        # Disable the warning: More than 20 figures have been opened.
        plt.rcParams["figure.max_open_warning"] = -1

        figs: t.List[plt.Figure] = []
        axs = []

        fig: plt.Figure
        if not self.individual_figures:
            fig = plt.figure()  # pyright: ignore[reportGeneralTypeIssues]
            figs.append(fig)

        for i, plotter in enumerate(self.plotters_):
            if plotter is None:
                continue

            if self.individual_figures:
                fig = plt.figure()  # pyright: ignore[reportGeneralTypeIssues]
                figs.append(fig)
                args = (1, 1, 1)
            else:
                fig = figs[-1]
                args = (self.num_rows_, self.num_cols_, i + 1)

            kwargs = {}
            if (points := plotter.points()) is not None:
                if points.shape[1] == 2:
                    pass
                elif points.shape[1] >= 3:
                    kwargs["projection"] = "3d"

            ax = fig.add_subplot(*args, **kwargs) if kwargs is not None else None
            axs.append(ax)

        for plotter_key, plotter, ax in zip(self.plotter_keys_, self.plotters_, axs):
            if plotter is None or ax is None:
                continue

            domain_key, _ = plotter_key
            plotter_transform = transforms.get(domain_key)
            plotter_limits = limits.get(plotter_key)
            plotter.plot(ax, transform=plotter_transform, limits=plotter_limits)

        if not self.individual_figures:
            assert len(figs) == 1

            title = f"{str(self.experiment.directory)!r}\nepoch = {self.epoch}"
            if self.epoch == self.experiment.train_result_.min_validation_loss_epoch:
                title += " (min_validation_loss_epoch)"

            figs[0].suptitle(title)

        return figs

    @cached_property
    def plotter_keys_(self):
        plotter_keys: t.List[PlotterKey] = []
        for i, plotter in enumerate(self.plotters_):
            (domain_key, limits_key) = plotter.points_key() if plotter is not None else ("", "")
            plotter_key = (
                (domain_key, i if limits_key == "" else 0),
                (limits_key, i if limits_key == "" else 0),
            )
            plotter_keys.append(plotter_key)

        assert len(plotter_keys) == len(self.plotters_)

        return plotter_keys

    @cached_property
    def plotters_(self) -> t.List[t.Optional["_Plotter"]]:
        plotters = [
            LossesPlotter(experiment=self.experiment, epoch=self.epoch),
            EmbeddingsPlotter(experiment=self.experiment, epoch=self.epoch),
        ]

        for eval_result in self.eval_results:
            plotters.append(SequenceEmbeddingsPlotter(experiment=self.experiment, eval_result=eval_result))

        plotters.extend(
            [
                AttentionNormalizedHiddenStatesPlotter(
                    experiment=self.experiment, epoch=self.epoch, eval_results=self.eval_results
                ),
            ]
        )

        if self.experiment.config_.network.attention_size_ < 2:
            raise ValueError("unsupported attention_size_\n{self.experiment.config_.network.attention_size_ = }")

        heads = (
            [None]
            if self.two_1d_attention_heads_to_2d and self.experiment.config_.network.attention_size_ == 2
            else range(self.experiment.config_.network.num_attention_heads)
        )

        for plotter_cls in [
            AttentionKeysClsQueryPlotter,
            AttentionValuesPlotter,
            ClsAttentionValuesPlotter,
        ]:
            for i, head in enumerate(heads):
                kwargs = {}
                if issubclass(plotter_cls, (AttentionKeysClsQueryPlotter, _ClsPlotter)):
                    kwargs["add_legend"] = i == 0

                plotters.append(
                    plotter_cls(
                        experiment=self.experiment,
                        epoch=self.epoch,
                        eval_results=self.eval_results,
                        head=head,
                        **kwargs,
                    )
                )

        plotters.extend(
            [
                ClsAttentionOutputPlotter(
                    experiment=self.experiment, epoch=self.epoch, eval_results=self.eval_results, add_legend=False
                ),
                ClsAttentionOutputHiddenStatesPlotter(
                    experiment=self.experiment, epoch=self.epoch, eval_results=self.eval_results, add_legend=False
                ),
                ClsFeedForwardNormalizedHiddenStatesPlotter(
                    experiment=self.experiment, epoch=self.epoch, eval_results=self.eval_results, add_legend=False
                ),
                ClsFeedForwardOutputsPlotter(
                    experiment=self.experiment, epoch=self.epoch, eval_results=self.eval_results, add_legend=False
                ),
                ClsFeedForwardOutputHiddenStatesPlotter(
                    experiment=self.experiment, epoch=self.epoch, eval_results=self.eval_results
                ),
                ParameterMagnitudesPlotter(experiment=self.experiment, epoch=self.epoch),
            ]
        )

        if self.include_transformed_embeddings:
            for eval_result in self.eval_results:
                plotters.append(
                    SequenceTransformedEmbeddingsPlotter(experiment=self.experiment, eval_result=eval_result)
                )

        return plotters


class _Plotter(abc.ABC):
    def points_key(self) -> t.Tuple[str, str]:
        return ("", "")

    @abc.abstractmethod
    def points(self):
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self, ax, *, transform=None, limits=None):
        raise NotImplementedError


class LossesPlotter(p.BaseModel, _Plotter):
    experiment: Experiment
    epoch: int

    def points(self):
        return None

    def plot(self, ax, *, transform=None, limits=None):
        if transform is not None:
            raise ValueError(f"transform must be empty\n{transform = }")
        if limits is not None:
            raise ValueError(f"limits must be empty\n{limits = }")

        epochs = np.arange(1, self.experiment.train_result_.num_epochs_)

        train_losses = self.experiment.train_result_.train_losses
        train_line: t.Any
        (train_line,) = ax.semilogy(epochs, train_losses, ".-", label="train_losses")

        validation_ax = ax
        if self.experiment.config_.experiment.validation_loss_ != self.experiment.config_.experiment.train_loss:
            validation_ax = ax.twinx()

        validation_losses = self.experiment.train_result_.validation_losses
        validation_line: t.Any
        (validation_line,) = validation_ax.semilogy(epochs, validation_losses, ".-", label="validation_losses")

        if self.epoch > 0:
            ax.semilogy([self.epoch], [train_losses[self.epoch - 1]], "o", c=train_line.get_color())
            validation_ax.semilogy(
                [self.epoch], [validation_losses[self.epoch - 1]], "o", c=validation_line.get_color()
            )

        ax.legend()

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")


class EmbeddingsPlotter(p.BaseModel, _Plotter):
    experiment: Experiment
    epoch: int

    def points_key(self):
        return ("hidden_states", "embeddings")

    def points(self):
        return self.embeddings_

    def plot(self, ax, *, transform=None, limits=None):
        _scatter_token_points(
            ax=ax,
            points=_maybe_transform(self.embeddings_, transform),
            tokens=self.experiment.tokenizer_.vocabulary_,
            title=_maybe_append_transform_info("embeddings", transform),
            limits=limits,
        )

    @cached_property
    def embeddings_(self):
        model = self.experiment.load_model(self.epoch)
        embeddings = model.embeddings.embeddings.weight.detach().numpy()

        return embeddings


class _SequenceEmbeddingsPlotter(p.BaseModel, _Plotter):
    experiment: Experiment
    eval_result: EvalResult

    @abc.abstractmethod
    def _get_title(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_embeddings(self) -> np.ndarray:
        raise NotImplementedError

    def points(self):
        return self.embeddings_

    def plot(self, ax, *, transform=None, limits=None):
        title = self._get_title()

        assert len(self.tokens_) == len(self.counts_)

        tokens_with_count = []
        for token, count in zip(self.tokens_, self.counts_):
            assert count >= 1
            if count > 1:
                token = f"{token}{{{count}}}"

            tokens_with_count.append(token)

        _scatter_token_points(
            ax=ax,
            points=_maybe_transform(self.embeddings_, transform),
            tokens=tokens_with_count,
            title=title,
            limits=limits,
        )

    @cached_property
    def tokens_(self):
        return self._data[0]

    @cached_property
    def counts_(self):
        return self._data[1]

    @cached_property
    def embeddings_(self):
        return self._data[2]

    @cached_property
    def _data(self):
        token_ids = self.eval_result.model_result.token_ids[self.eval_result.i]
        tokens = [self.experiment.tokenizer_.vocabulary_[token_id] for token_id in token_ids]

        embeddings = self._get_embeddings()

        tokens, counts, embeddings = _unique_tokens_hidden_states(self.experiment, tokens, embeddings)

        return tokens, counts, embeddings


class SequenceEmbeddingsPlotter(_SequenceEmbeddingsPlotter):
    def points_key(self):
        return ("hidden_states", "embeddings")

    def _get_title(self) -> str:
        title = f"embeddings: {self.eval_result.string!r}"

        return title

    def _get_embeddings(self) -> np.ndarray:
        embeddings = self.eval_result.model_result.embeddings[self.eval_result.i].numpy()

        return embeddings


class SequenceTransformedEmbeddingsPlotter(_SequenceEmbeddingsPlotter):
    def points_key(self):
        return ("transformed_embeddings", "transformed_embeddings")

    def _get_title(self) -> str:
        title = f"transformed_embeddings: {self.eval_result.string!r}"

        return title

    def _get_embeddings(self) -> np.ndarray:
        transformed_embeddings = self.eval_result.model_result.self_attention_block_result.output_hidden_states[
            self.eval_result.i
        ].numpy()

        return transformed_embeddings


class AttentionNormalizedHiddenStatesPlotter(p.BaseModel, _Plotter):
    experiment: Experiment
    epoch: int
    eval_results: t.List[EvalResult]

    def points_key(self):
        return ("hidden_states", "embeddings")

    def points(self):
        return self.hidden_states_

    def plot(self, ax, *, transform=None, limits=None):
        _scatter_token_points(
            ax=ax,
            points=_maybe_transform(self.hidden_states_, transform),
            tokens=self.tokens_,
            title="attention_normalized_hidden_states",
            limits=limits,
        )

    @cached_property
    def tokens_(self):
        return self._data[0]

    @cached_property
    def hidden_states_(self):
        return self._data[1]

    @cached_property
    def _data(self):
        token_to_hidden_state = {}
        for eval_result in self.eval_results:
            token_ids = eval_result.model_result.token_ids[eval_result.i]
            tokens = [self.experiment.tokenizer_.vocabulary_[token_id] for token_id in token_ids]

            attention_normalized_hidden_states = (
                eval_result.model_result.self_attention_block_result.attention_normalized_hidden_states
            )
            assert attention_normalized_hidden_states is not None
            hidden_states = attention_normalized_hidden_states[eval_result.i].numpy()

            for token, hidden_state in zip(tokens, hidden_states):
                token_to_hidden_state[token] = hidden_state

        tokens_hidden_states = sorted(
            token_to_hidden_state.items(), key=lambda x: self.experiment.tokenizer_.token_to_token_id_[x[0]]
        )

        tokens = []
        hidden_states = np.empty(
            (len(tokens_hidden_states), self.experiment.config_.network.hidden_size), dtype=np.float32
        )
        for i, (token, hidden_state) in enumerate(tokens_hidden_states):
            tokens.append(token)

            assert hidden_state.dtype == hidden_states.dtype
            hidden_states[i] = hidden_state

        return tokens, hidden_states


class AttentionKeysClsQueryPlotter(p.BaseModel, _Plotter):
    experiment: Experiment
    epoch: int
    eval_results: t.List[EvalResult]

    head: t.Optional[int] = None

    add_legend: bool = True

    def points(self):
        return self.points_

    def plot(self, ax, *, transform=None, limits=None):
        keys = _maybe_transform(self.keys_, transform)
        queries = _maybe_transform(self.cls_queries_, transform)
        _plot_vectors(
            ax=ax,
            labeled_vectors=[
                ("key", keys, self.tokens_),
                ("query", queries, self.cls_tokens_),
            ],
        )

        if self.add_legend:
            ax.legend()

        _apply_limits_and_labels(ax, keys.shape[1], limits=limits)

        title = _maybe_append_head("token_attention_keys_and_queries", self.head)

        ax.set_title(title)

    @cached_property
    def points_(self):
        points = np.vstack([self.keys_, self.cls_queries_])

        return points

    @cached_property
    def tokens_(self):
        return self._data[0]

    @cached_property
    def keys_(self):
        return self._data[1]

    @cached_property
    def cls_tokens_(self):
        return self._data[2]

    @cached_property
    def cls_queries_(self):
        return self._data[3]

    @cached_property
    def _data(self):
        skip_tokens = _skip_tokens(self.experiment)
        tokens, keys = _get_token_attention_result(self.experiment, self.eval_results, "K", skip_tokens=skip_tokens)

        if self.head is not None:
            keys = _get_head(self.experiment, keys, self.head)

        tokens_, queries = _get_token_attention_result(self.experiment, self.eval_results, "Q")
        cls_query_index = tokens_.index(self.experiment.tokenizer_.cls_token_)
        cls_query = queries[cls_query_index]

        cls_tokens = [self.experiment.tokenizer_.cls_token_]
        cls_queries = cls_query[np.newaxis]
        if self.head is not None:
            cls_queries = _get_head(self.experiment, cls_queries, self.head)

        keys = _atleast_2d(keys, name="keys")
        cls_queries = _atleast_2d(cls_queries, name="cls_queries")

        return tokens, keys, cls_tokens, cls_queries


class AttentionValuesPlotter(p.BaseModel, _Plotter):
    experiment: Experiment
    epoch: int
    eval_results: t.List[EvalResult]

    head: t.Optional[int] = None

    def points_key(self):
        points_key = (f"attention_values_{self.head}", "values")

        return points_key

    def points(self):
        return self.values_

    def plot(self, ax, *, transform=None, limits=None):
        points = _maybe_transform(self.values_, transform)

        title = _maybe_append_head("token_attention_values", self.head)

        _scatter_token_points(ax=ax, points=points, tokens=self.tokens_, title=title, limits=limits)

    @cached_property
    def tokens_(self):
        return self._data[0]

    @cached_property
    def values_(self):
        return self._data[1]

    @cached_property
    def _data(self):
        skip_tokens = _skip_tokens(self.experiment)
        tokens, values = _get_token_attention_result(self.experiment, self.eval_results, "V", skip_tokens=skip_tokens)

        if self.head is not None:
            values = _get_head(self.experiment, values, self.head)

        values = _atleast_2d(values, name="values")

        return tokens, values


class _ClsPlotter(p.BaseModel, _Plotter):
    experiment: Experiment
    epoch: int
    eval_results: t.List[EvalResult]

    head: t.Optional[int] = None

    add_legend: bool = True

    @abc.abstractmethod
    def _get_title(self, transform: t.Optional[Transform]) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_values(self, eval_result: EvalResult):
        raise NotImplementedError

    def _get_values_size(self):
        return self.experiment.config_.network.hidden_size

    def _get_labeled_vectors(self):
        return None

    def points_key(self):
        return ("transformed_embeddings", "transformed_cls_embeddings")

    def points(self):
        return self.points_

    @cached_property
    def points_(self):
        points_ = [self.cls_values_]
        if self.labeled_vectors_:
            for _, vectors, _ in self.labeled_vectors_:  # pyright: ignore[reportGeneralTypeIssues]
                points_.append(vectors)

        points = np.vstack(points_) if len(points_) > 1 else points_[0]

        return points

    def plot(self, ax, *, transform=None, limits=None):
        title = self._get_title(transform)
        title = _maybe_append_head(title, self.head)

        labeled_vectors = self.labeled_vectors_
        if labeled_vectors:
            labeled_vectors = [
                (label, _maybe_transform(vectors, transform), texts)
                for label, vectors, texts in labeled_vectors  # pyright: ignore[reportGeneralTypeIssues]
            ]

        _scatter_sequence_points(
            ax=ax,
            points=_maybe_transform(self.cls_values_, transform),
            strings=self.strings_,
            positive_character_set=self.experiment.config_.data.positive_character_set,
            output_probabilities=self.output_probabilities_,
            title=title,
            labeled_vectors=labeled_vectors,
            add_legend=self.add_legend,
            limits=limits,
        )

    @cached_property
    def strings_(self):
        return self._data[0]

    @cached_property
    def cls_values_(self):
        return self._data[1]

    @cached_property
    def output_probabilities_(self):
        return self._data[2]

    @cached_property
    def labeled_vectors_(self):
        return self._data[3]

    @cached_property
    def _data(self):
        strings = []
        cls_values = np.empty((len(self.eval_results), self._get_values_size()), dtype=np.float32)
        output_probabilities = np.empty((len(self.eval_results),), dtype=np.float32)
        for i, eval_result in enumerate(self.eval_results):
            strings.append(eval_result.string)

            values = self._get_values(eval_result)

            token_ids = eval_result.model_result.token_ids[eval_result.i]
            assert (
                token_ids.ndim == 1
                and token_ids.shape[0] > 0
                and token_ids[0] == self.experiment.tokenizer_.cls_token_id_
            )
            cls_value = values[0]

            cls_values[i] = cls_value.numpy()

            output_probabilities[i] = eval_result.output_probability

        if self.head is not None:
            cls_values = _get_head(self.experiment, cls_values, self.head)

        cls_values = _atleast_2d(cls_values, name="cls_values")

        labeled_vectors = self._get_labeled_vectors()

        return strings, cls_values, output_probabilities, labeled_vectors


class ClsAttentionValuesPlotter(_ClsPlotter):
    def points_key(self):
        points_key = (f"attention_values_{self.head}", "values")

        return points_key

    def _get_title(self, transform: t.Optional[Transform]):
        return "cls_attention_values"

    def _get_values(self, eval_result: EvalResult):
        attention_result = eval_result.model_result.self_attention_block_result.attention_result
        assert attention_result is not None
        attention_values = attention_result.attention_values[eval_result.i]

        return attention_values

    def _get_values_size(self):
        return self.experiment.config_.network.attention_size_

    def _get_labeled_vectors(self):
        labeled_vectors = []

        if False:
            model = self.experiment.load_model(self.epoch)

            if (attention := model.self_attention_block.attention) is not None:
                attention_head_output_weights = (
                    attention.output.weight.detach()
                    .numpy()
                    .reshape(
                        -1,
                        self.experiment.config_.network.num_attention_heads,
                        self.experiment.config_.network.attention_head_size,
                    )[:, self.head]
                )
                labeled_vectors.extend(
                    (f"attention_output[{i}]", x[np.newaxis], [None])
                    for i, x in enumerate(attention_head_output_weights)
                )

        return labeled_vectors


class ClsAttentionOutputPlotter(_ClsPlotter):
    def _get_title(self, transform: t.Optional[Transform]):
        title = _maybe_append_transform_info("cls_attention_output", transform)

        return title

    def _get_values(self, eval_result: EvalResult):
        attention_result = eval_result.model_result.self_attention_block_result.attention_result
        assert attention_result is not None
        attention_outputs = attention_result.output[eval_result.i]

        return attention_outputs


class ClsAttentionOutputHiddenStatesPlotter(_ClsPlotter):
    def _get_title(self, transform: t.Optional[Transform]):
        return "cls_attention_output_hidden_states"

    def _get_values(self, eval_result: EvalResult):
        attention_output_hidden_states_ = (
            eval_result.model_result.self_attention_block_result.attention_output_hidden_states
        )
        assert attention_output_hidden_states_ is not None
        attention_output_hidden_states = attention_output_hidden_states_[eval_result.i]

        return attention_output_hidden_states


class ClsFeedForwardNormalizedHiddenStatesPlotter(_ClsPlotter):
    def _get_title(self, transform: t.Optional[Transform]):
        return "cls_feed_forward_normalized_hidden_states"

    def _get_values(self, eval_result: EvalResult):
        feed_forward_normalized_hidden_states_ = (
            eval_result.model_result.self_attention_block_result.feed_forward_normalized_hidden_states
        )
        assert feed_forward_normalized_hidden_states_ is not None
        feed_forward_normalized_hidden_states = feed_forward_normalized_hidden_states_[eval_result.i]

        return feed_forward_normalized_hidden_states


class ClsFeedForwardOutputsPlotter(_ClsPlotter):
    def _get_title(self, transform: t.Optional[Transform]):
        return "cls_feed_forward_outputs"

    def _get_values(self, eval_result: EvalResult):
        feed_forward_output = eval_result.model_result.self_attention_block_result.feed_forward_output
        assert feed_forward_output is not None
        feed_forward_outputs = feed_forward_output[eval_result.i]

        return feed_forward_outputs


class ClsFeedForwardOutputHiddenStatesPlotter(_ClsPlotter):
    def _get_title(self, transform: t.Optional[Transform]):
        return "cls_feed_forward_output_hidden_states"

    def _get_values(self, eval_result: EvalResult):
        feed_forward_output_hidden_states_ = (
            eval_result.model_result.self_attention_block_result.feed_forward_output_hidden_states
        )
        assert feed_forward_output_hidden_states_ is not None
        feed_forward_output_hidden_states = feed_forward_output_hidden_states_[eval_result.i]

        return feed_forward_output_hidden_states

    def _get_labeled_vectors(self):
        model = self.experiment.load_model(self.epoch)
        binary_linear = model.binary_linear.linear.weight.detach().numpy()
        return [("binary_linear", binary_linear, [None])]


class ParameterMagnitudesPlotter(p.BaseModel, _Plotter):
    experiment: Experiment
    epoch: int

    def points(self):
        return None

    def plot(self, ax, *, transform=None, limits=None):
        if transform is not None:
            raise ValueError(f"transform must be empty\n{transform = }")
        if limits is not None:
            raise ValueError(f"limits must be empty\n{limits = }")

        model = self.experiment.load_model(self.epoch)

        c = self.experiment.config_.network

        bars: t.List[t.Tuple[str, np.ndarray]] = []
        unhandled_parameters: t.List[str] = []
        for name, param in model.named_parameters():
            X = param.detach().numpy()

            magnitudes: t.Optional[np.ndarray] = None
            if name == "embeddings.embeddings.weight":
                magnitudes = np.sqrt((X**2).sum(axis=1))
            elif name in {
                "self_attention_block.attention.query.weight",
                "self_attention_block.attention.key.weight",
                "self_attention_block.attention.value.weight",
            }:
                X = X.reshape(c.num_attention_heads, c.attention_head_size, c.hidden_size)
                magnitudes = np.sqrt((X**2).sum(axis=2).sum(axis=1))
            elif name in {"self_attention_block.attention.output.weight"}:
                X = X.reshape(c.hidden_size, c.num_attention_heads, c.attention_head_size)
                magnitudes = np.sqrt((X**2).sum(axis=2).sum(axis=0))
            elif name == "self_attention_block.feed_forward.input_.weight":
                magnitudes = np.sqrt((X**2).sum(axis=1))
            elif name == "self_attention_block.feed_forward.output.weight":
                magnitudes = np.sqrt((X**2).sum(axis=0))
            elif name == "binary_linear.linear.weight":
                magnitudes = np.sqrt((X**2).sum(axis=1))
            else:
                unhandled_parameters.append(name)

            if magnitudes is not None:
                assert magnitudes.ndim == 1
                bars.append((name, magnitudes))

        if unhandled_parameters:
            raise ValueError(f"one or more unhandled parameters\n{unhandled_parameters = }")

        offset = 0
        for name, magnitudes in bars:
            xs = np.arange(offset, offset + magnitudes.shape[0])
            ax.bar(xs, magnitudes, width=0.6, align="edge", label=name)
            offset += magnitudes.shape[0]

        ax.legend()

        ax.set_xticks([])
        ax.set_ylabel("magnitude")

        ax.set_title("parameter_magnitudes")


def _maybe_transform(points, transform: t.Optional[Transform]):
    transformed_points = transform.transform(points) if transform is not None else points

    return transformed_points


def _maybe_append_transform_info(title: str, transform: t.Optional[Transform]):
    if transform is not None:
        explained_variance: float = transform.pca.explained_variance_ratio_.sum().item()
        title = (
            f"{title} "
            f"(explained variance: {explained_variance * 100.0:.2f}%, "
            f"reconstruction error: {transform.reconstruction_error:.2g})"
        )

    return title


def _unique_tokens_hidden_states(experiment: Experiment, tokens: t.List[str], hidden_states: np.ndarray):
    assert hidden_states.shape == (len(tokens), experiment.config_.network.hidden_size)

    token_to_count = {}
    unique_tokens = []
    unique_token_indices = []
    for i, token in enumerate(tokens):
        count = token_to_count.get(token, 0)

        if count == 0:
            unique_tokens.append(token)
            unique_token_indices.append(i)

        count += 1
        token_to_count[token] = count

    counts = [token_to_count[token] for token in unique_tokens]
    hidden_states = hidden_states[unique_token_indices]

    return unique_tokens, counts, hidden_states


def _skip_tokens(experiment: Experiment) -> t.Set[str]:
    skip_tokens = set(experiment.config_.tokenizer.special_tokens_)
    if not experiment.config_.network.no_attend_cls:
        skip_tokens.discard(experiment.config_.tokenizer.cls_token_)

    return skip_tokens


def _get_token_attention_result(
    experiment: Experiment,
    eval_results: t.List[EvalResult],
    property_: str,
    *,
    skip_tokens: t.Optional[t.Container[str]] = None,
):
    assert property_ in "QKV"

    token_to_x = {}
    for eval_result in eval_results:
        token_ids = eval_result.model_result.token_ids[eval_result.i]
        tokens = [experiment.tokenizer_.vocabulary_[token_id] for token_id in token_ids]

        attention_result = eval_result.model_result.self_attention_block_result.attention_result
        assert attention_result is not None
        X = getattr(attention_result, property_)[eval_result.i].numpy()

        max_sequence_length = X.shape[1]
        assert X.shape == (
            experiment.config_.network.num_attention_heads,
            max_sequence_length,
            experiment.config_.network.attention_head_size,
        )

        xs = X.transpose(1, 0, 2).reshape(max_sequence_length, -1)
        assert xs.shape == (max_sequence_length, experiment.config_.network.attention_size_)

        for token, x in zip(tokens, xs):
            if not skip_tokens or token not in skip_tokens:
                token_to_x[token] = x

    tokens_xs = sorted(token_to_x.items(), key=lambda x: experiment.tokenizer_.token_to_token_id_[x[0]])

    tokens = []
    xs = np.empty((len(tokens_xs), experiment.config_.network.attention_size_), dtype=np.float32)
    for i, (token, x) in enumerate(tokens_xs):
        tokens.append(token)

        assert x.dtype == xs.dtype
        xs[i] = x

    return tokens, xs


def _get_head(experiment: Experiment, values: np.ndarray, head: int):
    c = experiment.config_.network
    values = values.reshape(-1, c.num_attention_heads, c.attention_head_size)

    return values[:, head]


def _atleast_2d(values: np.ndarray, name="values"):
    if values.shape[1] < 1:
        raise ValueError(f"{name}.shape[1] < 1\n{name}.shape[1] = {values.shape[1]!r}")

    if values.shape[1] < 2:
        values = np.c_[values, [0] * len(values)]

    return values


def _scatter_token_points(*, ax, points, tokens, title, limits=None):
    assert len(tokens) == points.shape[0]

    ax.scatter(*points.T, marker="o")

    for hidden_state, token in zip(points, tokens):
        ax.text(*hidden_state, token)

    _apply_limits_and_labels(ax, points.shape[1], limits=limits)

    ax.set_title(title)


def _scatter_sequence_points(
    *,
    ax,
    points,
    strings,
    positive_character_set,
    output_probabilities,
    title,
    labeled_vectors=None,
    add_legend=True,
    limits=None,
):
    assert len(strings) == points.shape[0]

    pred_label = output_probabilities >= 0.5
    true_label = np.array([all(c in string for c in positive_character_set) for string in strings], dtype=bool)

    true_negatives = np.flatnonzero(~true_label & ~pred_label)
    true_positives = np.flatnonzero(true_label & pred_label)
    false_negatives = np.flatnonzero(true_label & ~pred_label)
    false_positives = np.flatnonzero(~true_label & pred_label)

    # (Set the `label` to `None` if there are no points so that it is not needlessly included in the legend.)
    ax.scatter(*points[true_negatives].T, marker="D", label="true_negatives" if true_negatives.size > 0 else None)
    ax.scatter(*points[true_positives].T, marker="D", label="true_positives" if true_positives.size > 0 else None)
    ax.scatter(*points[false_negatives].T, marker="D", label="false_negatives" if false_negatives.size > 0 else None)
    ax.scatter(*points[false_positives].T, marker="D", label="false_positives" if false_positives.size > 0 else None)

    for hidden_state, string in zip(points, strings):
        ax.text(*hidden_state, string)

    if labeled_vectors is not None:
        # `colors_start=4` to skip the first four colors used by the four calls to `ax.scatter` directly above.
        _plot_vectors(ax=ax, labeled_vectors=labeled_vectors, colors_start=4)

    if add_legend:
        ax.legend()

    _apply_limits_and_labels(ax, points.shape[1], limits=limits)

    ax.set_title(title)


def _plot_vectors(*, ax, labeled_vectors, colors_start=0):
    from matplotlib import pyplot as plt

    from workshop.ext.matplotlib import FancyArrow3DPatch, mpatches

    # Reference:
    # https://stackoverflow.com/a/56212312
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    assert prop_cycle is not None
    colors: t.List[str] = prop_cycle.by_key()["color"]

    iter_colors = itertools.islice(itertools.cycle(colors), colors_start, None)

    base_arrow_kwargs = {
        "arrowstyle": mpatches.ArrowStyle("Simple", head_length=1.0, head_width=1.0),
        "mutation_scale": 10,
        "shrinkA": 0,
        "shrinkB": 0,
    }

    for label, vectors, texts in labeled_vectors:
        assert vectors.shape[0] == len(texts)

        color = next(iter_colors)

        arrow_kwargs = base_arrow_kwargs.copy()
        arrow_kwargs.update(
            {
                "color": color,
                "label": label,
            }
        )

        for vector, text in zip(vectors, texts):
            if len(vector) == 2:
                arrow_cls = mpatches.FancyArrowPatch
            else:
                assert len(vector) == 3
                arrow_cls = FancyArrow3DPatch

            zeros = [0] * len(vector)
            arrow = arrow_cls(zeros, vector, **arrow_kwargs)
            ax.add_patch(arrow)

            # (Only include the label *once*.)
            arrow_kwargs.pop("label", None)

            if text is not None:
                ax.text(*vector, text)


def _apply_limits_and_labels(ax, ndim, *, limits):
    if ndim not in {2, 3}:
        raise ValueError(f"ndim not in {2, 3}\n{ndim = }")

    if limits is not None:
        if limits.shape[0] != ndim:
            raise ValueError(f"limits.shape[0] != ndim\n{limits.shape[0] = }\n{ndim = }")

        if limits.shape[0] >= 2:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
        if limits.shape[0] >= 3:
            ax.set_zlim(limits[2])

    ax.set_aspect("equal")

    if ndim >= 2:
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    if ndim >= 3:
        ax.set_zlabel("$z$")


def _maybe_append_head(title: str, head: t.Optional[int]):
    if head is not None:
        title = f"{title} (head: {head})"

    return title
