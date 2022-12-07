##########################################
# File: experiment.py                    #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import textwrap
import time
import typing as t
from collections import Counter
from enum import Enum
from functools import cached_property

import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader

from workshop.ext import pydantic as p

from .data import AllFixedLengthSequences, Batch, DataConfig, Item, VariableLengthSequences, collate_items
from .model import Model, ModelConfig, ModelResult, NetworkConfig, validate_token_ids
from .tokenizer import Tokenizer, TokenizerConfig


class Dataset(str, Enum):
    variable_length_sequences = "variable_length_sequences"
    all_fixed_length_sequences = "all_fixed_length_sequences"


class Loss(str, Enum):
    default = "default"
    sigmoid_scaled = "sigmoid_scaled"


class ExperimentConfig(p.BaseModel):
    train_dataset: Dataset = Dataset.variable_length_sequences
    train_loss: Loss = Loss.default
    train_sequence_length: p.PositiveInt = 10
    train_max_sequence_length: p.PositiveInt = 10
    train_alpha: p.PositiveFloat = 1.0
    train_epoch_size: p.PositiveInt = 10000
    train_batch_size: p.PositiveInt = 64
    train_seed: int = 0
    validation_dataset: Dataset = Dataset.variable_length_sequences
    validation_loss: t.Optional[Loss] = None
    validation_sequence_length: p.PositiveInt = 10
    validation_max_sequence_length: p.PositiveInt = 50
    validation_alpha: p.PositiveFloat = 0.5
    validation_epoch_size: p.PositiveInt = 1000
    validation_batch_size: p.PositiveInt = 64
    validation_seed: int = 0
    min_validation_loss: float = 5e-2
    patience: int = 3
    model_seed: int = 1
    min_num_epochs: p.NonNegativeInt = 5
    max_num_epochs: p.PositiveInt = 30
    learning_rate: p.PositiveFloat = 1e-2
    linear_schedule_end_factor: p.PositiveFloat = 0.5
    test_max_sequence_length: p.PositiveInt = 200
    test_alpha: p.PositiveFloat = 0.1
    test_epoch_size: p.PositiveInt = 10000
    test_batch_size: p.PositiveInt = 256
    test_seed: int = 2

    @cached_property
    def validation_loss_(self) -> Loss:
        validation_loss = self.validation_loss
        if validation_loss is None:
            validation_loss = self.train_loss

        return validation_loss


class Config(p.BaseModel):
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataConfig = DataConfig()
    network: NetworkConfig = NetworkConfig()
    experiment: ExperimentConfig = ExperimentConfig()

    def __post_init__(self):
        tokenizer = Tokenizer(config=self.tokenizer)
        unsupported_characters = {c for c in self.data.positive_character_set if c not in tokenizer.token_to_token_id_}
        if len(unsupported_characters) > 0:
            raise ValueError(f"one or more unsupported characters\n{sorted(unsupported_characters) = }")


class TrainResult(p.BaseModel):
    train_time_taken: float
    train_losses: t.List[float]
    validation_losses: t.List[float]
    min_validation_loss_epoch: int
    min_validation_loss: float

    def __post_init__(self):
        if len(self.train_losses) != len(self.validation_losses):
            raise ValueError(
                textwrap.dedent(
                    f"""\
                    len(train_losses) != len(validation_losses)"
                    len(train_losses) = {len(self.train_losses)!r}
                    len(validation_losses) = {len(self.validation_losses)!r}"""
                )
            )

    @cached_property
    def num_epochs_(self):
        return len(self.train_losses) + 1


class TestResult(p.BaseModel):
    incorrect_strings: t.List[str]
    confusion_matrix: t.List[t.List[int]]


class EvalResult(p.BaseModel):
    string: str
    model_result: ModelResult
    i: int
    output_probability: float


class Experiment(p.BaseModel):
    directory: p.DirectoryPath
    verbose: bool = True

    @cached_property
    def config_path_(self):
        config_path = self.directory / "config.json"

        return config_path

    def init_config(self, *, force=False, overrides=None):
        if not force and self.config_path_.exists():
            return

        self.__dict__.pop("config_", None)

        config = Config()
        config.apply_overrides(overrides)
        config.dump_json(self.config_path_)

    @cached_property
    def config_(self):
        self.init_config()

        config = Config.parse_file(self.config_path_)

        return config

    def build_generator(self):
        generator = torch.Generator().manual_seed(self.config_.experiment.train_seed)

        return generator

    def collate_fn(self, items: t.List[Item]) -> Batch:
        batch = collate_items(self.config_.tokenizer, items)

        return batch

    @cached_property
    def train_result_path_(self):
        train_result_path = self.directory / "train_result.json"

        return train_result_path

    def train(self, *, force=False):
        if not force and self.train_result_path_.exists():
            return

        self.__dict__.pop("train_result_path_", None)

        config = self.config_

        with torch.random.fork_rng():
            torch.manual_seed(config.experiment.model_seed)

            model_config = ModelConfig(tokenizer=config.tokenizer, network=config.network)
            model = Model(model_config)
            model.post_init_reset_parameters()

        loss_fn = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.experiment.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config.experiment.linear_schedule_end_factor,
            total_iters=config.experiment.max_num_epochs,
        )

        torch.manual_seed(config.experiment.train_seed)

        train_data_loader = self._build_data_loader(
            dataset=config.experiment.train_dataset,
            sequence_length=config.experiment.train_sequence_length,
            max_sequence_length=config.experiment.train_max_sequence_length,
            alpha=config.experiment.train_alpha,
            epoch_size=config.experiment.train_epoch_size,
            batch_size=config.experiment.train_batch_size,
            seed=config.experiment.train_seed,
        )

        train_start_time = time.perf_counter()

        self.save_model(model, 0)

        train_losses = []
        validation_losses = []
        patience = config.experiment.patience
        min_validation_loss: t.Optional[float] = None
        for epoch in range(1, config.experiment.max_num_epochs + 1):
            model.train()

            train_loss = 0.0
            train_batch: Batch
            for train_batch in train_data_loader:
                model_result: ModelResult = model(train_batch.token_ids)

                target = train_batch.labels.to(model_result.output_logits.dtype)

                output_logits = model_result.output_logits
                if config.experiment.train_loss == "sigmoid_scaled":
                    output_logits = output_logits / (1 + torch.sigmoid(output_logits))
                    output_logits = -output_logits
                    output_logits = output_logits / (1 + torch.sigmoid(output_logits))
                    output_logits = -output_logits
                loss = loss_fn(output_logits, target)
                train_loss += len(train_batch.token_ids) * loss.item()

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            lr_scheduler.step()

            train_losses.append(train_loss)

            self.save_model(model, epoch)

            validation_data_loader = self._build_data_loader(
                dataset=config.experiment.validation_dataset,
                sequence_length=config.experiment.validation_sequence_length,
                max_sequence_length=config.experiment.validation_max_sequence_length,
                alpha=config.experiment.validation_alpha,
                epoch_size=config.experiment.validation_epoch_size,
                batch_size=config.experiment.validation_batch_size,
                seed=config.experiment.validation_seed,
            )

            model.eval()

            validation_loss = 0.0
            with torch.no_grad():
                validation_batch: Batch
                for validation_batch in validation_data_loader:
                    model_result: ModelResult = model(validation_batch.token_ids)

                    target = validation_batch.labels.to(model_result.output_logits.dtype)

                    output_logits = model_result.output_logits
                    if config.experiment.validation_loss_ == "sigmoid_scaled":
                        output_logits = output_logits / (1 + torch.sigmoid(output_logits))
                        output_logits = -output_logits
                        output_logits = output_logits / (1 + torch.sigmoid(output_logits))
                        output_logits = -output_logits
                    loss = loss_fn(output_logits, target)
                    validation_loss += len(validation_batch.token_ids) * loss.item()

            if self.verbose:
                print(f"[{epoch}] {train_loss = }, {validation_loss = }")

            validation_losses.append(validation_loss)

            if epoch < config.experiment.min_num_epochs:
                continue

            if validation_loss < config.experiment.min_validation_loss:
                break

            if min_validation_loss is None or validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                patience = config.experiment.patience
            else:
                patience -= 1
                if patience <= 0:
                    break

        train_end_time = time.perf_counter()
        train_time_taken = train_end_time - train_start_time

        assert train_losses
        assert validation_losses

        min_validation_loss_epoch, min_validation_loss_ = min(enumerate(validation_losses), key=lambda x: (x[1], x[0]))
        min_validation_loss_epoch += 1

        result = TrainResult(
            train_time_taken=train_time_taken,
            train_losses=train_losses,
            validation_losses=validation_losses,
            min_validation_loss_epoch=min_validation_loss_epoch,
            min_validation_loss=min_validation_loss_,
        )
        result.dump_json(self.train_result_path_)

    def _build_data_loader(
        self,
        *,
        dataset: Dataset,
        sequence_length: int,
        max_sequence_length: int,
        alpha: float,
        epoch_size: int,
        batch_size: int,
        seed: int,
    ) -> t.Union[VariableLengthSequences, DataLoader]:
        config = self.config_

        if dataset == "variable_length_sequences":
            data_loader = VariableLengthSequences(
                tokenizer_config=config.tokenizer,
                data_config=config.data,
                max_sequence_length=max_sequence_length,
                alpha=alpha,
                batch_size=batch_size,
                num_batches_per_epoch=epoch_size // batch_size,
                seed=seed,
            )
        else:
            assert dataset == "all_fixed_length_sequences"

            dataset_ = AllFixedLengthSequences(
                tokenizer_config=config.tokenizer,
                data_config=config.data,
                sequence_length=sequence_length,
            )
            data_loader = DataLoader(
                dataset_,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                generator=self.build_generator(),
            )

        return data_loader

    @cached_property
    def train_result_(self):
        self.train()

        train_result = TrainResult.parse_file(self.train_result_path_)

        return train_result

    @cached_property
    def test_result_path_(self):
        test_result_path = self.directory / "test_result.json"

        return test_result_path

    def test(self, *, force=False):
        if not force and self.test_result_path_.exists():
            return

        self.__dict__.pop("test_result_path_", None)

        config = self.config_

        train_result = self.train_result_
        model = self.load_model(train_result.min_validation_loss_epoch)

        test_data_loader = VariableLengthSequences(
            tokenizer_config=config.tokenizer,
            data_config=config.data,
            max_sequence_length=config.experiment.test_max_sequence_length,
            alpha=config.experiment.test_alpha,
            batch_size=config.experiment.test_batch_size,
            num_batches_per_epoch=config.experiment.test_epoch_size // config.experiment.test_batch_size,
            seed=config.experiment.test_seed,
        )
        test_incorrect_strings, test_confusion_matrix = eval_metrics(self.tokenizer_, model, test_data_loader)

        result = TestResult(
            incorrect_strings=test_incorrect_strings,
            confusion_matrix=test_confusion_matrix.tolist(),
        )
        result.dump_json(self.test_result_path_)

    @cached_property
    def test_result_(self):
        self.test()

        test_result = TestResult.parse_file(self.test_result_path_)

        return test_result

    def eval_(self, strings: t.List[str], *, model: t.Optional[Model] = None, epoch: t.Optional[int] = None):
        token_ids = []
        for i, string in enumerate(strings):
            try:
                token_ids_ = self.tokenizer_.encode(string)
            except ValueError as e:
                raise ValueError(
                    textwrap.dedent(
                        f"""\
                        unable to tokenize input string
                        {i = }
                        {strings[i] = }"""
                    )
                ) from e

            token_ids.append(token_ids_)

        token_ids = validate_token_ids(self.config_.tokenizer, token_ids)
        num_sequences = token_ids.shape[0]

        if model is None:
            model = self.load_model(epoch)

        with torch.no_grad():
            model_result: ModelResult = model(token_ids)

        assert model_result.output_logits.shape == (num_sequences,)

        output_probabilities = torch.sigmoid(model_result.output_logits)

        results = []
        for i, (string, output_probability) in enumerate(zip(strings, output_probabilities.tolist())):
            result = EvalResult(
                string=string,
                i=i,
                model_result=model_result,
                output_probability=output_probability,
            )
            results.append(result)

        return results

    @cached_property
    def tokenizer_(self):
        tokenizer = Tokenizer(config=self.config_.tokenizer)

        return tokenizer

    def save_model(self, model, epoch):
        state_dict = model.state_dict()

        model_path = self.models_directory_ / f"{epoch}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(state_dict, model_path)

    def load_model(self, epoch: t.Optional[int] = None):
        if epoch is None:
            epoch = self.train_result_.min_validation_loss_epoch

        model_path = self.models_directory_ / f"{epoch}.pt"
        state_dict = torch.load(model_path)

        config = self.config_
        model_config = ModelConfig(tokenizer=config.tokenizer, network=config.network)
        model = Model(model_config)
        model.load_state_dict(state_dict)

        model.eval()

        return model

    @cached_property
    def models_directory_(self):
        models_directory = self.directory / "models"

        return models_directory


def eval_metrics(tokenizer: Tokenizer, model: Model, data_loader):
    labels = []
    predictions = []
    incorrect_strings = []
    with torch.no_grad():
        batch: Batch
        for batch in data_loader:
            labels.append(batch.labels)

            model_result: ModelResult = model(batch.token_ids)
            predicted_labels = torch.as_tensor(model_result.output_logits > 0.0, dtype=batch.labels.dtype)
            predictions.append(predicted_labels)

            incorrect_indices = torch.nonzero(predicted_labels != batch.labels).ravel()
            for i in incorrect_indices.numpy():
                token_ids: t.List[int] = batch.token_ids[i].tolist()
                tokens = tokenizer.decode(token_ids)

                token_counts = Counter(tokens)
                fragments = []
                for token, count in sorted(token_counts.items()):
                    fragment = f"{token}{{{count}}}" if count > 1 else token
                    fragments.append(fragment)

                string = "".join(fragments)
                incorrect_strings.append(string)

    incorrect_strings.sort()

    labels = torch.hstack(labels)
    predictions = torch.hstack(predictions)
    confusion_matrix = metrics.confusion_matrix(labels, predictions, labels=[0, 1])

    return incorrect_strings, confusion_matrix
