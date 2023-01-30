##########################################
# File: pydantic.py                      #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import argparse
import re
import typing as t
from functools import cached_property
from pathlib import Path

import pydantic as _
import pydantic.fields as pf
import pydantic.main as pm
import pydantic.validators as pv
import torch
from typing_extensions import dataclass_transform

from .torch import as_dtype, as_size


class TorchTensorError(_.PydanticValueError):
    code = "torch_tensor_instance"
    msg_template = "{value} is not a torch.Tensor"


def torch_tensor_validator(v: t.Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v

    tensor = _maybe_decode_torch_tensor(v)
    if tensor is not None:
        return tensor

    raise TorchTensorError(value=v)


_STRICT_VALIDTORS = {
    pv.str_validator: pv.strict_str_validator,
    pv.bytes_validator: pv.strict_bytes_validator,
    pv.int_validator: pv.strict_int_validator,
    pv.float_validator: pv.strict_float_validator,
}


def _use_strict_validators(original_validators):
    validators_ = []
    for val_type, validators in original_validators:
        validators = [_STRICT_VALIDTORS.get(v, v) if not isinstance(v, pv.IfConfig) else v for v in validators]
        validators_.append((val_type, validators))

    validators_.append((torch.Tensor, [torch_tensor_validator]))

    return validators_


pv._VALIDATORS[:] = _use_strict_validators(pv._VALIDATORS)


if t.TYPE_CHECKING:
    PositiveInt = int
    NegativeInt = int
    NonPositiveInt = int
    NonNegativeInt = int
    PositiveFloat = float
    NegativeFloat = float
    NonPositiveFloat = float
    NonNegativeFloat = float
    FiniteFloat = float
else:

    class PositiveInt(_.PositiveInt):
        strict = True

    class NegativeInt(_.NegativeInt):
        strict = True

    class NonPositiveInt(_.NonPositiveInt):
        strict = True

    class NonNegativeInt(_.NonNegativeInt):
        strict = True

    StrictInt = _.StrictInt

    class PositiveFloat(_.PositiveFloat):
        strict = True

    class NegativeFloat(_.NegativeFloat):
        strict = True

    class NonPositiveFloat(_.NonPositiveFloat):
        strict = True

    class NonNegativeFloat(_.NonNegativeFloat):
        strict = True

    StrictFloat = _.StrictFloat

    class FiniteFloat(_.FiniteFloat):
        strict = True


Field = _.Field

validator = _.validator
ValidationError = _.ValidationError

validate_arguments = _.validate_arguments

FilePath = _.FilePath
DirectoryPath = _.DirectoryPath

_ModelPath = t.Tuple[str, ...]

_OverridePath = t.Sequence[str]
_OverrideValue = t.Any
_Override = t.Tuple[_OverridePath, _OverrideValue]
_Overrides = t.Optional[t.List[_Override]]


# The `dataclass_transform` decorator is required for type checking of arguments (among other things).
@dataclass_transform(kw_only_default=True, field_descriptors=(pf.Field, pf.FieldInfo))
class ModelMetaclass(pm.ModelMetaclass):
    def __new__(mcs, name, bases, namespace, **kwargs):  # pyright: ignore[reportSelfClsParameterName]
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if cls.__fields__:
            include_fields = cls.__include_fields__
            if include_fields is None:
                include_fields = {}

            name_: str
            field: pf.ModelField
            for name_, field in cls.__fields__.items():
                if name_ not in include_fields:
                    assert field.field_info.include is None

                    include = _default_include_fields(field.annotation)
                    include_fields[name_] = include

            if include_fields:
                cls.__include_fields__ = include_fields

        return cls


NoneType = type(None)


_DEFAULT_INCLUDE_FIELDS = {}


def _default_include_fields(cls):
    origin = t.get_origin(cls)
    if origin is t.Union:
        args = t.get_args(cls)
        if args is None:
            raise NotImplementedError(f"unsupported empty args\n{cls = }")

        try:
            i = args.index(NoneType)
        except ValueError:
            pass
        else:
            args = args[:i] + args[i + 1 :]

        include_fields_ = [_default_include_fields(arg) for arg in args]

        if len(include_fields_) > 1:
            if all(x == include_fields_[0] for x in include_fields_[1:]):
                include_fields_ = include_fields_[:1]

        if len(include_fields_) != 1:
            raise NotImplementedError(f"unsupported union annotation\n{cls = }\n{args = }")

        include_fields = include_fields_[0]

        return include_fields

    if not _issubclass_base_model(cls):
        return {}

    include_fields = _DEFAULT_INCLUDE_FIELDS.get(cls)
    if include_fields is None:
        include_fields = {}
        for name, field in cls.__fields__.items():
            include = _default_include_fields(field.annotation)
            include_fields[name] = include

        _DEFAULT_INCLUDE_FIELDS[cls] = include_fields

    return include_fields


def _issubclass_base_model(maybe_cls):
    b = False
    try:
        b = issubclass(maybe_cls, BaseModel)
    except TypeError:
        pass

    return b


def _torch_tensor_json_encoder(obj: torch.Tensor):
    data = obj.tolist()

    any_obj = {
        "dtype": str(obj.dtype),
        "shape": tuple(obj.shape),
        "data": data,
    }

    return any_obj


def _maybe_decode_torch_tensor(v: t.Any) -> t.Optional[torch.Tensor]:
    if not isinstance(v, dict):
        return None

    dtype_str = v.get("dtype")
    if dtype_str is None:
        return None

    dtype = as_dtype(dtype_str)

    shape = v.get("shape")
    if shape is None:
        return None

    shape = as_size(shape)

    data = v.get("data")
    if data is None:
        return None

    tensor = torch.tensor(data, dtype=dtype).reshape(shape)

    return tensor


class BaseModel(_.BaseModel, metaclass=ModelMetaclass):
    class Config:
        # Reference:
        # https://docs.pydantic.dev/usage/model_config/
        extra = "forbid"

        keep_untouched = (cached_property,)

        validate_assignment = True
        validate_all = True

        # Models are *not* copied on validation. This avoids bugs related
        # to lifecycle management which rely on models having one or more
        # references.
        #
        # Reference:
        # https://github.com/pydantic/pydantic/pull/4093
        copy_on_model_validation = "none"

        json_encoders = {
            torch.Tensor: _torch_tensor_json_encoder,
        }

    def __init__(self, **data: t.Any) -> None:
        super().__init__(**data)

        self.__post_init__()

    def __post_init__(self):
        pass

    def apply_overrides(self, overrides: _Overrides = None):
        apply_model_overrides(self, overrides)

    def json(self, *, indent: int = 2, exclude_none: bool = True, include=None, **kwargs) -> str:
        s = super().json(indent=indent, exclude_none=exclude_none, include=include, **kwargs)

        return s

    def dump_json(self, path: t.Union[str, Path], **kwargs):
        dump_json(self, path, **kwargs)


def dump_json(model: _.BaseModel, path: t.Union[str, Path], *, indent: int = 2, exclude_none: bool = True, **kwargs):
    with Path(path).open("w") as f:
        f.write(model.json(indent=indent, exclude_none=exclude_none, **kwargs))
        f.write("\n")


def apply_model_overrides(model: _.BaseModel, overrides: _Overrides = None):
    if not overrides:
        return

    for path, value in overrides:
        if not path:
            raise ValueError(f"empty path\n{model = }\n{overrides = }")

        obj = get_model_value(model, path[:-1])

        setattr(obj, path[-1], value)


def get_model_value(model: _.BaseModel, path: t.Iterable[str]) -> t.Any:
    obj = model
    for name in path:
        obj = getattr(obj, name)

    return obj


def get_model_paths_shortest_suffixes(
    *, model_class=None, model_schema=None, paths: t.Optional[t.List[_ModelPath]] = None
):
    if paths is None:
        paths = get_model_paths(model_class=model_class, model_schema=model_schema)

    if not paths:
        return []

    max_path_length = max(len(path) for path in paths)

    items: t.List[t.Tuple[_ModelPath, t.Tuple[int, _ModelPath]]] = []
    for i, path in enumerate(paths):
        padded = path
        if (n := max_path_length - len(path)) > 0:
            padded = ("",) * n + padded

        items.append((padded[::-1], (i, path)))

    items.sort()

    indexed_paths_shortest_suffixes: t.List[t.Tuple[int, t.Tuple[_ModelPath, _ModelPath]]] = []
    invalid_prefixes = set()

    for i, (reversed_path, (original_i, path)) in enumerate(items):
        next_reversed_path, _ = items[j] if (j := i + 1) < len(items) else (None, None)

        n = 1
        for n in range(1, max_path_length + 1):
            prefix = reversed_path[:n]
            if prefix in invalid_prefixes:
                continue

            if next_reversed_path is None:
                break

            next_prefix = next_reversed_path[:n]
            if prefix != next_prefix:
                break

            invalid_prefixes.add(prefix)
        else:
            raise ValueError(f"duplicate reversed_path\n{reversed_path = }")

        indexed_paths_shortest_suffixes.append((original_i, (path, path[-n:])))

    indexed_paths_shortest_suffixes.sort()

    paths_shortest_suffixes = [x[1] for x in indexed_paths_shortest_suffixes]

    return paths_shortest_suffixes


def get_model_paths(
    *,
    model_class=None,
    model_schema=None,
    prefix: _ModelPath = (),
    definitions=None,
    model_paths: t.Optional[t.List[_ModelPath]] = None,
):
    model_schema = validate_model_schema(model_class, model_schema)

    if model_paths is None:
        model_paths = []

    for name, schema in model_schema["properties"].items():
        if (ref := maybe_get_ref(schema)) is not None:
            definition = parse_ref(ref)
            if definitions is None:
                definitions = model_schema["definitions"]
            next_model_schema = definitions[definition]

            if next_model_schema["type"] == "object":
                next_prefix = prefix + (name,)
                get_model_paths(
                    model_schema=next_model_schema, prefix=next_prefix, definitions=definitions, model_paths=model_paths
                )
                continue

        fragments = list(prefix)
        fragments.append(name)
        model_path = tuple(fragments)
        model_paths.append(model_path)

    return model_paths


def add_model_arguments(
    parser: argparse.ArgumentParser, *, model_class=None, model_schema=None, prefix=(), definitions=None
):
    model_schema = validate_model_schema(model_class, model_schema)

    for name, schema in model_schema["properties"].items():
        if (ref := maybe_get_ref(schema)) is not None:
            definition = parse_ref(ref)
            if definitions is None:
                definitions = model_schema["definitions"]
            next_model_schema = definitions[definition]

            if next_model_schema["type"] == "object":
                next_prefix = prefix + (name,)
                add_model_arguments(parser, model_schema=next_model_schema, prefix=next_prefix, definitions=definitions)
                continue

            default = schema.get("default")
            schema = next_model_schema
        else:
            default = schema.get("default")

        kwargs = {}
        type_ = None
        include_type_kwarg = True
        schema_type = schema["type"]
        if schema_type == "integer":
            type_ = int
        elif schema_type == "boolean":
            type_ = bool
            include_type_kwarg = False
            if default is None or default is False:
                kwargs["action"] = "store_true"
            elif default is True:
                raise ValueError(f"bool field with default True not supported\n{name = }\n{schema = }")
            else:
                raise ValueError(f"unable to handle default for bool\n{name = }\n{schema = }")
        elif schema_type == "number":
            type_ = float
        elif schema_type == "string":
            type_ = str
            if enum := schema.get("enum"):
                kwargs["choices"] = enum
        elif schema_type == "array":
            kwargs["nargs"] = "+"
            if schema["items"]["type"] == "integer":
                type_ = int
            elif schema["items"]["type"] == "string":
                type_ = str

        if type_ is None:
            raise ValueError(f"unable to determine type for property\n{name = }\n{schema = }")

        # (The `type` keyword argument is *not* included, for example, when action is `"store_true"`.)
        if include_type_kwarg:
            kwargs["type"] = type_

        fragments = list(prefix)
        fragments.append(name)
        argument_name = f"--{'-'.join(fragments).replace('_', '-')}"

        help_ = f"(default: {default!r})"

        parser.add_argument(argument_name, help=help_, **kwargs)


def get_model_overrides(
    args: argparse.Namespace, *, model_class=None, model_schema=None, prefix=(), definitions=None, overrides=None
):
    model_schema = validate_model_schema(model_class, model_schema)

    if overrides is None:
        overrides = []

    for name, schema in model_schema["properties"].items():
        if (ref := maybe_get_ref(schema)) is not None:
            definition = parse_ref(ref)
            if definitions is None:
                definitions = model_schema["definitions"]
            next_model_schema = definitions[definition]

            if next_model_schema["type"] == "object":
                next_prefix = prefix + (name,)
                get_model_overrides(
                    args,
                    model_schema=next_model_schema,
                    prefix=next_prefix,
                    definitions=definitions,
                    overrides=overrides,
                )
                continue

        fragments = list(prefix)
        fragments.append(name)
        args_name = "_".join(fragments)

        if (value := getattr(args, args_name)) is not None:
            overrides.append((fragments, value))

    return overrides


def validate_model_schema(model_class, model_schema):
    if (model_class is None) == (model_schema is None):
        raise TypeError("one and only one of model_class or model_schema required")

    if model_schema is None:
        assert model_class is not None
        model_schema = model_class.schema()

    return model_schema


PARSE_REF_RE = re.compile("^\\#/definitions/(.+)")


def parse_ref(ref):
    m = PARSE_REF_RE.match(ref)
    if m is None:
        raise ValueError(f"unable to parse ref\n{ref = }")

    (definition,) = m.groups()

    return definition


def maybe_get_ref(schema):
    ref = schema.get("$ref")
    if ref is None:
        if (all_of := schema.get("allOf")) is not None:
            if len(all_of) != 1:
                raise ValueError(f"len(all_of) != 1\n{len(all_of) = }\n{all_of = }")

            ref = all_of[0]["$ref"]

    return ref


class ArbitraryTypesModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
