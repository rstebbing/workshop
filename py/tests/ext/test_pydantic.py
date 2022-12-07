##########################################
# File: test_pydantic.py                 #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import textwrap
import typing as t
from enum import Enum
from functools import cached_property
from pathlib import Path

import pydantic
import pytest
import torch

from workshop.ext import pydantic as p


# The `pydantic` module should be accessible as `_` in `p`. Common functions and/or classes should
# be available on `p` directly even if they are not extensions.
assert p._ is pydantic

assert p.validator is pydantic.validator
assert p.ValidationError is pydantic.ValidationError
assert p.FilePath is pydantic.FilePath
assert p.DirectoryPath is pydantic.DirectoryPath

del pydantic


class Foo(p.BaseModel):
    class Bar(p.BaseModel):
        d: float = 1.0

    a: int = 0
    b: t.List[p.NonNegativeInt] = [0]
    c: Bar = Bar()

    @cached_property
    def b_(self):
        return self.a + 1

    def __post_init__(self):
        if self.b != [self.a]:
            raise ValueError(f"b != [a]\nb = {self.b!r}\na = {self.a!r}")


def test_base_model():
    # Extra fields should be forbidden (instead of allowed) by default.
    assert Foo.Config.extra == "forbid"

    with pytest.raises(p.ValidationError) as exc_info:
        Foo(z=1)  # pyright: ignore[reportGeneralTypeIssues]
    assert str(exc_info.value) == textwrap.dedent(
        """\
        1 validation error for Foo
        z
          extra fields not permitted (type=value_error.extra)"""
    )

    # With this, `Foo.__new__` fails due to a deep copy of `cached_property` *not* being possible.
    assert cached_property in Foo.Config.keep_untouched

    # Validation should be done on assignment (instead of skipped) by default. Furthermore, `int`
    # should be validated strictly (and not coerced). This is true even for subclasses of
    # `ConstrainedInt` like `p.NonNegativeInt`.
    foo = Foo()
    with pytest.raises(p.ValidationError) as exc_info:
        foo.a = 1.5  # pyright: ignore[reportGeneralTypeIssues]
    assert str(exc_info.value) == textwrap.dedent(
        """\
        1 validation error for Foo
        a
          value is not a valid integer (type=type_error.integer)"""
    )

    with pytest.raises(p.ValidationError) as exc_info:
        foo.b = [0, 1, 0.5]  # pyright: ignore[reportGeneralTypeIssues]
    assert str(exc_info.value) == textwrap.dedent(
        """\
        1 validation error for Foo
        b -> 2
          value is not a valid integer (type=type_error.integer)"""
    )

    class InvalidFoo(Foo):
        c: float = "A"  # pyright: ignore[reportGeneralTypeIssues]

    # Default values should be validated too (instead of skipped) by default.
    with pytest.raises(p.ValidationError) as exc_info:
        InvalidFoo()
    assert str(exc_info.value) == textwrap.dedent(
        """\
        1 validation error for InvalidFoo
        c
          value is not a valid float (type=type_error.float)"""
    )

    # `__post_init__` should be invoked by the default `__init__`.
    with pytest.raises(ValueError) as exc_info:
        Foo(b=[1])
    assert str(exc_info.value) == textwrap.dedent(
        """\
        b != [a]
        b = [1]
        a = 0"""
    )

    # Fields with mutable default values (e.g. `b`) are copied and not shared between instances.
    # (This is just a sanity check.)
    foo = Foo()
    assert foo.json() == '{"a": 0, "b": [0], "c": {"d": 1.0}}'

    mutated_foo = Foo()
    mutated_foo.b.append(1)
    assert mutated_foo.json() == '{"a": 0, "b": [0, 1], "c": {"d": 1.0}}'
    assert foo.json() == '{"a": 0, "b": [0], "c": {"d": 1.0}}'

    # Cached properties do *not* impact equality (because of the explicit setting of
    # `__include_fields__` in `ModelMetaclass`.
    other_foo = Foo()
    assert repr(other_foo) == "Foo(a=0, b=[0], c=Bar(d=1.0))"
    assert other_foo == foo

    other_foo.b_
    assert repr(other_foo) == "Foo(a=0, b=[0], c=Bar(d=1.0), b_=1)"
    assert other_foo == foo


def test_base_model_apply_overrides():
    # `apply_model_overrides` is supported via the `apply_overrides` method.
    foo = Foo()
    foo.apply_overrides()
    assert foo.json() == '{"a": 0, "b": [0], "c": {"d": 1.0}}'

    foo.apply_overrides([(["a"], 1), (["c", "d"], 3.0)])
    assert foo.json() == '{"a": 1, "b": [0], "c": {"d": 3.0}}'

    # A `ValueError` is raised if an empty path is provided ...
    with pytest.raises(ValueError) as exc_info:
        foo.apply_overrides([([], 0)])
    assert str(exc_info.value) == textwrap.dedent(
        """\
        empty path
        model = Foo(a=1, b=[0], c=Bar(d=3.0))
        overrides = [([], 0)]"""
    )

    # ... or an invalid field is specified (via `pydantic`).
    with pytest.raises(ValueError) as exc_info:
        foo.apply_overrides([(["x"], 0)])
    assert str(exc_info.value) == '"Foo" object has no field "x"'

    # (Strict) validation is still done too.
    with pytest.raises(p.ValidationError) as exc_info:
        foo.apply_overrides([(["c", "d"], 1)])
    assert str(exc_info.value) == textwrap.dedent(
        """\
        1 validation error for Bar
        d
          value is not a valid float (type=type_error.float)"""
    )


def test_base_model_dump_json(temp_dir: Path):
    foo = Foo()
    assert repr(foo) == "Foo(a=0, b=[0], c=Bar(d=1.0))"

    foo.b_
    assert repr(foo) == "Foo(a=0, b=[0], c=Bar(d=1.0), b_=1)"

    for path_or_str in [temp_dir / "x.json", str(temp_dir / "y.json")]:
        p = Path(path_or_str)
        assert not p.exists()

        # A `Path` or `str` is OK, and in either case, the cached property should *not* be serialized.
        foo.dump_json(path_or_str)

        assert p.read_text() == textwrap.dedent(
            """\
            {
              "a": 0,
              "b": [
                0
              ],
              "c": {
                "d": 1.0
              }
            }
            """
        )

        parsed_foo = Foo.parse_file(path_or_str)
        assert repr(parsed_foo) == "Foo(a=0, b=[0], c=Bar(d=1.0))"
        assert parsed_foo == foo


def test_base_model_torch_tensor():
    class Tensors(p.BaseModel):
        t: torch.Tensor

    tensors = Tensors(t=torch.tensor([1.0]))
    assert repr(tensors) == "Tensors(t=tensor([1.]))"

    with pytest.raises(p.ValidationError) as exc_info:
        Tensors(t=1)  # pyright: ignore[reportGeneralTypeIssues]
    assert str(exc_info.value) == textwrap.dedent(
        """\
        1 validation error for Tensors
        t
          1 is not a torch.Tensor (type=value_error.torch_tensor_instance; value=1)"""
    )

    # A `torch.Tensor` can be serialized to JSON ...
    tensors_json = tensors.json()
    assert tensors_json == '{"t": {"dtype": "torch.float32", "shape": [1], "data": [1.0]}}'

    # ... and deserialized too because the data type and shape are persisted too. This is done
    # because `torch.Tensor` as a type annotation says nothing about either.
    parsed_tensors = Tensors.parse_raw(tensors_json)
    assert parsed_tensors.t.dtype == tensors.t.dtype
    assert parsed_tensors.t.shape == tensors.t.shape
    assert (parsed_tensors.t == tensors.t).all()

    # The shape is necessary to persist because `data` does *not* contain all information.
    # For example, when there is one or more dimensions of size zero.
    other_tensors = Tensors(t=torch.tensor([], dtype=torch.bool).reshape(1, 1, 0, 1, 1))
    other_tensors_json = other_tensors.json()
    assert other_tensors_json == '{"t": {"dtype": "torch.bool", "shape": [1, 1, 0, 1, 1], "data": [[[]]]}}'

    parsed_other_tensors = Tensors.parse_raw(other_tensors_json)
    assert parsed_other_tensors.t.dtype == other_tensors.t.dtype
    assert parsed_other_tensors.t.shape == other_tensors.t.shape
    assert (parsed_other_tensors.t == other_tensors.t).all()


def test_base_model_enum():
    class E(str, Enum):
        x = "x"
        y = "y"

    class Enums(p.BaseModel):
        e: E = E.x

    enums = Enums()
    assert type(enums.e) is E
    assert enums.e == "x"
    assert enums.json() == '{"e": "x"}'

    # `Enum`s are JSON serializable, so `dict()` does *not* conver them to their values.
    #
    # Reference:
    # https://stackoverflow.com/a/65211727
    assert repr(enums.dict()) == "{'e': <E.x: 'x'>}"

    enums = Enums.parse_obj({"e": "y"})
    assert type(enums.e) is E
    assert enums.e == "y"
    assert enums.e == E.y


def test_get_model_paths_shortest_suffixes():
    class Empty(p.BaseModel):
        pass

    paths_shortest_suffixes = p.get_model_paths_shortest_suffixes(model_class=Empty)
    assert paths_shortest_suffixes == []

    class Foo(p.BaseModel):
        class Bar(p.BaseModel):
            class Baz(p.BaseModel):
                a: int
                d: int

            a: int = 1
            b: Baz = Baz(a=2, d=20)
            c: Baz = Baz(a=3, d=30)
            e: int = 4

        a: int = 0
        b: Bar = Bar()

    paths_shortest_suffixes = p.get_model_paths_shortest_suffixes(model_class=Foo)
    assert [
        (("a",), ("a",)),
        (("b", "a"), ("b", "a")),
        (("b", "b", "a"), ("b", "b", "a")),
        (("b", "b", "d"), ("b", "d")),
        (("b", "c", "a"), ("c", "a")),
        (("b", "c", "d"), ("c", "d")),
        (("b", "e"), ("e",)),
    ]

    with pytest.raises(ValueError) as exc_info:
        p.get_model_paths_shortest_suffixes(model_class=Foo, paths=[("a",), ("a",)])
    assert str(exc_info.value) == textwrap.dedent(
        """\
        duplicate reversed_path
        reversed_path = ('a',)"""
    )
