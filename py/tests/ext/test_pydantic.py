##########################################
# File: test_pydantic.py                 #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import sqlite3
import textwrap
import typing as t
from contextlib import closing
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

assert p.Field is pydantic.Field
assert p.validator is pydantic.validator
assert p.ValidationError is pydantic.ValidationError
assert p.validate_arguments is pydantic.validate_arguments
assert p.FilePath is pydantic.FilePath
assert p.DirectoryPath is pydantic.DirectoryPath

del pydantic


class Foo(p.BaseModel):
    class Bar(p.BaseModel):
        d: float = 1.0

    class Baz(p.BaseModel):
        x: int = 2

        @cached_property
        def y_(self):
            return self.x + 1

    a: int = 0
    b: t.List[p.NonNegativeInt] = [0]
    c: t.Optional[Bar] = Bar()
    d: t.Optional[t.List[Bar]] = None

    # `p.FilePath` should normally be used in place of `t.Union[str, Path]` so that the field is
    # always parsed to a `Path`. But, this example is retained to test the expected handling of
    # union fields.
    e: t.Union[str, Path] = ""
    f: t.Optional[t.Tuple[str, Bar, Baz]] = None

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

    # Fields with mutable default values (e.g. `b` or `c`) are copied and not
    # shared between instances.
    # (This is just a sanity check.)
    foo = Foo()
    assert foo.json() == textwrap.dedent(
        """\
        {
          "a": 0,
          "b": [
            0
          ],
          "c": {
            "d": 1.0
          },
          "e": ""
        }"""
    )

    mutated_foo = Foo()
    assert mutated_foo.b is not foo.b
    assert mutated_foo.c is not foo.c

    mutated_foo.b.append(1)
    assert mutated_foo.c is not None
    mutated_foo.c.d = 2.0

    assert mutated_foo.json() == textwrap.dedent(
        """\
        {
          "a": 0,
          "b": [
            0,
            1
          ],
          "c": {
            "d": 2.0
          },
          "e": ""
        }""",
    )
    assert foo.json() == textwrap.dedent(
        """\
        {
          "a": 0,
          "b": [
            0
          ],
          "c": {
            "d": 1.0
          },
          "e": ""
        }""",
    )

    # Cached properties do *not* impact equality (because of the explicit setting of
    # `__include_fields__` in `ModelMetaclass`.
    other_foo = Foo()
    assert repr(other_foo) == "Foo(a=0, b=[0], c=Bar(d=1.0), d=None, e='', f=None)"
    assert other_foo == foo

    other_foo.b_
    assert repr(other_foo) == "Foo(a=0, b=[0], c=Bar(d=1.0), d=None, e='', f=None, b_=1)"
    assert other_foo == foo

    # Unions of types that are *not* `BaseModel`s are supported.
    foo = Foo(e=Path("z.txt"))
    other_foo = Foo(e="z.txt")
    assert foo != other_foo
    assert foo.json() == other_foo.json()

    with pytest.raises(p.ValidationError) as exc_info:
        Foo(e=1)  # pyright: ignore[reportGeneralTypeIssues]
    assert str(exc_info.value) == textwrap.dedent(
        """\
        2 validation errors for Foo
        e
          str type expected (type=type_error.str)
        e
          value is not a valid path (type=type_error.path)"""
    )

    # Tuples are supported too. (Cached properties are accessed to ensure
    # they do *not*
    baz = Foo.Baz(x=3)
    baz.y_
    foo = Foo(f=("A", Foo.Bar(d=3.0), baz))
    foo.b_
    assert repr(foo) == "Foo(a=0, b=[0], c=Bar(d=1.0), d=None, e='', f=('A', Bar(d=3.0), Baz(x=3, y_=4)), b_=1)"
    assert foo.json() == textwrap.dedent(
        """\
        {
          "a": 0,
          "b": [
            0
          ],
          "c": {
            "d": 1.0
          },
          "e": "",
          "f": [
            "A",
            {
              "d": 3.0
            },
            {
              "x": 3
            }
          ]
        }"""
    )

    # The `BaseModel`s contained in inputs are *not* shallow or deep copied, but
    # their containers (e.g. a `list`) are (see `other_foo.d is not foo.d` below).
    c = Foo.Bar(d=3.0)
    d = [Foo.Bar(d=4.0)]
    foo = Foo(c=c, d=d)
    other_foo = Foo(c=c, d=d)

    assert other_foo.c is foo.c

    assert other_foo.d is not foo.d
    assert other_foo.d is not None and foo.d is not None and other_foo.d == foo.d
    assert all(other_d is d for other_d, d in zip(other_foo.d, foo.d))


def test_base_model_apply_overrides():
    # `apply_model_overrides` is supported via the `apply_overrides` method.
    foo = Foo()
    foo.apply_overrides()
    assert foo.json() == textwrap.dedent(
        """\
        {
          "a": 0,
          "b": [
            0
          ],
          "c": {
            "d": 1.0
          },
          "e": ""
        }"""
    )

    foo.apply_overrides([(["a"], 1), (["c", "d"], 3.0)])
    assert foo.json() == textwrap.dedent(
        """\
        {
          "a": 1,
          "b": [
            0
          ],
          "c": {
            "d": 3.0
          },
          "e": ""
        }"""
    )

    # A `ValueError` is raised if an empty path is provided ...
    with pytest.raises(ValueError) as exc_info:
        foo.apply_overrides([([], 0)])
    assert str(exc_info.value) == textwrap.dedent(
        """\
        empty path
        model = Foo(a=1, b=[0], c=Bar(d=3.0), d=None, e='', f=None)
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
    foo = Foo(d=[Foo.Bar(d=2.0), Foo.Bar(d=3.0)], e=Path("z.txt"))
    assert repr(foo) == "Foo(a=0, b=[0], c=Bar(d=1.0), d=[Bar(d=2.0), Bar(d=3.0)], e=PosixPath('z.txt'), f=None)"

    foo.b_
    assert repr(foo) == "Foo(a=0, b=[0], c=Bar(d=1.0), d=[Bar(d=2.0), Bar(d=3.0)], e=PosixPath('z.txt'), f=None, b_=1)"

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
              },
              "d": [
                {
                  "d": 2.0
                },
                {
                  "d": 3.0
                }
              ],
              "e": "z.txt"
            }
            """
        )

        parsed_foo = Foo.parse_file(path_or_str)

        # Fields that are a union like `e`:
        #
        #   e: t.Union[str, Path] = ""
        #
        # deserialize to the first type (`str`).
        #
        # This means `parsed_foo` is *not* equal to `foo` but is after `e` is converted to a `Path`.
        assert repr(parsed_foo) == "Foo(a=0, b=[0], c=Bar(d=1.0), d=[Bar(d=2.0), Bar(d=3.0)], e='z.txt', f=None)"

        assert parsed_foo != foo
        parsed_foo.e = Path(parsed_foo.e)
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
    assert tensors_json == textwrap.dedent(
        """\
        {
          "t": {
            "dtype": "torch.float32",
            "shape": [
              1
            ],
            "data": [
              1.0
            ]
          }
        }"""
    )

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
    assert other_tensors_json == textwrap.dedent(
        """\
        {
          "t": {
            "dtype": "torch.bool",
            "shape": [
              1,
              1,
              0,
              1,
              1
            ],
            "data": [
              [
                []
              ]
            ]
          }
        }"""
    )

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
    assert enums.json() == textwrap.dedent(
        """\
        {
          "e": "x"
        }"""
    )

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


def test_arbitrary_types_model(temp_dir: Path):
    with pytest.raises(RuntimeError) as exc_info:

        class _(p.BaseModel):
            conn: sqlite3.Connection

    assert exc_info.value.args == (
        "no validator found for <class 'sqlite3.Connection'>, see `arbitrary_types_allowed` in Config",
    )

    class Qux(p.ArbitraryTypesModel):
        conn: sqlite3.Connection

    path = temp_dir / "db"
    with closing(sqlite3.connect(str(path))) as conn:
        Qux(conn=conn)
