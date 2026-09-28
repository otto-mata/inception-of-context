import re
import sys
from typing import Any, TypeVar

import clang.cindex

T = TypeVar("T")


def force_as(o: Any, cls: T) -> T:
    assert isinstance(o, cls)  # pyright: ignore[reportArgumentType]
    return o


index = clang.cindex.Index.create()
tu = index.parse(sys.argv[1])
with open(sys.argv[1], "rb") as f:  # bytes: clang offsets are byte offsets
    src = f.read()
    f.seek(0)
    lines = f.readlines()


def function_def(cur: clang.cindex.Cursor):
    body = next(
        (
            c
            for c in cur.get_children()
            if c.kind == clang.cindex.CursorKind.COMPOUND_STMT
        ),
        None,
    )
    if body is None:
        ext = force_as(cur.extent, clang.cindex.SourceRange)
        source = src[ext.start.offset : ext.end.offset + 1].decode()
        return re.sub(r" {2,}", r" ", source).replace("\n", "")

    start, end = cur.location.offset, body.extent.end.offset
    print(
        cur.kind.name,
        cur.spelling,
        cur.location.line,
    )
    return src[start:end].decode()


print("Translation unit:", tu.spelling)
for cur in tu.cursor.get_children():
    assert isinstance(cur, clang.cindex.Cursor)
    assert isinstance(cur.kind, clang.cindex.CursorKind)
    if cur.location is None:
        continue
    assert isinstance(cur.location, clang.cindex.SourceLocation)
    assert isinstance(cur.location.file, clang.cindex.File)
    if cur.location.file.name != tu.spelling:
        continue
    if cur.kind == clang.cindex.CursorKind.FUNCTION_DECL:
        print(function_def(cur))
