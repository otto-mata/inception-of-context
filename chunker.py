from __future__ import annotations

import importlib
from bisect import bisect_right
from dataclasses import dataclass

from tree_sitter import Language, Parser, Tree

MAX_CHUNK_BYTES = 3000

# ---------------------------------------------------------------- language specs


@dataclass
class Spec:
    module: str  # grammar package
    func: str  # function returning the grammar pointer
    defs: dict  # node type -> kind label
    containers: frozenset = frozenset()  # defs that can be split into members
    wrappers: frozenset = (
        frozenset()
    )  # nodes that wrap a definition (decorators, export)
    leading: frozenset = frozenset(
        {"comment"}
    )  # nodes glued to the definition below them
    needs_body: frozenset = (
        frozenset()
    )  # only a definition if it has a struct/enum body


_C_DEFS = {
    "function_definition": "function",
    "struct_specifier": "struct",
    "union_specifier": "union",
    "enum_specifier": "enum",
    "type_definition": "typedef",
    "declaration": "declaration",
}
_C_NEEDS_BODY = frozenset(
    {
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
        "type_definition",
        "declaration",
    }
)

_CLASSES = frozenset({"class_declaration", "abstract_class_declaration"})

SPECS: dict[str, Spec] = {
    ".c": Spec("tree_sitter_c", "language", _C_DEFS, needs_body=_C_NEEDS_BODY),
    ".h": Spec("tree_sitter_c", "language", _C_DEFS, needs_body=_C_NEEDS_BODY),
}

_SPECIFIERS = {"struct_specifier", "union_specifier", "enum_specifier"}
_IDENTS = {"identifier", "field_identifier", "type_identifier", "property_identifier"}
_PARSERS: dict[str, Parser | None] = {}


def _get_parser(ext: str) -> Parser | None:
    if ext not in _PARSERS:
        spec = SPECS[ext]
        try:
            mod = importlib.import_module(spec.module)
            _PARSERS[ext] = Parser(Language(getattr(mod, spec.func)()))
        except (ImportError, AttributeError):
            _PARSERS[ext] = None
    return _PARSERS[ext]


# ---------------------------------------------------------------- chunk type


@dataclass
class Chunk:
    text: str
    start_line: int
    end_line: int
    kind: str  # function, class, struct, method, module, class_header, text...
    name: str | None = None
    parent: str | None = None
    part: str | None = None  # "2/3" when an oversized definition was split

    def embed_text(self, path: str) -> str:
        label = f"{self.kind} {self.name}" if self.name else self.kind
        if self.parent:
            label += f" (in {self.parent})"
        if self.part:
            label += f" [part {self.part}]"
        return f"# {path}:{self.start_line}-{self.end_line} | {label}\n{self.text}"


# ---------------------------------------------------------------- chunker


class _Chunker:
    def __init__(self, data: bytes, spec: Spec, max_bytes: int):
        self.data, self.spec, self.max = data, spec, max_bytes
        self.chunks: list[Chunk] = []
        self.line_starts = [0] + [i + 1 for i, b in enumerate(data) if b == 10]

    # -- helpers
    def line_of(self, byte: int) -> int:
        return bisect_right(self.line_starts, max(byte, 0))  # 1-based

    def text(self, node) -> str:
        return self.data[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def name_of(self, node) -> str | None:
        n = node.child_by_field_name("name")
        if n is not None:
            return self.text(n)
        d = node.child_by_field_name("declarator")  # C: function/typedef names
        while d is not None and d.type not in _IDENTS:
            d = d.child_by_field_name("declarator")
        if d is not None:
            return self.text(d)
        ty = node.child_by_field_name("type")  # Rust impl, C bare struct
        if ty is not None:
            return self.name_of(ty) or (self.text(ty) if ty.type in _IDENTS else None)
        for c in node.named_children:  # Go: type_declaration > type_spec
            n = c.child_by_field_name("name")
            if n is not None:
                return self.text(n)
        return None

    def body_specifier(self, node):
        """First struct/union/enum specifier that has a body (C definitions)."""
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type in _SPECIFIERS and n.child_by_field_name("body") is not None:
                return n
            stack.extend(reversed(n.children))
        return None

    def definition(self, node):
        """Return (kind, name, inner_node) if node is a definition worth its own chunk."""
        inner = node
        while inner.type in self.spec.wrappers:
            inner = inner.child_by_field_name(
                "definition"
            ) or inner.child_by_field_name("declaration")
            if inner is None:
                return None
        kind = self.spec.defs.get(inner.type)
        if kind is None:
            return None
        name = self.name_of(inner)
        if inner.type in self.spec.needs_body:
            spec_node = self.body_specifier(inner)
            if spec_node is None:
                return None  # prototype, plain variable...
            kind = self.spec.defs[spec_node.type]
            name = name or self.name_of(spec_node)
        return kind, name, inner

    def leading_start(self, children, i) -> int:
        """Extend start upward over adjacent comments/attributes (no blank line between)."""
        start = children[i].start_byte
        j = i - 1
        while j >= 0 and children[j].type in self.spec.leading:
            if self.line_of(start) - self.line_of(children[j].end_byte - 1) > 1:
                break
            start = children[j].start_byte
            j -= 1
        return start

    # -- emitting
    def emit(self, start, end, kind, name, parent, gap=False):
        raw = self.data[start:end]
        start += len(raw) - len(raw.lstrip())
        end = start + len(raw.strip())
        if end <= start:
            return
        body = self.data[start:end].decode("utf-8", "replace")
        if gap and not any(c.isalnum() for c in body):  # lone "}" etc.
            return
        if end - start <= self.max:
            self._add(start, end, body, kind, name, parent)
            return
        # Oversized: split on line boundaries, repeat the signature on later parts.
        header = body.splitlines()[0].strip()[:200]
        windows = list(self._windows(start, end))
        for n, (s, e) in enumerate(windows, 1):
            piece = self.data[s:e].decode("utf-8", "replace").strip()
            if n > 1 and not gap:
                piece = f"{header}\n    ...\n{piece}"
            self._add(s, e, piece, kind, name, parent, part=f"{n}/{len(windows)}")

    def _windows(self, start, end):
        pos = cur = start
        for line in self.data[start:end].splitlines(keepends=True):
            if pos > cur and pos - cur + len(line) > self.max:
                yield cur, pos
                cur = pos
            pos += len(line)
        yield cur, end

    def _add(self, s, e, text, kind, name, parent, part=None):
        self.chunks.append(
            Chunk(text, self.line_of(s), self.line_of(e - 1), kind, name, parent, part)
        )

    # -- main walk
    def walk(self, children, start, end, gap_kind, gap_name, gap_parent, member_parent):
        cursor = start
        for i, child in enumerate(children):
            info = self.definition(child)
            if info is None:
                continue  # stays in the surrounding gap
            kind, name, inner = info
            dstart = max(self.leading_start(children, i), cursor)
            self.emit(cursor, dstart, gap_kind, gap_name, gap_parent, gap=True)

            body = inner.child_by_field_name("body")
            too_big = child.end_byte - dstart > self.max
            if (
                too_big
                and inner.type in self.spec.containers
                and body is not None
                and body.named_child_count
            ):
                # Split a big class/impl into member chunks + header chunk(s)
                self.walk(
                    list(body.named_children),
                    dstart,
                    child.end_byte,
                    f"{kind}_header",
                    name,
                    member_parent,
                    name,
                )
            else:
                self.emit(dstart, child.end_byte, kind, name, member_parent)
            cursor = child.end_byte
        self.emit(cursor, end, gap_kind, gap_name, gap_parent, gap=True)


# ---------------------------------------------------------------- public API


def _line_windows(text: str, window=40, overlap=8) -> list[Chunk]:
    lines, out, step = text.splitlines(), [], window - overlap
    for s in range(0, len(lines), step):
        piece = lines[s : s + window]
        if "".join(piece).strip():
            out.append(Chunk("\n".join(piece), s + 1, s + len(piece), "text"))
        if s + window >= len(lines):
            break
    return out


def parse_source(ext: str, source: str) -> Tree | list[Chunk]:
    spec = SPECS.get(ext)
    parser = _get_parser(ext) if spec else None
    if parser is None:
        return _line_windows(source)  # unsupported or grammar missing

    data = source.encode("utf-8")
    return parser.parse(data)


def chunk_source(
    ext: str, source: str, max_bytes: int = MAX_CHUNK_BYTES
) -> list[Chunk]:
    parsed = parse_source(ext, source)
    if isinstance(parsed, list):
        return parsed
    root = parsed.root_node
    ck = _Chunker(data, spec, max_bytes)
    ck.walk(list(root.named_children), 0, len(data), "module", None, None, None)
    return ck.chunks or _line_windows(source)


if __name__ == "__main__":
    import pathlib
    import sys

    p = pathlib.Path(sys.argv[1])
    content = p.read_text()
    tree = parse_source(p.suffix, content)
    if isinstance(tree, list):
        print("Parsing failed !")
        sys.exit(1)
    for i, ch in enumerate(tree.root_node.children):
        if not ch.child_count:
            print(
                i,
                ch.grammar_name,
                content[ch.start_byte : ch.end_byte],
                ch.descendant_count,
            )
