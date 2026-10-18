from __future__ import annotations

import ast
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

from .schema import ContextUnit
from .tokenization import TokenCounter

_TEST_RE = re.compile(r"(^|/)(tests?|testdata)(/|$)|(^|/)(test_[^/]+|[^/]+\.(test|spec))\.", re.I)


def _bare(symbol: str) -> str:
    """Last segment of a possibly qualified symbol, e.g. `QuerySet.only` -> `only`."""
    return symbol.rsplit(".", 1)[-1]


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+([A-Za-z0-9_./-]+)", re.M),
    re.compile(r"^\s*from\s+([A-Za-z0-9_./-]+)\s+import\s+", re.M),
    re.compile(r"(?:require\(|from\s+|import\s+).*?[\"']([^\"']+)[\"']", re.M),
    re.compile(r"^\s*#include\s*[<\"]([^>\"]+)[>\"]", re.M),
]

# Relations the scanner can emit. `contains` is physical packaging, `encloses` is
# a class and its own methods, `state` is an attribute-sharing proxy, and the rest
# are dependency-like.
ALL_RELATIONS = (
    "contains",
    "encloses",
    "imports",
    "tests",
    "calls",
    "inherits",
    "decorates",
    "references",
    "state",
)

# Relations that may be walked when measuring hops. `contains` is excluded: two
# unrelated symbols sharing a file is packaging, not proximity, and treating it
# as one hop is what made a large file's every member a neighbour. `references`
# and `state` are recall-oriented and too dense to define a neighbourhood. All
# three still contribute to coherence.
DEFAULT_TRAVERSAL_RELATIONS = ("encloses", "imports", "tests", "calls", "inherits", "decorates")


@dataclass(frozen=True)
class HyperEdge:
    """One repository relation instance, e.g. a single file's contents or one
    resolved import.

    `members` is the full participant set and drives traversal. `sources` and
    `targets` record direction where the relation has one (the importer versus
    the imported symbol, the caller versus the callee); symmetric relations such
    as `contains` leave both empty.
    """

    relation: str
    members: frozenset[str]
    sources: frozenset[str] = frozenset()
    targets: frozenset[str] = frozenset()

    @property
    def directed(self) -> bool:
        return bool(self.sources) and bool(self.targets)


class Hypergraph:
    """Repository relations as hyperedges over context-unit ids.

    A hyperedge with k members is behaviourally equivalent to a k-clique of
    pairwise edges for reachability purposes: the union of a node's incident
    hyperedges is exactly the set of nodes it would be adjacent to under clique
    expansion. The difference shows up when something *counts* edges (coherence,
    MST weight) rather than merely traversing them.

    `traversal_relations` restricts which relations `neighbors`/`hop_lengths`
    may walk, so a relation can contribute to coherence without inflating
    neighbourhoods.
    """

    def __init__(self, traversal_relations: Iterable[str] | None = None) -> None:
        self.node_attrs: dict[str, dict] = {}
        self.edges: list[HyperEdge] = []
        self._incident: dict[str, list[int]] = {}
        self.traversal_relations = frozenset(traversal_relations) if traversal_relations is not None else None

    def add_node(self, uid: str, **attrs: object) -> None:
        self.node_attrs.setdefault(uid, {}).update(attrs)
        self._incident.setdefault(uid, [])

    def add_hyperedge(
        self,
        relation: str,
        members: Iterable[str] | None = None,
        *,
        sources: Iterable[str] = (),
        targets: Iterable[str] = (),
    ) -> None:
        src = frozenset(s for s in sources if s in self.node_attrs)
        dst = frozenset(t for t in targets if t in self.node_attrs)
        if members is None:
            uniq = src | dst
        else:
            uniq = frozenset(m for m in members if m in self.node_attrs)
        if len(uniq) < 2:
            return
        idx = len(self.edges)
        self.edges.append(HyperEdge(relation=relation, members=uniq, sources=src & uniq, targets=dst & uniq))
        for uid in uniq:
            self._incident[uid].append(idx)

    def __contains__(self, uid: object) -> bool:
        return uid in self.node_attrs

    def __len__(self) -> int:
        return len(self.node_attrs)

    @property
    def nodes(self) -> Iterator[str]:
        return iter(self.node_attrs)

    def incident_edge_ids(self, uid: str) -> list[int]:
        return self._incident.get(uid, [])

    def relation_counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for edge in self.edges:
            out[edge.relation] += 1
        return dict(out)

    def _allowed(self, relations: Iterable[str] | None) -> frozenset[str] | None:
        if relations is not None:
            return frozenset(relations)
        return self.traversal_relations

    def neighbors(self, uid: str, relations: Iterable[str] | None = None) -> set[str]:
        allowed = self._allowed(relations)
        out: set[str] = set()
        for idx in self._incident.get(uid, []):
            edge = self.edges[idx]
            if allowed is not None and edge.relation not in allowed:
                continue
            out.update(edge.members)
        out.discard(uid)
        return out

    def hop_lengths(
        self,
        source: str,
        cutoff: int | None = None,
        relations: Iterable[str] | None = None,
    ) -> dict[str, int]:
        """Unweighted BFS distance from `source` over traversable relations."""
        if source not in self.node_attrs:
            return {}
        allowed = self._allowed(relations)
        dist = {source: 0}
        queue: deque[str] = deque([source])
        while queue:
            u = queue.popleft()
            d = dist[u]
            if cutoff is not None and d >= cutoff:
                continue
            for v in self.neighbors(u, relations=allowed):
                if v not in dist:
                    dist[v] = d + 1
                    queue.append(v)
        return dist

    def induced(self, keep: Iterable[str]) -> "Hypergraph":
        """Restrict to a unit subset: hyperedges keep only members in `keep`,
        and are dropped once fewer than two members survive."""
        keep_set = set(keep)
        out = Hypergraph(traversal_relations=self.traversal_relations)
        for uid in keep_set:
            if uid in self.node_attrs:
                out.add_node(uid, **self.node_attrs[uid])
        for edge in self.edges:
            restricted = edge.members & keep_set
            if len(restricted) >= 2:
                out.add_hyperedge(
                    edge.relation,
                    restricted,
                    sources=edge.sources & keep_set,
                    targets=edge.targets & keep_set,
                )
        return out

    def clique_edges(self) -> Iterator[tuple[str, str, dict]]:
        """Pairwise edges implied by every hyperedge, for algorithms (MST, the
        frozen rooted backbone) that are only defined on ordinary graphs."""
        for edge in self.edges:
            members = sorted(edge.members)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    yield members[i], members[j], {"relation": edge.relation}


@dataclass
class RepositoryScanResult:
    units: list[ContextUnit]
    graph: Hypergraph


@dataclass
class _FileInfo:
    """Per-file scanning artifacts used to resolve relations across files."""

    units: list[ContextUnit] = field(default_factory=list)
    text: str = ""
    mentions: set[str] = field(default_factory=set)
    symbols: dict[str, list[str]] = field(default_factory=dict)  # symbol name -> uids
    calls: dict[str, set[str]] = field(default_factory=dict)  # uid -> called names
    self_calls: dict[str, set[str]] = field(default_factory=dict)  # uid -> self.<name>() names
    bases: dict[str, set[str]] = field(default_factory=dict)  # uid -> base class names
    decorators: dict[str, set[str]] = field(default_factory=dict)  # uid -> decorator names
    type_refs: dict[str, set[str]] = field(default_factory=dict)  # uid -> non-call name loads
    attrs: dict[str, set[str]] = field(default_factory=dict)  # uid -> self.<attr> names
    imported_names: dict[str, set[str]] = field(default_factory=dict)  # module dotted -> symbols
    owner_class: dict[str, str] = field(default_factory=dict)  # method uid -> class qualname
    class_members: dict[str, dict[str, str]] = field(default_factory=dict)  # class qualname -> name -> uid
    enclosed: dict[str, list[str]] = field(default_factory=dict)  # class uid -> member uids
    is_python: bool = False


class RepositoryScanner:
    """Build bounded repository artifacts and a lightweight heterogeneous hypergraph.

    Python uses AST-level function/class units. Other languages fall back to bounded
    file chunks, matching the manuscript's "when parser support is available" rule.

    Relations are resolved with scope rather than by global short-name matching:
    a call or type reference is looked up in the caller's own file first, then in
    the files it actually imports, and only then globally when the name is nearly
    unique. Import and test relations are narrowed to the units that genuinely
    mention the imported symbols, so one module-level fact no longer links every
    symbol of one file to every symbol of another.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        granularity: str = "symbol",
        include_extensions: Iterable[str] | None = None,
        exclude_dirs: Iterable[str] | None = None,
        max_file_bytes: int = 500_000,
        generic_chunk_lines: int = 120,
        relations: Iterable[str] | None = None,
        traversal_relations: Iterable[str] | None = None,
        max_symbol_fanout: int = 4,
    ):
        self.token_counter = token_counter
        self.granularity = granularity
        self.include_extensions = set(include_extensions or [".py", ".md"])
        self.exclude_dirs = set(exclude_dirs or [".git", "node_modules", ".venv"])
        self.max_file_bytes = max_file_bytes
        self.generic_chunk_lines = generic_chunk_lines
        self.relations = set(relations) if relations is not None else set(ALL_RELATIONS)
        self.traversal_relations = (
            set(traversal_relations) if traversal_relations is not None else set(DEFAULT_TRAVERSAL_RELATIONS)
        )
        self.max_symbol_fanout = int(max_symbol_fanout)

    # ------------------------------------------------------------------ scan

    def scan(self, repo_root: str | Path) -> RepositoryScanResult:
        repo_root = Path(repo_root).resolve()
        files = self._iter_files(repo_root)
        units: list[ContextUnit] = []
        infos: dict[str, _FileInfo] = {}

        for path in files:
            rel = path.relative_to(repo_root).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            info = _FileInfo(text=text, mentions=set(_IDENT_RE.findall(text)))
            if self.granularity == "file":
                info.units = [self._make_unit(rel, text, "file", None, 1, text.count("\n") + 1)]
            elif path.suffix.lower() == ".py":
                info.is_python = True
                self._python_units(rel, text, info)
            else:
                info.units = self._generic_units(rel, text)
            for u in info.units:
                if not u.symbol:
                    continue
                # Methods carry `Class.method`; the bare name is also indexed so
                # name-based resolution still works.
                info.symbols.setdefault(u.symbol, []).append(u.uid)
                bare = u.symbol.rsplit(".", 1)[-1]
                if bare != u.symbol:
                    info.symbols.setdefault(bare, []).append(u.uid)
            units.extend(info.units)
            infos[rel] = info

        graph = Hypergraph(traversal_relations=self.traversal_relations & self.relations)
        for u in units:
            graph.add_node(u.uid, path=u.path, kind=u.kind, symbol=u.symbol)

        unit_mentions = {u.uid: set(_IDENT_RE.findall(u.text)) for u in units}
        import_targets = self._resolve_imports(infos)

        if "contains" in self.relations:
            self._add_contains(graph, infos)
        if "encloses" in self.relations:
            self._add_encloses(graph, infos)
        if "imports" in self.relations:
            self._add_import_edges(graph, infos, import_targets, unit_mentions)
        if "tests" in self.relations:
            self._add_test_edges(graph, infos, unit_mentions)
        self._add_symbol_edges(graph, infos, import_targets)
        if "state" in self.relations:
            self._add_state_edges(graph, infos)
        return RepositoryScanResult(units=units, graph=graph)

    # ------------------------------------------------------- units and files

    def _iter_files(self, root: Path) -> list[Path]:
        out: list[Path] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel_parts = p.relative_to(root).parts
            if any(part in self.exclude_dirs for part in rel_parts[:-1]):
                continue
            if p.suffix.lower() not in self.include_extensions:
                continue
            try:
                if p.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            out.append(p)
        return sorted(out)

    def _make_unit(
        self,
        rel: str,
        text: str,
        kind: str,
        symbol: str | None,
        start_line: int,
        end_line: int,
        ordinal: int = 0,
    ) -> ContextUnit:
        uid = f"{rel}::{kind}:{symbol or ordinal}:{start_line}-{end_line}"
        return ContextUnit(
            uid=uid,
            path=rel,
            text=text,
            kind=kind,
            symbol=symbol,
            start_line=start_line,
            end_line=end_line,
            token_cost=self.token_counter.count(text),
            node_id=uid,
        )

    def _python_units(self, rel: str, text: str, info: _FileInfo) -> None:
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            info.is_python = False
            info.units = self._generic_units(rel, text)
            return
        info.imported_names = self._python_imports(tree)
        top_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if not top_nodes:
            info.units = [self._make_unit(rel, text, "file", None, 1, max(1, len(lines)))]
            return
        for node in top_nodes:
            self._emit_python_unit(rel, lines, node, info, owner=None)

    def _emit_python_unit(
        self,
        rel: str,
        lines: list[str],
        node: ast.AST,
        info: _FileInfo,
        owner: str | None,
    ) -> list[ContextUnit]:
        """Emit one unit per definition, descending into class bodies.

        A class becomes a shell (decorators, header, docstring, class-level
        attributes, and one signature line per member) plus a separate unit per
        method. Keeping whole classes as single units made an edit to one 30-line
        method cost the entire class, and hid every intra-class call from the
        graph because both endpoints lived inside the same unit.
        """
        name = str(getattr(node, "name", ""))
        qual = f"{owner}.{name}" if owner else name
        start = int(getattr(node, "lineno", 1))
        end = int(getattr(node, "end_lineno", start))

        if isinstance(node, ast.ClassDef):
            children = [c for c in node.body if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            shell = self._class_shell(lines, node, children)
            unit = self._make_unit(rel, shell, "class", qual, start, end)
            info.units.append(unit)
            if owner:
                info.owner_class[unit.uid] = owner
            self._collect_symbol_usage(node, unit.uid, info, skip_nested=True)
            members: list[ContextUnit] = []
            for child in children:
                members.extend(self._emit_python_unit(rel, lines, child, info, owner=qual))
            if members:
                info.enclosed[unit.uid] = [u.uid for u in members]
                info.class_members[qual] = {
                    (u.symbol or "").rsplit(".", 1)[-1]: u.uid for u in members if u.symbol
                }
            return [unit, *members]

        kind = "test" if name.startswith("test") else ("method" if owner else "function")
        snippet = "\n".join(lines[start - 1 : end])
        unit = self._make_unit(rel, snippet, kind, qual, start, end)
        info.units.append(unit)
        if owner:
            info.owner_class[unit.uid] = owner
        self._collect_symbol_usage(node, unit.uid, info, skip_nested=False)
        return [unit]

    @staticmethod
    def _class_shell(lines: list[str], node: ast.ClassDef, children: list[ast.AST]) -> str:
        """Class text with member bodies replaced by their signatures."""
        spans: list[tuple[int, int]] = []
        for child in children:
            c_start = min(
                [int(getattr(child, "lineno", 1))]
                + [int(getattr(d, "lineno", 1)) for d in (getattr(child, "decorator_list", None) or [])]
            )
            spans.append((c_start, int(getattr(child, "end_lineno", c_start))))
        spans.sort()

        out: list[str] = []
        i = int(getattr(node, "lineno", 1))
        end = int(getattr(node, "end_lineno", i))
        idx = 0
        while i <= end:
            if idx < len(spans) and i == spans[idx][0]:
                c_start, c_end = spans[idx]
                out.extend(RepositoryScanner._signature_lines(lines, c_start, c_end))
                i = c_end + 1
                idx += 1
                continue
            out.append(lines[i - 1])
            i += 1
        return "\n".join(out)

    @staticmethod
    def _signature_lines(lines: list[str], start: int, end: int) -> list[str]:
        """Decorator and signature lines of a definition, followed by an ellipsis."""
        out: list[str] = []
        for ln in range(start, min(end, start + 20) + 1):
            line = lines[ln - 1]
            out.append(line)
            if line.rstrip().endswith(":"):
                break
        # Indent from the definition's own line, not a signature continuation line.
        first = out[0] if out else ""
        indent = len(first) - len(first.lstrip())
        out.append(" " * (indent + 4) + "...")
        return out

    @staticmethod
    def _python_imports(tree: ast.Module) -> dict[str, set[str]]:
        """Map each imported module (dotted, relative kept as leading dots) to the
        symbol names taken from it; a bare `import x` maps to an empty set."""
        out: dict[str, set[str]] = defaultdict(set)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    out[alias.name].add("")
            elif isinstance(node, ast.ImportFrom):
                module = "." * int(node.level or 0) + (node.module or "")
                for alias in node.names:
                    out[module].add(alias.name)
        return {k: {n for n in v if n} for k, v in out.items()}

    @staticmethod
    def _own_nodes(node: ast.AST, skip_nested: bool) -> Iterator[ast.AST]:
        """Nodes belonging to this definition, optionally without descending into
        nested definitions (so a class shell does not absorb its methods' usage)."""
        stack = list(ast.iter_child_nodes(node))
        while stack:
            cur = stack.pop()
            if skip_nested and isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            yield cur
            stack.extend(ast.iter_child_nodes(cur))

    @staticmethod
    def _collect_symbol_usage(node: ast.AST, uid: str, info: _FileInfo, skip_nested: bool) -> None:
        calls: set[str] = set()
        self_calls: set[str] = set()
        called_nodes: set[int] = set()
        bases: set[str] = set()
        decorators: set[str] = set()
        type_refs: set[str] = set()
        attrs: set[str] = set()

        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.add(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.add(base.attr)
        for dec in getattr(node, "decorator_list", None) or []:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Name):
                decorators.add(target.id)
            elif isinstance(target, ast.Attribute):
                decorators.add(target.attr)

        for sub in RepositoryScanner._own_nodes(node, skip_nested):
            if isinstance(sub, ast.Call):
                called_nodes.add(id(sub.func))
                if isinstance(sub.func, ast.Name):
                    calls.add(sub.func.id)
                elif isinstance(sub.func, ast.Attribute):
                    # `self.foo()` resolves inside the owning class first.
                    if isinstance(sub.func.value, ast.Name) and sub.func.value.id == "self":
                        self_calls.add(sub.func.attr)
                    else:
                        calls.add(sub.func.attr)
            elif isinstance(sub, ast.Attribute):
                if isinstance(sub.value, ast.Name) and sub.value.id == "self":
                    attrs.add(sub.attr)
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if id(sub) not in called_nodes:
                    type_refs.add(sub.id)

        info.calls[uid] = calls
        info.self_calls[uid] = self_calls
        # A name that is called is a call, not a bare reference.
        info.type_refs[uid] = type_refs - calls - {"self"}
        info.attrs[uid] = attrs - self_calls
        if bases:
            info.bases[uid] = bases
        if decorators:
            info.decorators[uid] = decorators

    def _generic_units(self, rel: str, text: str) -> list[ContextUnit]:
        lines = text.splitlines()
        if not lines:
            return [self._make_unit(rel, "", "file", None, 1, 1)]
        if len(lines) <= self.generic_chunk_lines:
            return [self._make_unit(rel, text, "file", None, 1, len(lines))]
        out: list[ContextUnit] = []
        for ordinal, start0 in enumerate(range(0, len(lines), self.generic_chunk_lines)):
            end0 = min(len(lines), start0 + self.generic_chunk_lines)
            snippet = "\n".join(lines[start0:end0])
            out.append(self._make_unit(rel, snippet, "chunk", None, start0 + 1, end0, ordinal))
        return out

    # -------------------------------------------------------- import resolution

    @staticmethod
    def _module_index(rels: Iterable[str]) -> dict[str, list[str]]:
        """Dotted module name (and every suffix of it) to the files providing it."""
        idx: dict[str, list[str]] = defaultdict(list)
        for rel in rels:
            p = PurePosixPath(rel)
            if p.suffix != ".py":
                continue
            parts = list(p.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = p.stem
            if not parts:
                continue
            for i in range(len(parts)):
                idx[".".join(parts[i:])].append(rel)
        return idx

    @staticmethod
    def _path_index(rels: Iterable[str]) -> dict[str, list[str]]:
        idx: dict[str, list[str]] = defaultdict(list)
        for rel in rels:
            p = PurePosixPath(rel)
            idx[rel].append(rel)
            idx[p.name].append(rel)
            idx[p.stem].append(rel)
        return idx

    def _resolve_module(
        self,
        module: str,
        importer: str,
        module_idx: dict[str, list[str]],
        path_idx: dict[str, list[str]],
    ) -> str | None:
        """Resolve one import target to a repository file, or None when the name
        is external or ambiguous."""
        if not module:
            return None
        level = len(module) - len(module.lstrip("."))
        name = module.lstrip(".")
        if level:
            base = PurePosixPath(importer).parent
            for _ in range(max(0, level - 1)):
                base = base.parent
            dotted = ".".join([p for p in base.parts if p] + ([name] if name else []))
            candidates = module_idx.get(dotted, [])
            if len(candidates) == 1:
                return candidates[0]
            if not name:
                return None
        parts = name.split(".")
        for i in range(len(parts)):
            hits = module_idx.get(".".join(parts[i:]), [])
            if len(hits) == 1:
                return hits[0]
        hits = path_idx.get(parts[-1], [])
        if len(hits) == 1:
            return hits[0]
        return None

    def _resolve_imports(self, infos: dict[str, _FileInfo]) -> dict[str, dict[str, set[str]]]:
        """For each file, the repository files it imports and the symbol names
        taken from each."""
        module_idx = self._module_index(infos)
        path_idx = self._path_index(infos)
        out: dict[str, dict[str, set[str]]] = {}
        for rel, info in infos.items():
            targets: dict[str, set[str]] = defaultdict(set)
            if info.is_python:
                refs: dict[str, set[str]] = dict(info.imported_names)
            else:
                # No parser for this language: fall back to regex import lines.
                raw: set[str] = set()
                for pattern in _IMPORT_PATTERNS:
                    raw.update(str(r) for r in pattern.findall(info.text))
                refs = {r.strip().replace("\\", "/").replace("/", "."): set() for r in raw}
            for module, names in refs.items():
                target = self._resolve_module(module, rel, module_idx, path_idx)
                if target is None or target == rel:
                    continue
                targets[target].update(names)
            if targets:
                out[rel] = dict(targets)
        return out

    # -------------------------------------------------------------- relations

    @staticmethod
    def _add_contains(graph: Hypergraph, infos: dict[str, _FileInfo]) -> None:
        # One hyperedge per file: every unit defined in that file. Physical
        # packaging, so it stays symmetric and undirected.
        for info in infos.values():
            if len(info.units) >= 2:
                graph.add_hyperedge("contains", (u.uid for u in info.units))

    @staticmethod
    def _add_encloses(graph: Hypergraph, infos: dict[str, _FileInfo]) -> None:
        # A class and its own methods. Unlike file-level packaging this is a real
        # scope, so it stays traversable: editing a method usually needs the class.
        for info in infos.values():
            for class_uid, members in info.enclosed.items():
                graph.add_hyperedge("encloses", sources=[class_uid], targets=members)

    def _add_import_edges(
        self,
        graph: Hypergraph,
        infos: dict[str, _FileInfo],
        import_targets: dict[str, dict[str, set[str]]],
        unit_mentions: dict[str, set[str]],
    ) -> None:
        for rel, targets in import_targets.items():
            src_info = infos.get(rel)
            if src_info is None:
                continue
            for target, names in targets.items():
                dst_info = infos.get(target)
                if dst_info is None:
                    continue
                wanted = {n for n in names if n}
                # Prefer the explicitly imported symbols; otherwise fall back to
                # whichever of the target's symbols this file actually mentions.
                target_units = [
                    u
                    for u in dst_info.units
                    if u.symbol and (_bare(u.symbol) in wanted or _bare(u.symbol) in src_info.mentions)
                ]
                if not target_units and len(dst_info.units) == 1:
                    target_units = list(dst_info.units)
                if not target_units:
                    continue
                keys = {_bare(u.symbol) for u in target_units if u.symbol}
                keys.add(PurePosixPath(target).stem)
                source_units = [u for u in src_info.units if unit_mentions.get(u.uid, set()) & keys]
                if not source_units:
                    continue
                graph.add_hyperedge(
                    "imports",
                    sources=[u.uid for u in source_units],
                    targets=[u.uid for u in target_units],
                )

    def _add_test_edges(
        self,
        graph: Hypergraph,
        infos: dict[str, _FileInfo],
        unit_mentions: dict[str, set[str]],
    ) -> None:
        stems: dict[str, list[str]] = defaultdict(list)
        for rel in infos:
            stem = PurePosixPath(rel).stem.lower().removeprefix("test_").removesuffix("_test")
            stems[stem].append(rel)
        for rel, info in infos.items():
            if not _TEST_RE.search(rel):
                continue
            stem = PurePosixPath(rel).stem.lower().removeprefix("test_").removesuffix("_test")
            for target in stems.get(stem, []):
                if target == rel or _TEST_RE.search(target):
                    continue
                dst_info = infos[target]
                # Only the implementation units this test actually names.
                target_units = [u for u in dst_info.units if u.symbol and _bare(u.symbol) in info.mentions]
                if not target_units and len(dst_info.units) == 1:
                    target_units = list(dst_info.units)
                if not target_units:
                    continue
                keys = {_bare(u.symbol) for u in target_units if u.symbol}
                keys.add(PurePosixPath(target).stem)
                source_units = [u for u in info.units if unit_mentions.get(u.uid, set()) & keys]
                if not source_units:
                    continue
                graph.add_hyperedge(
                    "tests",
                    sources=[u.uid for u in source_units],
                    targets=[u.uid for u in target_units],
                )

    def _add_symbol_edges(
        self,
        graph: Hypergraph,
        infos: dict[str, _FileInfo],
        import_targets: dict[str, dict[str, set[str]]],
    ) -> None:
        """Call, inheritance, decorator and type-reference relations.

        All four resolve a bare name through the same scope ladder, which is what
        keeps `calls` from linking every same-named function in the repository.
        """
        global_symbols: dict[str, list[str]] = defaultdict(list)
        for info in infos.values():
            for name, uids in info.symbols.items():
                global_symbols[name].extend(uids)

        def resolve(name: str, caller_file: str, allow_global: bool) -> list[str]:
            own = infos[caller_file].symbols.get(name)
            if own:
                return own
            hits: list[str] = []
            for target in import_targets.get(caller_file, {}):
                hits.extend(infos[target].symbols.get(name, []))
            if hits:
                return hits[: self.max_symbol_fanout]
            if not allow_global:
                return []
            cands = global_symbols.get(name, [])
            # A name matched everywhere carries no information; drop it.
            return cands if 0 < len(cands) <= self.max_symbol_fanout else []

        specs = (
            ("calls", "calls", True),
            ("inherits", "bases", True),
            ("decorates", "decorators", True),
            ("references", "type_refs", False),
        )
        for relation, attr, allow_global in specs:
            if relation not in self.relations:
                continue
            for rel, info in infos.items():
                for uid, names in getattr(info, attr).items():
                    for name in names:
                        targets = [t for t in resolve(name, rel, allow_global) if t != uid]
                        if not targets:
                            continue
                        graph.add_hyperedge(relation, sources=[uid], targets=targets)

        if "calls" not in self.relations:
            return
        for rel, info in infos.items():
            for uid, names in info.self_calls.items():
                owner = info.owner_class.get(uid)
                members = info.class_members.get(owner or "", {})
                for name in names:
                    # A `self.foo()` is its own class's method whenever one exists;
                    # only fall back to the wider ladder for inherited methods.
                    own = members.get(name)
                    targets = [own] if own and own != uid else [t for t in resolve(name, rel, True) if t != uid]
                    if not targets:
                        continue
                    graph.add_hyperedge("calls", sources=[uid], targets=targets)

    @staticmethod
    def _add_state_edges(graph: Hypergraph, infos: dict[str, _FileInfo]) -> None:
        """Units touching the same instance attribute within one class (or one
        file for module-level code).

        A cheap data-flow proxy: many bugs live in shared state rather than at a
        call site. Now that methods are separate units this is mostly a
        method-to-method relation inside a class, which is where it belongs.
        """
        for rel, info in infos.items():
            by_attr: dict[tuple[str, str], list[str]] = defaultdict(list)
            for uid, attrs in info.attrs.items():
                scope = info.owner_class.get(uid) or rel
                for attr in attrs:
                    if attr.startswith("__"):
                        continue
                    by_attr[(scope, attr)].append(uid)
            for uids in by_attr.values():
                if len(set(uids)) >= 2:
                    graph.add_hyperedge("state", set(uids))
