# Expression AST

Active contributors: KeigoShimadaCC

## Purpose

`noether/npr/ast.py` defines the typed expression tree that holds the Lagrangian and every kernel result. The Lagrangian and every kernel result live as trees of these nodes, never as strings. LaTeX is a rendering of the tree (`noether/npr/latex.py`), not a source of truth. This is `docs/02_TECH_SPEC.md` section 4 made concrete.

The tree uses abstract indices: a name plus a variance (`up` or `down`), with no implicit position. Index balance is checked structurally at V0 (`noether/npr/validate.py`).

## Node table

| Node | Literal | Fields | Meaning |
|---|---|---|---|
| `Num` | `num` | `p: int`, `q: int = 1` | Rational number `p/q`. |
| `Sym` | `sym` | `name: str` | Named scalar symbol: a constant or a scalar field, per its `ObjectDecl`. |
| `Func` | `func` | `name: str`, `args: list[Expr]` | Function of scalar arguments, e.g. `F(phi)` or `K(phi, X)`. Arguments must be scalars (V0 rejects free indices in args). |
| `Tensor` | `tensor` | `name: str`, `indices: list[Index] = []`, `connection: str \| None = None` | Abstract-index tensor instance, e.g. `R_{mu nu}` or `g^{mu nu}`. The optional `connection` annotates which connection the tensor refers to. |
| `Deriv` | `deriv` | `op: "covariant" \| "partial"`, `index: Index`, `expr: Expr`, `connection: str \| None = "metric"` | Single derivative. Nest for higher derivatives. The `connection` field names which connection a covariant derivative uses; `"metric"` is the default. |
| `Pow` | `pow` | `base: Expr`, `exp: int` | Integer power. V0 rejects powers of expressions with free indices. |
| `Prod` | `prod` | `factors: list[Expr]` | Ordered product. |
| `Sum` | `sum` | `terms: list[Expr]` | Ordered sum. Terms must share free indices (V0). |
| `Index` | n/a | `name: str`, `variance: "up" \| "down"` | An abstract index. `flipped()` returns the opposite variance. |

`Expr` is the discriminated union of the eight node types, keyed on the `node` literal field. pydantic rebuilds the forward references after definition so the recursive types resolve.

## Convenience constructors

`noether/npr/ast.py` ships helpers that keep eval and test code readable:

| Constructor | Returns |
|---|---|
| `up(name)` | `Index(name=name, variance="up")` |
| `down(name)` | `Index(name=name, variance="down")` |
| `num(p, q=1)` | `Num(p=p, q=q)` |
| `tensor(name, *indices, connection=None)` | `Tensor(name=name, indices=list(indices), connection=connection)` |
| `cov(index, expr, *, connection="metric")` | `Deriv(op="covariant", index=index, expr=expr, connection=connection)` |
| `prod(*factors)` | `Prod(factors=list(factors))` |
| `add(*terms)` | `Sum(terms=list(terms))` |

There is no `partial` convenience constructor; partial derivatives are built directly as `Deriv(op="partial", ...)`.

## Key abstractions

| Abstraction | Where | Role |
|---|---|---|
| `Expr` discriminated union | `ast.py` | The single type the rest of the codebase accepts when it takes an expression. |
| `Index` with variance | `ast.py` | Abstract-index bookkeeping; `flipped()` is the only variance mutation. |
| `connection` annotation on `Tensor` and `Deriv` | `ast.py` | Names which affine connection a tensor or covariant derivative refers to, so metric-compatible and independent-connection objects are distinguishable in the tree. |
| `node` literal discriminator | each node | Drives pydantic deserialization and the match statements in the renderer, validator, and geometric-cue walker. |

## How it works

1. The LaTeX action parser (`noether/npr/parse.py`) builds an `Expr` tree for the Lagrangian. The tree is stored on `Action.lagrangian`; `Action.lagrangian_tex` is a cached rendering, never the source of truth.
2. The renderer (`noether/npr/latex.py`) walks the tree with a `match` statement over every node type. Rendering is deterministic: same NPR in, byte-identical LaTeX out. Sums render in stored order; canonical term ordering is the good-form pipeline's job, not the renderer's.
3. The structural validator (`noether/npr/validate.py`) computes free indices with a `match` statement over every node type. Within a product, an index name may appear at most twice and only as an up/down pair. Across a sum, free indices must agree exactly. Unless `metric_compatible` is explicitly set, validation never treats raising/lowering across `nabla` as free.
4. Kernel results come back as `Expr` trees (rendered to LaTeX for display). A result tree carries the same node types as the input Lagrangian.

## The coupling rule

`noether/orchestrator/elicit.py` `_detect_geometric_cues walks the action's AST to extract structural cues (curvature, connection, torsion, non-metricity, `f(Q)`/`f(T)` family) that ground geometry-inference proposals in the action's actual content. The walk uses a `match` statement covering every current node type.

Adding a new node type to `ast.py` without adding a corresponding case in `_detect_geometric_cues` raises `UnhandledASTNodeError` at runtime. This is fail-loud by design: an unhandled node means the geometric-cue walk is incomplete and inference may miss structural cues. The fix is to add the case, not to silence the error.

## Integration points

| System | How it uses the AST |
|---|---|
| NPR schema | `Action.lagrangian: Expr` is the parsed action. |
| LaTeX renderer | `noether/npr/latex.py` `render(expr)` walks the tree. |
| Structural validation | `noether/npr/validate.py` `free_indices` and `validate_expression` walk the tree (V0). |
| Elicitation | `_detect_geometric_cues` walks the Lagrangian to ground geometry proposals. |
| Kernel adapters | Script generators read the tree to emit kernel-native declarations. |
| Provenance | Derivations are stored as `Expr` trees in the bundle. |

## Entry points for modification

- Add a node type: define it in `ast.py` with a fresh `node` literal, add it to the `Expr` union, rebuild forward references, then add cases to `latex.py` `render`, `validate.py` `free_indices`, and `elicit.py` `_detect_geometric_cues`. Missing any case is a fail-loud bug.
- Add a convenience constructor: add it in `ast.py` next to the existing helpers.
- Change rendering of an existing node: edit `latex.py` only. The tree is unchanged.
- Change index-balance rules: edit `validate.py` only. Keep the `metric_compatible` gate explicit.

## Key source files

| File | Role |
|---|---|
| `noether/npr/ast.py` | The node types, `Index`, `Expr` union, convenience constructors. Authoritative source. |
| `noether/npr/latex.py` | Deterministic LaTeX rendering of `Expr`. |
| `noether/npr/validate.py` | V0 structural validation of `Expr`. |
| `noether/npr/parse.py` | LaTeX action parser that builds the Lagrangian tree. |
| `noether/npr/schema.py` | `Action.lagrangian: Expr` and `Action.lagrangian_tex`. |
| `noether/orchestrator/elicit.py` | `_detect_geometric_cues` and `UnhandledASTNodeError`. |
