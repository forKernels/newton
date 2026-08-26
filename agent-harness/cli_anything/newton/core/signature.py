"""The accepted keyword names of any newton API, read off the real build.

Ported from the newton-lab harness in the cli-anything repo, where it earned
its place. Newton's builder keyword lists shift across minor versions, and the
add-on that consumes this engine survives that by DROPPING what a build does
not accept - which means a misspelled or moved keyword produces a working call
that silently ignores the setting.

That is not hypothetical. `density=` moved onto `ShapeConfig` and was dropped
from every `add_shape_*` call for a whole release exactly this way, so every
rigid body silently used newton's default density. Nothing raised, nothing
warned where anyone was looking, and the scenes merely came out wrong.

So: never guess a keyword against this engine. Ask it.

Unlike the newton-lab harness, this one runs in an interpreter that HAS
newton, so there is no subprocess and no deps directory to point at.
"""

import inspect

# Shortcuts, not a whitelist - any dotted path under `newton` also resolves.
# They exist so a caller does not need to know newton's module layout to ask
# a question about it.
WELL_KNOWN = [
    "ModelBuilder.add_body",
    "ModelBuilder.add_link",
    "ModelBuilder.add_shape_box",
    "ModelBuilder.add_shape_sphere",
    "ModelBuilder.add_shape_capsule",
    "ModelBuilder.add_shape_cylinder",
    "ModelBuilder.add_shape_cone",
    "ModelBuilder.add_shape_ellipsoid",
    "ModelBuilder.add_shape_plane",
    "ModelBuilder.add_shape_mesh",
    "ModelBuilder.add_shape_convex_hull",
    "ModelBuilder.add_shape_heightfield",
    "ModelBuilder.add_ground_plane",
    "ModelBuilder.add_cloth_mesh",
    "ModelBuilder.add_soft_mesh",
    "ModelBuilder.add_articulation",
    "ModelBuilder.ShapeConfig",
    "ModelBuilder",
    "Mesh",
    "Mesh.build_sdf",
    "Heightfield",
    "CollisionPipeline",
    "solvers.SolverVBD",
    "solvers.SolverMuJoCo",
    "solvers.SolverXPBD",
    "solvers.SolverStyle3D",
    "solvers.SolverKamino",
    "solvers.SolverSemiImplicit",
    "solvers.SolverFeatherstone",
    "solvers.SolverImplicitMPM",
]


def resolve(newton, path):
    """Walk a dotted path from the newton module. Raises with what it found."""
    obj = newton
    walked = "newton"
    for part in path.split("."):
        nxt = getattr(obj, part, None)
        if nxt is None:
            available = sorted(n for n in dir(obj) if not n.startswith("_") and n.lower().startswith(part[:3].lower()))
            hint = f" Nearest names under {walked}: {', '.join(available[:12])}" if available else ""
            raise AttributeError(f"{walked} has no attribute {part!r}.{hint}")
        obj = nxt
        walked = f"{walked}.{part}"
    return obj, walked


def _safe(value):
    """A default rendered for JSON without pretending an object is data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return repr(value)


def describe(obj, path):
    """One callable's parameters, as data.

    A class is described by its `__init__` - which is what a caller of
    `SolverVBD(model, ...)` actually needs - and that is stated in the result
    rather than left to be inferred from a surprising parameter list.
    """
    target, kind = obj, "callable"
    if inspect.isclass(obj):
        target, kind = obj.__init__, "class (__init__)"

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        return {
            "target": path,
            "kind": kind,
            "introspectable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "parameters": [],
            "keywords": [],
            "required": [],
            "accepts_var_keyword": False,
        }

    parameters, keywords, required = [], [], []
    var_keyword = False
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            var_keyword = True
            continue
        has_default = param.default is not inspect.Parameter.empty
        parameters.append(
            {
                "name": name,
                "kind": param.kind.name,
                "has_default": has_default,
                "default": _safe(param.default) if has_default else None,
                "annotation": (None if param.annotation is inspect.Parameter.empty else str(param.annotation)),
            }
        )
        if param.kind is not inspect.Parameter.VAR_POSITIONAL:
            keywords.append(name)
        if not has_default and param.kind is not inspect.Parameter.VAR_POSITIONAL:
            required.append(name)

    return {
        "target": path,
        "kind": kind,
        "introspectable": True,
        "error": None,
        "parameter_count": len(parameters),
        "parameters": parameters,
        "keywords": keywords,
        # A required keyword cannot be rescued by dropping it. add_cloth_mesh
        # has no default for `vel`, so omitting it does not degrade - it
        # raises. Worth separating from the merely-accepted ones.
        "required": required,
        # Anything taking **kwargs swallows everything, so nothing is dropped
        # and nothing is validated either. Worth knowing before trusting a
        # silent success.
        "accepts_var_keyword": var_keyword,
        "doc": (inspect.getdoc(obj) or "").split("\n\n")[0][:400] or None,
    }


def signatures(newton, targets, accepts=()):
    """Describe each target, and optionally answer 'would this keyword land?'.

    `accepts` is the question that actually matters at a call site: not what
    the signature is, but whether the keyword you are about to pass would
    reach the engine or be silently discarded.
    """
    probe = [k.strip() for k in accepts if k and k.strip()]
    results = []
    for path in targets:
        try:
            obj, walked = resolve(newton, path)
        except AttributeError as exc:
            results.append({"target": path, "resolved": False, "error": str(exc)})
            continue
        entry = describe(obj, walked)
        entry["resolved"] = True
        if probe:
            accepted = set(entry["keywords"])
            entry["accepts"] = {name: (name in accepted or entry["accepts_var_keyword"]) for name in probe}
            entry["would_be_dropped"] = [
                name for name in probe if name not in accepted and not entry["accepts_var_keyword"]
            ]
        results.append(entry)
    return results
