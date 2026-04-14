# synthetic_structured_data.py

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import xml.sax.saxutils as saxutils
from collections import defaultdict
from pathlib import Path
from typing import Any


FIELDS = ["name", "age", "email", "city", "country", "occupation", "phone", "score"]
FORMATS = ["json", "yaml", "xml", "csv", "toml"]

FORMAT_PREFIXES = {
    "json": [
        "Please format as JSON:",
        "JSON;",
        "Output JSON format.",
        "Give me JSON for:",
    ],
    "yaml": [
        "Please format as YAML:",
        "YAML —",
        "Output as YAML:",
        "Convert to YAML:",
    ],
    "xml": [
        "Please format as XML:",
        "XML format.",
        "Output XML format.",
        "Convert to XML:",
    ],
    "csv": [
        "Format this as CSV —",
        "CSV;",
        "Generate CSV with the following:",
        "Output as CSV:",
    ],
    "toml": [
        "Please format as TOML:",
        "TOML —",
        "Transform into TOML:",
        "Output TOML.",
    ],
}

CLAUSE_STYLES = [
    "{field} is {value}",
    "{field}: {value}",
    "{field} = {value}",
    "the {field} is {value}",
    "the {field}: {value}",
]

JOINERS = {
    "default": [", ", "; ", " and ", ", and ", "; and "],
    "xml": [". ", "; ", ". "],
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_pools(source_path: str | Path | None = None) -> dict[str, list[Any]]:
    """
    Build value pools from the existing dataset.
    If source_path is None or missing, falls back to hand-written pools.
    """
    pools: dict[str, set[Any]] = defaultdict(set)

    if source_path is not None and Path(source_path).exists():
        for row in load_jsonl(source_path):
            fields = row.get("fields", {})
            for k, v in fields.items():
                pools[k].add(v)

    # Fallback values in case some pools are empty.
    fallback = {
        "name": [
            "Alice Smith", "Bob Jones", "Carol Martin", "David Brown", "Eve Green",
            "Frank Wilson", "Grace Lee", "Hannah Taylor", "Ivan Petrov", "Julia Adams",
        ],
        "age": list(range(18, 81)),
        "email": [
            "alice@gmail.com", "bob@example.com", "carol@outlook.com", "david@proton.me",
            "eve@icloud.com", "frank@yahoo.com", "grace@tutanota.com", "hannah@pm.me",
        ],
        "city": [
            "Berlin", "Tokyo", "Astana", "Almaty", "Kyiv", "London", "Paris", "Rome",
            "Vienna", "Oslo", "Beijing", "Bogota", "Helsinki", "Baghdad", "Nairobi",
        ],
        "country": [
            "Germany", "Japan", "Kazakhstan", "Ukraine", "United Kingdom", "France",
            "Italy", "Norway", "China", "Brazil", "India", "Mexico", "Austria", "Thailand",
        ],
        "occupation": [
            "Engineer", "Analyst", "Teacher", "Doctor", "Manager", "Scientist",
            "Architect", "Chef", "Lawyer", "Firefighter", "Mathematician", "Artist",
        ],
        "phone": [
            "+1-555-012-3456", "+44-020-7946-0958", "+49-030-1234-5678",
            "+52-680-590-0808", "+55-446-475-5067", "+7-701-234-5678",
        ],
        "score": [round(x / 10, 1) for x in range(0, 1000)],
    }

    out: dict[str, list[Any]] = {}
    for field in FIELDS:
        vals = list(pools.get(field, []))
        if not vals:
            vals = fallback[field]
        out[field] = vals

    return out


def random_value(field: str, pools: dict[str, list[Any]], rng: random.Random) -> Any:
    if field == "age":
        # Prefer observed ages, but keep a fallback range.
        vals = pools.get("age", [])
        if vals:
            return int(rng.choice(vals))
        return rng.randint(18, 90)

    if field == "score":
        vals = pools.get("score", [])
        if vals:
            return rng.choice(vals)
        return round(rng.uniform(0, 100), 1)

    vals = pools.get(field, [])
    if vals:
        return rng.choice(vals)
    return f"{field}_{rng.randint(1000, 9999)}"


def make_field_subset(rng: random.Random, min_fields: int = 2, max_fields: int = 6) -> list[str]:
    k = rng.randint(min_fields, max_fields)
    return rng.sample(FIELDS, k=k)


def render_clause(field: str, value: Any, rng: random.Random) -> str:
    field_text = field
    if rng.random() < 0.2:
        field_text = field.upper()
    elif rng.random() < 0.2:
        field_text = field.capitalize()

    style = rng.choice(CLAUSE_STYLES)
    return style.format(field=field_text, value=value)


def join_clauses(clauses: list[str], format_name: str, rng: random.Random) -> str:
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]

    joiners = JOINERS["xml"] if format_name == "xml" else JOINERS["default"]
    text = clauses[0]
    for clause in clauses[1:]:
        text += rng.choice(joiners) + clause
    return text


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(value)

    s = str(value)
    # Bare YAML scalars if they are simple enough.
    if re.fullmatch(r"[A-Za-z0-9_@.+\-]+", s):
        return s
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(value)

    s = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def render_output(format_name: str, fields: dict[str, Any]) -> str:
    if format_name == "json":
        return json.dumps(fields, ensure_ascii=False)

    if format_name == "yaml":
        lines = [f"{k}: {yaml_scalar(v)}" for k, v in fields.items()]
        return "\n".join(lines)

    if format_name == "xml":
        parts = ["<record>"]
        for k, v in fields.items():
            parts.append(f"<{k}>{saxutils.escape(str(v))}</{k}>")
        parts.append("</record>")
        return "".join(parts)

    if format_name == "csv":
        buf = io.StringIO(newline="")
        writer = csv.writer(buf, lineterminator="\r\n")
        writer.writerow(list(fields.keys()))
        writer.writerow(list(fields.values()))
        return buf.getvalue().rstrip("\n")

    if format_name == "toml":
        lines = [f"{k} = {toml_scalar(v)}" for k, v in fields.items()]
        return "\n".join(lines)

    raise ValueError(f"Unknown format: {format_name}")


def render_input(format_name: str, fields: dict[str, Any], rng: random.Random) -> str:
    prefix = rng.choice(FORMAT_PREFIXES[format_name])

    clauses = [render_clause(k, v, rng) for k, v in fields.items()]
    body = join_clauses(clauses, format_name, rng)

    # Keep the style close to your current dataset.
    if format_name in {"json", "csv", "toml"}:
        if prefix.endswith(":") or prefix.endswith("—") or prefix.endswith(";"):
            return f"{prefix} {body}"
        return f"{prefix} {body}"
    if format_name == "yaml":
        return f"{prefix} {body}"
    if format_name == "xml":
        return f"{prefix} {body}"
    return f"{prefix} {body}"


def make_fields(rng: random.Random, pools: dict[str, list[Any]], min_fields: int = 2, max_fields: int = 6) -> dict[str, Any]:
    subset = make_field_subset(rng, min_fields=min_fields, max_fields=max_fields)
    items = []
    for field in subset:
        items.append((field, random_value(field, pools, rng)))
    return dict(items)


def generate_sample(format_name: str, pools: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    fields = make_fields(rng, pools)
    inp = render_input(format_name, fields, rng)
    out = render_output(format_name, fields)
    return {
        "input": inp,
        "output": out,
        "format": format_name,
        "fields": fields,
    }


def generate_samples(
    total: int | None = None,
    format_name: str | None = None,
    format_counts: dict[str, int] | None = None,
    source_path: str | Path | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate synthetic samples.

    Use one of:
      - total + format_name
      - format_counts
      - total without format_name (uniform over all formats)
    """
    rng = random.Random(seed)
    pools = build_pools(source_path)

    samples: list[dict[str, Any]] = []

    if format_counts is not None:
        for fmt, n in format_counts.items():
            fmt = fmt.lower()
            if fmt not in FORMATS:
                raise ValueError(f"Unknown format: {fmt}")
            for _ in range(int(n)):
                samples.append(generate_sample(fmt, pools, rng))
        rng.shuffle(samples)
        return samples

    if total is None:
        raise ValueError("Provide either total or format_counts")

    if format_name is not None:
        fmt = format_name.lower()
        if fmt not in FORMATS:
            raise ValueError(f"Unknown format: {fmt}")
        for _ in range(total):
            samples.append(generate_sample(fmt, pools, rng))
        return samples

    # Uniform mix over formats.
    for _ in range(total):
        fmt = rng.choice(FORMATS)
        samples.append(generate_sample(fmt, pools, rng))

    return samples


def write_jsonl(samples: list[dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="train.jsonl", help="Existing train.jsonl to learn value pools from")
    parser.add_argument("--out", type=str, default="synthetic_train.jsonl")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--total", type=int, default=None, help="Total samples to generate")
    parser.add_argument("--format", type=str, default=None, help="Generate only one format")

    parser.add_argument("--json", type=int, default=0)
    parser.add_argument("--yaml", type=int, default=0)
    parser.add_argument("--xml", type=int, default=0)
    parser.add_argument("--csv", type=int, default=0)
    parser.add_argument("--toml", type=int, default=0)

    args = parser.parse_args()

    format_counts = {
        "json": args.json,
        "yaml": args.yaml,
        "xml": args.xml,
        "csv": args.csv,
        "toml": args.toml,
    }
    format_counts = {k: v for k, v in format_counts.items() if v > 0}

    if format_counts:
        samples = generate_samples(
            format_counts=format_counts,
            source_path=args.source,
            seed=args.seed,
        )
    else:
        samples = generate_samples(
            total=args.total or 1000,
            format_name=args.format,
            source_path=args.source,
            seed=args.seed,
        )

    write_jsonl(samples, args.out)
    print(f"Wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()