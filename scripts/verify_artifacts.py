"""Check distribution contents, metadata, integrity, and reproducibility.

The optional rebuild uses uv to resolve build requirements from configured read
indexes. Reports and temporary rebuild files stay in the requested local output
directory. This script does not upload artifacts.
"""

from __future__ import annotations

import argparse
import base64
import csv
import email.policy
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10's build tooling supplies tomli.
    import tomli as tomllib  # type: ignore[no-redef]

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

FORBIDDEN_PARTS = {
    ".venv",
    "venv",
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".hypothesis",
    ".tox",
    ".nox",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
}
FORBIDDEN_NAMES = {".pypirc", ".netrc", ".DS_Store", "Thumbs.db"}
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bpypi-[A-Za-z0-9_-]{40,}\b"),
]


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    require(
        not path.is_absolute() and ".." not in path.parts and "\\" not in name,
        "An archive contains an unsafe member path.",
    )
    return path


def wheel_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "A wheel contains duplicate member paths.")
        result = {}
        for member in archive.infolist():
            safe_member(member.filename)
            require(not member.is_dir(), "The wheel contains an unexpected directory entry.")
            require((member.external_attr >> 16) & 0o170000 != 0o120000, "A wheel contains a symbolic link.")
            result[member.filename] = archive.read(member)
        return result


def sdist_members(path: Path) -> tuple[str, dict[str, bytes]]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), "An sdist contains duplicate member paths.")
        prefixes = {safe_member(name).parts[0] for name in names}
        require(len(prefixes) == 1, "The sdist must have one root directory.")
        prefix = prefixes.pop()
        result = {}
        for member in members:
            require(member.isfile() or member.isdir(), "An sdist contains a link or special file.")
            if member.isdir():
                continue
            name = str(PurePosixPath(member.name).relative_to(prefix))
            resource = archive.extractfile(member)
            require(resource is not None, "An sdist member cannot be read.")
            assert resource is not None
            result[name] = resource.read()
        return prefix, result


def inspect_private_content(members: dict[str, bytes]) -> None:
    for name, data in members.items():
        path = safe_member(name)
        require(
            not (set(path.parts) & FORBIDDEN_PARTS) and path.name not in FORBIDDEN_NAMES,
            "An artifact contains private/cache content: " + name,
        )
        require(
            not any(part.startswith(".env") for part in path.parts),
            "An artifact contains local environment configuration: " + name,
        )
        require(
            path.suffix.lower() not in {".pem", ".key", ".p12", ".pfx", ".pyc", ".pyo"},
            "An artifact contains a forbidden file type: " + name,
        )
        require(
            not any(pattern.search(data) for pattern in SECRET_PATTERNS),
            "An artifact matched a credential/private-key pattern: " + name,
        )


def expected_sdist(root: Path, settings: dict[str, Any]) -> dict[str, bytes]:
    result = {}
    # Hatchling also retains the reviewed VCS exclusion file in the sdist.
    if (root / ".gitignore").is_file():
        result[".gitignore"] = (root / ".gitignore").read_bytes()
    exclusions = [str(value).lstrip("/") for value in settings.get("exclude", [])]
    for item in settings["include"]:
        source = root / str(item).lstrip("/")
        require(source.exists(), "An explicitly included sdist source is missing: " + str(item))
        paths = source.rglob("*") if source.is_dir() else [source]
        for path in paths:
            if not path.is_file():
                continue
            name = path.relative_to(root).as_posix()
            parts = set(PurePosixPath(name).parts)
            if parts & {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
                continue
            if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
                continue
            if any(
                fnmatch.fnmatchcase(name, pattern) or name.startswith(pattern.rstrip("/") + "/")
                for pattern in exclusions
            ):
                continue
            result[name] = path.read_bytes()
    return result


def inspect_metadata(data: bytes, project: dict[str, Any], readme: bytes) -> dict[str, Any]:
    metadata = BytesParser(policy=email.policy.default).parsebytes(data)
    require(metadata["Name"] == project["name"], "Distribution name differs from source metadata.")
    require(metadata["Version"] == project["version"], "Distribution version differs from source metadata.")
    require(
        SpecifierSet(metadata["Requires-Python"]) == SpecifierSet(project["requires-python"]),
        "Python range differs from source metadata.",
    )
    require(metadata["License-Expression"] == "MIT", "The license expression is not MIT.")
    require(
        "Private :: Do Not Upload" in metadata.get_all("Classifier", []),
        "This local candidate is missing the private upload blocker.",
    )
    requirements = sorted(str(Requirement(value)) for value in metadata.get_all("Requires-Dist", []))
    expected = sorted(str(Requirement(value)) for value in project["dependencies"])
    require(requirements == expected, "Wheel/sdist dependencies differ from declared runtime dependencies.")
    names = {Requirement(value).name.lower().replace("_", "-") for value in requirements}
    require(
        names == {"strands-agents", "httpx", "jsonschema"},
        "Unexpected runtime framework, SDK, or developer dependency.",
    )
    require(metadata["Description-Content-Type"] == "text/markdown", "README is not declared Markdown.")
    require(
        data.replace(b"\r\n", b"\n").partition(b"\n\n")[2].strip() == readme.strip(),
        "Distributed README differs from the reviewed source.",
    )
    return {
        "name": metadata["Name"],
        "version": metadata["Version"],
        "requires_python": metadata["Requires-Python"],
        "license_expression": metadata["License-Expression"],
        "requires_dist": requirements,
        "private_classifier": True,
    }


def inspect_record(members: dict[str, bytes], dist_info: str) -> None:
    record = dist_info + "/RECORD"
    rows = list(csv.reader(io.StringIO(members[record].decode())))
    require(len(rows) == len(members), "Wheel RECORD has a different member count.")
    require({row[0] for row in rows} == set(members), "Wheel RECORD does not enumerate exact members.")
    for name, digest, size in rows:
        if name == record:
            require(digest == size == "", "Wheel RECORD must not hash itself.")
            continue
        expected = base64.urlsafe_b64encode(hashlib.sha256(members[name]).digest()).decode().rstrip("=")
        require(
            digest == "sha256=" + expected and size == str(len(members[name])),
            "Wheel RECORD hash/size mismatch: " + name,
        )


def run_rebuild(root: Path, sdist: Path, wheel: Path, output: Path, epoch: str) -> dict[str, Any]:
    directory = output / "sdist-rebuild"
    directory.mkdir(parents=True, exist_ok=True)
    command = ["uv", "build", "--wheel", str(sdist.resolve()), "--out-dir", str(directory.resolve())]
    process = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "SOURCE_DATE_EPOCH": epoch},
        text=True,
        capture_output=True,
        check=False,
    )
    (output / "sdist-rebuild.log").write_text(process.stdout + process.stderr)
    require(process.returncode == 0, "Sdist rebuild failed; see the local rebuild log.")
    rebuilt = directory / wheel.name
    require(rebuilt.exists(), "The sdist rebuild did not produce the expected wheel.")
    original_members = wheel_members(wheel)
    rebuilt_members = wheel_members(rebuilt)
    require(original_members == rebuilt_members, "Sdist-rebuilt wheel differs from the source-built wheel.")
    require(wheel.read_bytes() == rebuilt.read_bytes(), "Normalized contents match, but wheel bytes differ.")
    return {
        "source_date_epoch": epoch,
        "rebuilt_wheel": str(rebuilt.resolve()),
        "sha256": sha256(rebuilt.read_bytes()),
        "member_equality": True,
        "byte_equality": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifacts", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path, default=Path("build/verification"))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--epoch",
        default=os.environ.get("SOURCE_DATE_EPOCH", "1788796800"),
        help="Rebuild timestamp; defaults to SOURCE_DATE_EPOCH or 1788796800.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    settings = tomllib.loads((root / "pyproject.toml").read_text())
    project = settings["project"]
    stem = project["name"].replace("-", "_") + "-" + project["version"]
    dist_info = stem + ".dist-info"
    wheel = args.artifacts / (stem + "-py3-none-any.whl")
    sdist = args.artifacts / (stem + ".tar.gz")
    require(wheel.is_file() and sdist.is_file(), "Build the expected wheel and sdist before verification.")
    wheels = wheel_members(wheel)
    prefix, source = sdist_members(sdist)
    require(prefix == stem, "The sdist root name differs from the reviewed name/version.")
    inspect_private_content(wheels)
    inspect_private_content(source)
    package_files = {
        path.relative_to(root / "src").as_posix(): path.read_bytes()
        for path in (root / "src/strands_neuraltrust").rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
    }
    expected_wheel = set(package_files) | {
        dist_info + "/" + name for name in ["METADATA", "WHEEL", "RECORD", "licenses/LICENSE"]
    }
    require(
        set(wheels) == expected_wheel, "The wheel manifest differs from the exact expected package files."
    )
    require(
        "strands_neuraltrust/__init__.py" in wheels and "strands_neuraltrust/py.typed" in wheels,
        "The wheel is missing its importable package or typing marker.",
    )
    for name, data in package_files.items():
        require(wheels[name] == data, "Wheel source differs from checkout: " + name)
    license_data = (root / "LICENSE").read_bytes()
    require(
        b"MIT License" in license_data and b"Permission is hereby granted" in license_data,
        "The source license is incomplete or unexpected.",
    )
    require(wheels[dist_info + "/licenses/LICENSE"] == license_data, "The wheel license differs from source.")
    expected_source = expected_sdist(root, settings["tool"]["hatch"]["build"]["targets"]["sdist"])
    require(
        set(source) == set(expected_source) | {"PKG-INFO"}, "The sdist manifest differs from curated source."
    )
    for name, data in expected_source.items():
        require(source[name] == data, "Sdist source differs from checkout: " + name)
    metadata = inspect_metadata(wheels[dist_info + "/METADATA"], project, (root / "README.md").read_bytes())
    require(
        inspect_metadata(source["PKG-INFO"], project, source["README.md"]) == metadata,
        "Sdist and wheel metadata differ.",
    )
    require(b"Root-Is-Purelib: true" in wheels[dist_info + "/WHEEL"], "Wheel is not a pure-Python artifact.")
    require(b"Tag: py3-none-any" in wheels[dist_info + "/WHEEL"], "Wheel tag differs from expected artifact.")
    inspect_record(wheels, dist_info)
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "metadata": metadata,
        "checks": {
            "exact_manifests": True,
            "source_equality": True,
            "license": True,
            "typed": True,
            "runtime_dependencies_only": True,
            "private_classifier": True,
            "record_hashes": True,
            "forbidden_content_absent": True,
            "credential_pattern_scan": True,
        },
        "artifacts": [
            {
                "file": str(path.resolve()),
                "sha256": sha256(path.read_bytes()),
                "size": path.stat().st_size,
                "members": sorted(members),
            }
            for path, members in [(wheel, wheels), (sdist, source)]
        ],
    }
    if args.rebuild:
        report["rebuild"] = run_rebuild(root, sdist, wheel, args.output, args.epoch)
    destination = args.output / "artifact-report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": "passed",
                "report": str(destination.resolve()),
                "wheel_sha256": report["artifacts"][0]["sha256"],
                "sdist_sha256": report["artifacts"][1]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
