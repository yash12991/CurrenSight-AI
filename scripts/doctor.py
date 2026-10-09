"""Run lightweight CurrenSight AI environment and project-integrity diagnostics."""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPENDENCIES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "numpy": "numpy",
    "opencv": "cv2",
    "pydantic": "pydantic",
    "torch": "torch",
    "torchvision": "torchvision",
    "pillow": "PIL",
    "multipart": "multipart",
}
MODEL_FILES = (
    "currency_classifier.pkl",
    "currency_classifier_pytorch.pth",
    "currency_authenticity_pytorch.pth",
)


def inspect_project():
    python_version = platform.python_version()
    python_supported = sys.version_info.major == 3 and 11 <= sys.version_info.minor <= 14
    dependencies = {
        name: importlib.util.find_spec(module_name) is not None
        for name, module_name in DEPENDENCIES.items()
    }

    model_dir = PROJECT_ROOT / "backend" / "models"
    models = {}
    for filename in MODEL_FILES:
        path = model_dir / filename
        models[filename] = {
            "present": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }

    smoke_manifest = PROJECT_ROOT / "tests" / "smoke_images.json"
    smoke_cases = []
    if smoke_manifest.is_file():
        try:
            smoke_cases = json.loads(smoke_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            smoke_cases = []
    missing_smoke_images = [
        case.get("path", "<missing path>")
        for case in smoke_cases
        if not (PROJECT_ROOT / case.get("path", "")).is_file()
    ]

    required_paths = (
        "backend/main.py",
        "backend/requirements.txt",
        "frontend/index.html",
        "frontend/app.js",
        "frontend/index.css",
        "README.md",
    )
    missing_project_files = [path for path in required_paths if not (PROJECT_ROOT / path).is_file()]

    ready = (
        python_supported
        and all(dependencies.values())
        and models["currency_classifier_pytorch.pth"]["present"]
        and not missing_project_files
        and len(smoke_cases) == 7
        and not missing_smoke_images
    )
    return {
        "ready": ready,
        "python": {"version": python_version, "supported": python_supported},
        "dependencies": dependencies,
        "models": models,
        "smoke_images": {
            "cases": len(smoke_cases),
            "missing": missing_smoke_images,
        },
        "missing_project_files": missing_project_files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    report = inspect_project()

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        state = "READY" if report["ready"] else "NOT READY"
        print(f"CurrenSight AI doctor: {state}")
        print(f"Python {report['python']['version']} (supported: {report['python']['supported']})")
        for name, available in report["dependencies"].items():
            print(f"  {'OK' if available else 'MISSING':7} {name}")
        for name, details in report["models"].items():
            print(f"  {'OK' if details['present'] else 'MISSING':7} {name} ({details['bytes']} bytes)")
        if report["missing_project_files"]:
            print("Missing project files:", ", ".join(report["missing_project_files"]))
        if report["smoke_images"]["missing"]:
            print("Missing smoke images:", ", ".join(report["smoke_images"]["missing"]))

    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
