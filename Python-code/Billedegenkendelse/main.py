import os
import subprocess
from pathlib import Path
import zipfile
import yaml

def prepare_data(project_root: Path):
    zip_path = project_root / "data" / "data.zip"
    extract_to = project_root / "data" / "custom_data"

    extract_to.mkdir(parents=True, exist_ok=True)

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)

    # Ensure train/val split runs with CWD at project root so it writes to project_root/data/...
    subprocess.run(
        ["python", "train_val_split.py", "--datapath", str(extract_to), "--train_pct", "0.8"],
        cwd=project_root,
        check=True
    )

def create_data_yaml(project_root: Path):
    classes_txt = project_root / "data" / "custom_data" / "classes.txt"
    if not classes_txt.exists():
        print(f"classes.txt file not found! Create it at: {classes_txt}")
        return None

    classes = [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    data_yaml_path = project_root / "data" / "data.yaml"

    # Use absolute path for robustness on Windows
    data = {
        "path": str((project_root / "data").resolve()),
        "train": "train/images",
        "val": "validation/images",
        "nc": len(classes),
        "names": classes
    }

    data_yaml_path.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
    print(f"\nCreated config file at: {data_yaml_path}\n")
    print("--- data.yaml ---")
    print(data_yaml_path.read_text(encoding="utf-8"))
    print("-----------------\n")
    return data_yaml_path

if __name__ == "__main__":
    project_root = Path(__file__).parent.resolve()  # ...\Billedegenkendelse

    # 1) Prepare data (unzips & splits)
    prepare_data(project_root)

    # 2) Create data.yaml (requires classes.txt)
    data_yaml = create_data_yaml(project_root)
    if not data_yaml:
        raise SystemExit(1)

    # 3) Train — point YOLO to the absolute path of data.yaml
    subprocess.run(
        ["yolo", "detect", "train",
         f"data={str(data_yaml)}",
         "model=yolo11s.pt",
         "epochs=60",
         "imgsz=480"],
        check=True
    )
