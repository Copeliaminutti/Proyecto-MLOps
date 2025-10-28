from pathlib import Path
import yaml, json
def read_yaml(path: Path):
    with open(path,'r') as f: return yaml.safe_load(f)
def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w') as f: json.dump(obj,f,indent=2)
