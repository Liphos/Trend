import yaml
from typing import Dict, Union, List
import torch
from model import model_by_name

def read_config(filename:str) -> Dict[str, Union[int, str, List[int]]]:
    """Read config from yaml

    Args:
        filename (str): the path to the yaml file

    Returns:
        Dict[str, Union[int, str, List[int]]]: the corresponding configs.
    """
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    config['device'] = "cuda" if torch.cuda.is_available() and config['device']=="cuda" else "cpu"
    config['model']["name"] = model_by_name(config['model']["name"])
    return config