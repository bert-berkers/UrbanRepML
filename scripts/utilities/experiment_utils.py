"""
Utilities for managing experiments in the UrbanRepML project.
"""

import os
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

class ExperimentManager:
    """Manages experiment folders and configurations."""
    
    def __init__(self, base_path: str = "experiments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def create_experiment(self, 
                         name: str, 
                         description: str = "",
                         config: Optional[Dict] = None) -> Path:
        """Create a new experiment folder with standard structure."""
        
        # Generate experiment ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        exp_id = f"{name}_{timestamp}"
        
        # Create experiment directory
        exp_path = self.base_path / exp_id
        if exp_path.exists():
            # Add time suffix if folder exists
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            exp_id = f"{name}_{timestamp}"
            exp_path = self.base_path / exp_id
        
        # Create directory structure
        subdirs = [
            "data",
            "analysis", 
            "plots",
            "logs",
            "models"
        ]
        
        for subdir in subdirs:
            (exp_path / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create config file
        if config is None:
            config = self._load_template_config()
        
        config.update({
            "experiment": {
                "name": name,
                "description": description,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "id": exp_id
            }
        })
        
        config_path = exp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Create README
        self._create_readme(exp_path, name, description)
        
        print(f"Created experiment: {exp_id}")
        print(f"Path: {exp_path}")
        
        return exp_path
    
    def list_experiments(self) -> list:
        """List all experiments."""
        experiments = []
        for path in self.base_path.iterdir():
            if path.is_dir():
                config_path = path / "config.yaml"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    experiments.append({
                        "id": path.name,
                        "name": config.get("experiment", {}).get("name", ""),
                        "date": config.get("experiment", {}).get("date", ""),
                        "path": str(path)
                    })
        return sorted(experiments, key=lambda x: x["date"], reverse=True)
    
    def get_experiment_path(self, exp_id: str) -> Path:
        """Get path to specific experiment."""
        return self.base_path / exp_id
    
    def _load_template_config(self) -> Dict:
        """Load template configuration."""
        template_path = Path("templates") / "experiment_config_template.yaml"
        if template_path.exists():
            with open(template_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _create_readme(self, exp_path: Path, name: str, description: str):
        """Create README file for experiment."""
        readme_content = f"""# {name}

**Date:** {datetime.now().strftime("%B %d, %Y")}  
**Experiment ID:** {exp_path.name}

## Objective
{description}

## Structure
```
├── config.yaml              # Experiment configuration
├── data/                     # Experiment-specific data
├── analysis/                 # Analysis results and metrics  
├── plots/                    # All visualizations
├── logs/                     # Processing logs
└── models/                   # Trained models and checkpoints
```

## Usage
Load experiment configuration:
```python
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

## Notes
- All outputs should be saved within this experiment folder
- Update this README with key findings and results
"""
        
        with open(exp_path / "README.md", 'w') as f:
            f.write(readme_content)

def get_experiment_config(exp_path: str) -> Dict:
    """Load experiment configuration."""
    config_path = Path(exp_path) / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(exp_path: str, results: Dict, filename: str = "results.yaml"):
    """Save results to experiment folder."""
    results_path = Path(exp_path) / "analysis" / filename
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, indent=2)