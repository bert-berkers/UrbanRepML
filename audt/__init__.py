"""
Autonomous Urban Digital Twin (AUDT)

This package implements the theoretical framework of Dynamic Liveability and Active Inference
for urban systems, integrating Alicia Juarrero's theory of constraints with hierarchical
representation learning.

Core Components:
- Hierarchical Active Inference Engine (The "Engine") 
- H3-based Hierarchical U-Net (The "Transmission")
- Niche Construction Dynamics
- Free Energy Principle optimization
- Accessibility-constrained learning

Theoretical Foundation:
Based on the integration of:
1. Free Energy Principle (Friston) - Fundamental organizing principle
2. Theory of Constraints (Juarrero) - Context-dependent system coherence  
3. Hierarchical Active Inference - Multi-scale temporal dynamics
4. Urban Representation Learning - Spatial pattern recognition
"""

from .engine import ActiveInferenceEngine
from .transmission import HierarchicalUNet
from .constraints import ConstraintSpace
from .niche_construction import NicheConstructor

__all__ = [
    'ActiveInferenceEngine',
    'HierarchicalUNet', 
    'ConstraintSpace',
    'NicheConstructor'
]

__version__ = "0.1.0"