"""
Aerial Imagery Processor with hierarchical aggregation.

Fetches aerial images from PDOK, encodes with DINOv3 at markov blanket resolution 10.

Simple straightforward image embeddings from DINOv3 for subsequent downstream use later.

"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import h3
import torch
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

from modalities.base import ModalityProcessor
from .pdok_client import PDOKClient, ImageTile
from .dinov3_encoder import DINOv3Encoder, EncodingResult

logger = logging.getLogger(__name__)
