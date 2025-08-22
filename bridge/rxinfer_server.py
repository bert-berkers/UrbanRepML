#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RxInfer Server Integration for GPU-Accelerated SRAI H3 Processing
Provides reactive inference capabilities for AlphaEarth TIFF to H3 conversion

Features:
- Reactive probabilistic inference using RxInfer patterns
- Real-time processing pipeline with GPU acceleration
- SRAI H3 operations at resolution 8
- Streaming data processing with backpressure
- Active inference for spatial predictions
"""

import asyncio
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import h3
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

# FastAPI for server interface
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our processors
import sys
sys.path.append(str(Path(__file__).parent.parent))
from experiments.del_norte_exploratory.scripts.srai_rioxarray_processor import SRAIRioxarrayProcessor
from experiments.del_norte_exploratory.scripts.pytorch_tiff_processor import PyTorchTiffProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============== Data Models ==============

@dataclass
class InferenceState:
    """Reactive inference state for H3 processing"""
    h3_beliefs: Dict[str, np.ndarray]  # H3 index -> belief distribution
    prediction_errors: List[float]      # Temporal prediction errors
    processing_rate: float               # Tiles per second
    gpu_memory_usage: float             # Current GPU memory usage
    timestamp: datetime


class ProcessingRequest(BaseModel):
    """Request model for processing AlphaEarth data"""
    source_dir: str = Field(..., description="Directory containing TIFF files")
    h3_resolution: int = Field(8, description="H3 resolution level")
    use_gpu: bool = Field(True, description="Enable GPU acceleration")
    batch_size: int = Field(10, description="Batch size for processing")
    stream_results: bool = Field(True, description="Stream results as they're processed")


class InferenceUpdate(BaseModel):
    """Real-time inference update"""
    h3_index: str
    embedding: List[float]
    confidence: float
    prediction_error: float
    timestamp: str


# ============== RxInfer Server ==============

class RxInferServer:
    """
    Reactive Inference Server for GPU-accelerated SRAI H3 processing
    Implements active inference principles for spatial data processing
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize RxInfer server with configuration"""
        self.config = self.load_config(config_path)
        self.app = FastAPI(title="RxInfer SRAI H3 Server", version="1.0.0")
        self.setup_routes()
        
        # Processing components
        self.srai_processor = None
        self.pytorch_processor = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Inference state
        self.current_state = InferenceState(
            h3_beliefs={},
            prediction_errors=[],
            processing_rate=0.0,
            gpu_memory_usage=0.0,
            timestamp=datetime.now()
        )
        
        # Active connections for streaming
        self.active_connections: List[WebSocket] = []
        
        # Processing queue for reactive pipeline
        self.processing_queue = asyncio.Queue(maxsize=100)
        self.result_queue = asyncio.Queue()
        
        # GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.device('cuda')
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.warning("No GPU detected, using CPU")
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration for processing"""
        import yaml
        config_file = Path(config_path)
        if not config_file.exists():
            # Use default configuration
            config = {
                'data': {
                    'h3_resolution': 8,
                    'source_dir': 'data/alphaearth_tiffs',
                    'pattern': '*.tif'
                },
                'processing': {
                    'batch_size': 10,
                    'min_pixels_per_hex': 10,
                    'chunk_size': 2048
                },
                'hardware': {
                    'max_cores': 12,
                    'gpu_memory_gb': 20,
                    'use_mixed_precision': True
                },
                'output': {
                    'log_level': 'INFO'
                }
            }
        else:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        
        return config
    
    def setup_routes(self):
        """Setup FastAPI routes for the server"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "RxInfer SRAI H3 Server",
                "status": "running",
                "gpu_available": self.gpu_available,
                "current_state": {
                    "h3_cells_processed": len(self.current_state.h3_beliefs),
                    "processing_rate": self.current_state.processing_rate,
                    "gpu_memory_usage": self.current_state.gpu_memory_usage
                }
            }
        
        @self.app.post("/process")
        async def process_tiles(request: ProcessingRequest, background_tasks: BackgroundTasks):
            """Start processing AlphaEarth tiles to H3"""
            try:
                # Update configuration
                self.config['data']['source_dir'] = request.source_dir
                self.config['data']['h3_resolution'] = request.h3_resolution
                self.config['processing']['batch_size'] = request.batch_size
                
                # Start background processing
                background_tasks.add_task(
                    self.process_tiles_reactive,
                    request.use_gpu,
                    request.stream_results
                )
                
                return JSONResponse({
                    "status": "processing_started",
                    "config": {
                        "source_dir": request.source_dir,
                        "h3_resolution": request.h3_resolution,
                        "use_gpu": request.use_gpu,
                        "batch_size": request.batch_size
                    }
                })
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stream")
        async def stream_results():
            """Stream processing results as Server-Sent Events"""
            async def event_generator():
                while True:
                    if not self.result_queue.empty():
                        result = await self.result_queue.get()
                        yield f"data: {json.dumps(result)}\n\n"
                    await asyncio.sleep(0.1)
            
            return StreamingResponse(event_generator(), media_type="text/event-stream")
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    # Send updates from result queue
                    if not self.result_queue.empty():
                        result = await self.result_queue.get()
                        await websocket.send_json(result)
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.active_connections.remove(websocket)
        
        @self.app.get("/state")
        async def get_inference_state():
            """Get current inference state"""
            return {
                "h3_cells": len(self.current_state.h3_beliefs),
                "avg_prediction_error": np.mean(self.current_state.prediction_errors) if self.current_state.prediction_errors else 0,
                "processing_rate": self.current_state.processing_rate,
                "gpu_memory_usage": self.current_state.gpu_memory_usage,
                "timestamp": self.current_state.timestamp.isoformat()
            }
        
        @self.app.post("/infer")
        async def active_inference(h3_index: str):
            """Perform active inference on specific H3 cell"""
            if h3_index not in self.current_state.h3_beliefs:
                raise HTTPException(status_code=404, detail=f"H3 cell {h3_index} not found")
            
            belief = self.current_state.h3_beliefs[h3_index]
            
            # Perform inference update (simplified active inference)
            prediction = self.generate_prediction(belief)
            error = self.calculate_prediction_error(belief, prediction)
            
            # Update belief
            updated_belief = self.update_belief(belief, prediction, error)
            self.current_state.h3_beliefs[h3_index] = updated_belief
            
            return {
                "h3_index": h3_index,
                "prediction_error": float(error),
                "belief_entropy": float(self.calculate_entropy(updated_belief)),
                "confidence": float(1.0 - error)
            }
    
    async def process_tiles_reactive(self, use_gpu: bool = True, stream_results: bool = True):
        """
        Reactive processing pipeline for AlphaEarth tiles
        Implements backpressure and streaming results
        """
        try:
            # Initialize appropriate processor
            if use_gpu and self.gpu_available:
                logger.info("Initializing PyTorch GPU processor")
                self.pytorch_processor = PyTorchTiffProcessor(self.config)
                processor = self.pytorch_processor
            else:
                logger.info("Initializing SRAI processor")
                self.srai_processor = SRAIRioxarrayProcessor(self.config)
                processor = self.srai_processor
            
            # Get files to process
            tiff_files = processor.get_tiff_files()
            total_files = len(tiff_files)
            logger.info(f"Starting reactive processing of {total_files} files")
            
            # Process files in batches with reactive streaming
            processed_count = 0
            start_time = datetime.now()
            
            for batch_start in range(0, total_files, self.config['processing']['batch_size']):
                batch_end = min(batch_start + self.config['processing']['batch_size'], total_files)
                batch_files = tiff_files[batch_start:batch_end]
                
                # Process batch asynchronously
                batch_results = await self.process_batch_async(processor, batch_files)
                
                # Update inference state and stream results
                for result in batch_results:
                    if result is not None and not result.empty:
                        await self.update_inference_state(result)
                        
                        if stream_results:
                            await self.stream_batch_results(result)
                        
                        processed_count += 1
                
                # Update processing rate
                elapsed = (datetime.now() - start_time).total_seconds()
                self.current_state.processing_rate = processed_count / elapsed if elapsed > 0 else 0
                
                # Update GPU memory usage
                if self.gpu_available:
                    self.current_state.gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3
                
                # Broadcast progress to WebSocket clients
                await self.broadcast_progress(processed_count, total_files)
            
            logger.info(f"Reactive processing complete: {processed_count} tiles processed")
            
        except Exception as e:
            logger.error(f"Error in reactive processing: {e}")
            raise
    
    async def process_batch_async(self, processor, batch_files: List[Path]) -> List[pd.DataFrame]:
        """Process batch of files asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Use thread pool for CPU-bound processing
        tasks = []
        for file_path in batch_files:
            if hasattr(processor, 'process_tile_gpu'):
                task = loop.run_in_executor(self.executor, processor.process_tile_gpu, file_path)
            else:
                task = loop.run_in_executor(self.executor, self.process_single_file, processor, file_path)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    def process_single_file(self, processor, file_path: Path) -> Optional[pd.DataFrame]:
        """Process single file with appropriate processor"""
        try:
            if hasattr(processor, 'load_tiff_optimized'):
                # SRAI processor
                da = processor.load_tiff_optimized(file_path)
                if da is not None:
                    return processor.process_dataarray_to_h3(da, file_path)
            return None
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    async def update_inference_state(self, result_df: pd.DataFrame):
        """Update reactive inference state with new results"""
        # Extract H3 embeddings
        band_cols = [f'band_{i:02d}' for i in range(64)]
        
        for idx, row in result_df.iterrows():
            h3_idx = row.get('h3_index', idx)
            
            # Extract embedding
            embedding = np.array([row.get(col, 0) for col in band_cols if col in row])
            
            # Update belief state
            if h3_idx in self.current_state.h3_beliefs:
                # Bayesian update with new observation
                prior = self.current_state.h3_beliefs[h3_idx]
                posterior = self.bayesian_update(prior, embedding)
                self.current_state.h3_beliefs[h3_idx] = posterior
                
                # Calculate prediction error
                error = np.linalg.norm(posterior - prior)
                self.current_state.prediction_errors.append(error)
            else:
                # Initialize belief
                self.current_state.h3_beliefs[h3_idx] = embedding
        
        self.current_state.timestamp = datetime.now()
    
    async def stream_batch_results(self, result_df: pd.DataFrame):
        """Stream batch results to connected clients"""
        band_cols = [f'band_{i:02d}' for i in range(64)]
        
        for idx, row in result_df.iterrows():
            h3_idx = row.get('h3_index', idx)
            embedding = [float(row.get(col, 0)) for col in band_cols if col in row]
            
            # Calculate confidence based on embedding magnitude
            confidence = float(np.linalg.norm(embedding) / (np.sqrt(len(embedding)) * 100))
            confidence = min(1.0, confidence)  # Normalize to [0, 1]
            
            # Get prediction error for this cell
            error = 0.0
            if h3_idx in self.current_state.h3_beliefs:
                prior = self.current_state.h3_beliefs[h3_idx]
                error = float(np.linalg.norm(np.array(embedding) - prior))
            
            update = {
                "h3_index": str(h3_idx),
                "embedding": embedding[:10],  # Send first 10 dims for efficiency
                "confidence": confidence,
                "prediction_error": error,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to result queue for streaming
            await self.result_queue.put(update)
    
    async def broadcast_progress(self, processed: int, total: int):
        """Broadcast processing progress to WebSocket clients"""
        progress = {
            "type": "progress",
            "processed": processed,
            "total": total,
            "percentage": (processed / total * 100) if total > 0 else 0,
            "processing_rate": self.current_state.processing_rate,
            "gpu_memory_usage": self.current_state.gpu_memory_usage
        }
        
        # Send to all connected WebSocket clients
        for connection in self.active_connections:
            try:
                await connection.send_json(progress)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)
    
    # ============== Active Inference Methods ==============
    
    def generate_prediction(self, belief: np.ndarray) -> np.ndarray:
        """Generate prediction from current belief state"""
        # Simple linear prediction with noise
        noise = np.random.normal(0, 0.01, belief.shape)
        return belief + noise
    
    def calculate_prediction_error(self, belief: np.ndarray, prediction: np.ndarray) -> float:
        """Calculate prediction error (free energy)"""
        return float(np.linalg.norm(prediction - belief))
    
    def update_belief(self, belief: np.ndarray, prediction: np.ndarray, error: float) -> np.ndarray:
        """Update belief using prediction error"""
        learning_rate = 0.1 * np.exp(-error)  # Adaptive learning rate
        return belief + learning_rate * (prediction - belief)
    
    def bayesian_update(self, prior: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """Bayesian belief update"""
        # Simplified Bayesian update
        alpha = 0.3  # Prior weight
        return alpha * prior + (1 - alpha) * observation
    
    def calculate_entropy(self, belief: np.ndarray) -> float:
        """Calculate entropy of belief distribution"""
        # Normalize to probability distribution
        p = np.abs(belief) / np.sum(np.abs(belief))
        p = np.clip(p, 1e-10, 1)  # Avoid log(0)
        return float(-np.sum(p * np.log(p)))
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the RxInfer server"""
        logger.info(f"Starting RxInfer SRAI H3 Server on {host}:{port}")
        logger.info(f"GPU: {'Available' if self.gpu_available else 'Not Available'}")
        logger.info(f"API Documentation: http://{host}:{port}/docs")
        
        uvicorn.run(self.app, host=host, port=port)


# ============== Main Entry Point ==============

def main():
    """Main entry point for RxInfer server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RxInfer SRAI H3 Processing Server")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    # Create and run server
    server = RxInferServer(config_path=args.config)
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()