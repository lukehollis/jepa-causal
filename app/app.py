from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import torch
from app.pipeline.orchestrator import CausalOrchestrator
from src.data.loaders import get_data_loaders
import os

app = FastAPI(title="Causal Engine API", version="0.1.0")

# Initialize Orchestrator
orchestrator = CausalOrchestrator()

class AnalysisRequest(BaseModel):
    # Data Source
    data_path: Optional[str] = None
    data_config: Optional[Dict[str, Any]] = None # For synthetic generation e.g. {'dataset_name': 'spatiotemporal'}
    
    # Causal Parameters
    treatment_col: Optional[str] = None # Not used yet for synthetic, but good for CSV
    outcome_col: Optional[str] = None
    
    # Hyperparameters
    config: Optional[Dict[str, Any]] = None

class AnalysisResult(BaseModel):
    status: str
    ate: Optional[float] = None
    ci_95: Optional[List[float]] = None
    stderr: Optional[float] = None
    error: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(orchestrator.device)}

@app.post("/analyze", response_model=AnalysisResult)
def analyze(request: AnalysisRequest):
    """
    Triggers a causal analysis job.
    Currently synchronous for simplicity, but designed for async expansion.
    """
    try:
        # 1. Load Data
        loader = None
        # Default config merging
        job_config = {
            'batch_size': 32,
            'k_folds': 2, # Low for speed in demo
            'jepa_epochs': 5,
            'dml_epochs': 5,
            'rep_dim': 128,
            'data_dims': (3, 16, 64, 64)
        }
        if request.config:
            job_config.update(request.config)

        if request.data_config and request.data_config.get('dataset_name') == 'spatiotemporal':
            print("[API] Generating synthetic data...")
            train_loader, _, _ = get_data_loaders(
                'spatiotemporal', 
                batch_size=job_config['batch_size'],
                n_samples=request.data_config.get('n_samples', 200)
            )
            loader = train_loader
        elif request.data_path and os.path.exists(request.data_path):
            # TODO: Implement loading from CSV/H5 into Tensor dataset
            # This requires a generic CSV loader which maps columns to X (video?), T, Y.
            # For complex video data, a simple CSV path isn't enough.
            # Assuming we handle specific formats later.
            raise HTTPException(status_code=501, detail="Custom file loading not yet implemented.")
        else:
            raise HTTPException(status_code=400, detail="Invalid data source provided.")

        # 2. Run Analysis
        print("[API] Starting analysis...")
        result = orchestrator.run_analysis(loader, job_config)
        
        return AnalysisResult(
            status="completed",
            ate=result['ate'],
            ci_95=list(result['ci_95']),
            stderr=result['stderr']
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import WebSocket
import json
import asyncio

@app.websocket("/ws/inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    try:
        # Wait for config
        data = await websocket.receive_text()
        config = json.loads(data)
        print(f"[WS] Received config: {config}")
        
        # Extract dims from config or use default
        data_dims = tuple(config.get('data_dims', [3, 8, 32, 32]))
        
        # Continuously generate and stream new data
        batch_count = 0
        while True:
            # Generate new batch
            train_loader, _, _ = get_data_loaders(
                'spatiotemporal', 
                batch_size=1, # Batch size 1 for visualization
                n_samples=10,  # Generate enough samples for train/val/test split
                data_dims=data_dims
            )
            
            # Stream the new batch
            for update in orchestrator.stream_analysis(train_loader, config):
                await websocket.send_json(update)
                await asyncio.sleep(0.1) # Rate limit
            
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"[WS] Streamed {batch_count} batches")
            
            # Small delay between batches to prevent overwhelming the client
            await asyncio.sleep(0.05)
        
    except Exception as e:
        print(f"[WS] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
