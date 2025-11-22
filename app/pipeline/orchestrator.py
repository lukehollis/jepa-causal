import torch
from app.pipeline.registry import ModelRegistry
from app.pipeline.trainer import SmartTrainer
from src.causal.dml_engine import execute_dml_workflow

class CausalOrchestrator:
    """
    Orchestrates the end-to-end causal analysis:
    Data -> Registry Check -> Train/Load Model -> DML Execution -> Result
    """
    def __init__(self, registry_dir='checkpoints/registry', device=None):
        self.registry = ModelRegistry(base_dir=registry_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run_analysis(self, data_loader, config):
        """
        Executes the causal analysis pipeline.
        
        Args:
            data_loader: DataLoader containing X, T, Y.
            config: Dictionary of hyperparameters.
            
        Returns:
            ate_estimate: Average Treatment Effect.
            confidence_interval: (low, high)
        """
        # 1. Identify Dataset
        # We need to access the underlying dataset/tensor to hash it.
        # Assuming data_loader.dataset has a .tensors attribute or similar, 
        # or we iterate to hash (slow).
        # For now, let's assume the user passes the raw data dict or we extract X.
        # But hashing a DataLoader is hard. 
        # Let's assume 'data_loader' is actually a Dataset or we can get the data.
        
        # Hack: Extract all X to hash (might be memory intensive for huge data)
        # Ideally, we rely on a dataset ID passed in config, but for now let's hash X.
        if hasattr(data_loader, 'dataset'):
            # Use the underlying dataset to ensure consistent order regardless of shuffle
            # We assume the dataset is map-style and small enough to iterate or has .data
            try:
                # Optimization: if dataset has .data or .tensors, use it directly
                if hasattr(data_loader.dataset, 'data') and isinstance(data_loader.dataset.data, dict):
                     # SpatiotemporalCausalDataset stores numpy array in .data['X']
                     X_tensor = torch.tensor(data_loader.dataset.data['X'])
                elif hasattr(data_loader.dataset, 'tensors'):
                     # TensorDataset
                     X_tensor = data_loader.dataset.tensors[0]
                else:
                    # Fallback: iterate dataset in order
                    X_all = [data_loader.dataset[i]['X'] for i in range(len(data_loader.dataset))]
                    X_tensor = torch.stack(X_all)
            except Exception as e:
                print(f"[Orchestrator] Warning: Could not access dataset directly ({e}). Falling back to loader iteration.")
                X_all = []
                for batch in data_loader:
                     X_all.append(batch['X'])
                X_tensor = torch.cat(X_all)
        else:
             X_all = []
             for batch in data_loader:
                  X_all.append(batch['X'])
             X_tensor = torch.cat(X_all)

        data_hash = self.registry.compute_dataset_hash(X_tensor)
        
        print(f"[Orchestrator] Dataset Hash: {data_hash}")
        
        # 2. Check Registry / Train Model
        encoder = None
        
        if self.registry.has_model(data_hash):
            print("[Orchestrator] Found cached World Model.")
            # We need to instantiate the model class first to load weights?
            # Registry.load_model returns state_dict if instance not provided.
            # We need to know which model class to init.
            # For now, assume vit_small_video based on config.
            from src.models.encoder import vit_small_video
            # We need to infer shape again or store it in metadata.
            # Let's infer from X_tensor
            _, C, T, H, W = X_tensor.shape
            # Initialize encoder
            # Using 12 heads as requested for better visualization
            encoder = vit_small_video(
                img_size=H,
                num_frames=T,
                patch_size=config.get('patch_size', 8),
                tubelet_size=config.get('tubelet_size', 2),
                num_heads=12, 
                embed_dim=384, # Keep embed dim same for now, just more heads
                in_chans=C,
                rep_dim=config.get('rep_dim', 256)
            ).to(self.device)
            
            self.registry.load_model(data_hash, encoder)
            encoder.eval()
            
        else:
            print("[Orchestrator] No cached model found. Training World Model...")
            trainer = SmartTrainer(config, self.device)
            encoder = trainer.train(data_loader)
            self.registry.save_model(encoder, data_hash)
            
        # 3. Execute DML with Fixed Encoder
        # We define a wrapper that returns our pre-trained encoder
        def get_fixed_encoder(loader, conf, dev):
            return encoder
            
        print("[Orchestrator] Running DML with World Model...")
        ate, stderr = execute_dml_workflow(
            data_loader.dataset, 
            config, 
            self.device, 
            train_encoder_func=get_fixed_encoder
        )
        
        return {
            'ate': ate,
            'stderr': stderr,
            'ci_95': (ate - 1.96 * stderr, ate + 1.96 * stderr)
        }

    def stream_analysis(self, data_loader, config):
        """
        Generator that yields inference steps for visualization.
        Yields:
            dict: {
                'type': 'input'|'encoding'|'prediction',
                'data': ...
            }
        """
        # 1. Identify/Train Model (Simplified for demo: just load or train)
        # For visualization, we might just want to run the encoder on data.
        
        # ... (Reuse logic to get encoder) ...
        # For now, let's assume we have the encoder.
        # TODO: Refactor run_analysis to separate model loading.
        
        # Hack: Just get a model for now.
        from src.models.encoder import vit_small_video
        
        print(f"[Orchestrator] Initializing encoder with: img_size={config['data_dims'][2]}, num_frames={config['data_dims'][1]}")
        
        # Dummy shape inference if needed, or just use config
        encoder = vit_small_video(
            img_size=config['data_dims'][2],
            patch_size=config.get('patch_size', 8),
            num_frames=config['data_dims'][1],
            tubelet_size=config.get('tubelet_size', 2),
            in_chans=config['data_dims'][0],
            num_heads=12,
            embed_dim=384,
            rep_dim=config.get('rep_dim', 256)
        ).to(self.device)
        
        print(f"[Orchestrator] Encoder pos_embed shape: {encoder.pos_embed.shape}")
        encoder.eval()
        
        # 2. Stream Data
        print("[Orchestrator] Streaming inference...")
        import numpy as np
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                x = batch['X'].to(self.device) # [B, C, T, H, W]
                
                # Yield Input
                # Convert to list for JSON serialization
                # Take first sample in batch for visualization
                x_vis = x[0].cpu().numpy() # [C, T, H, W]
                
                yield {
                    'type': 'input',
                    'batch_idx': batch_idx,
                    'data': x_vis.tolist() 
                }
                
                # Forward Pass
                # Encoder returns representation
                rep, attn = encoder(x, return_last_attn=True) # [B, RepDim], [B, NumHeads, N, N]
                rep_vis = rep[0].cpu().numpy()
                attn_vis = attn[0].cpu().numpy() # [NumHeads, N, N]
                
                yield {
                    'type': 'representation',
                    'batch_idx': batch_idx,
                    'data': rep_vis.tolist(),
                    'attention': attn_vis.tolist()
                }
                
                # Simulate some "prediction" or "causal effect" visualization
                # e.g. Attention map or counterfactual
                
                # Slow down slightly for demo effect if needed
                import time; time.sleep(1.5)
                
                if batch_idx >= 1000: # Run longer
                    break
