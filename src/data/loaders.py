import torch
from src.data.dataset import SpatiotemporalCausalDataset
from src.data.colored_mnist import get_colored_mnist_loaders
from src.data.ihdp import get_ihdp_loaders
from src.data.twins import get_twins_loaders

def get_data_loaders(dataset_name, batch_size=64, **kwargs):
    """
    Factory function to get data loaders for a specific dataset.
    
    Args:
        dataset_name (str): Name of the dataset. 
                            Options: 'spatiotemporal', 'colored_mnist', 'ihdp', 'twins'.
        batch_size (int): Batch size.
        **kwargs: Additional arguments passed to specific loader functions.
        
    Returns:
        If 'spatiotemporal': (train_loader, val_loader, test_loader)
        If 'colored_mnist': (train_loader, test_loader)
        If 'ihdp': (train_loader, val_loader, test_loader)
        If 'twins': (train_loader, val_loader, test_loader)
    """
    
    if dataset_name == 'spatiotemporal':
        # This dataset is generated on-the-fly
        # Default settings for simulation
        print(f"[get_data_loaders] kwargs: {kwargs}")
        n_samples = kwargs.get('n_samples', 1000)
        data_dims = kwargs.get('data_dims', (3, 16, 64, 64))
        print(f"[get_data_loaders] Using data_dims: {data_dims}")
        dataset = SpatiotemporalCausalDataset(n_samples=n_samples, data_dims=data_dims)
        
        # Split
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
        
    elif dataset_name == 'colored_mnist':
        # Keyword args could include confounding_strength, noise_level
        return get_colored_mnist_loaders(batch_size=batch_size, **kwargs)
        
    elif dataset_name == 'ihdp':
        # Keyword args could include replication
        return get_ihdp_loaders(batch_size=batch_size, **kwargs)
        
    elif dataset_name == 'twins':
        # Keyword args could include synthetic
        return get_twins_loaders(batch_size=batch_size, **kwargs)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
