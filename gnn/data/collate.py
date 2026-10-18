# gnn/data/collate.py
from typing import List, Tuple, Any

def instance_pool_collate(batch: List[Tuple[Any, ...]]):
    """
    We keep 'one instance per batch' semantics.
    batch size should be 1 (one instance with its pool). If you want micro-batching multiple instances,
    apply the listwise softmax per item after forward, not across the whole collated batch.
    """
    assert len(batch) == 1, "Use batch_size=1 for instance-wise softmax training."
    return batch[0]
