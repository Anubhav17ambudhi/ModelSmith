from pydantic import BaseModel
from datetime import datetime

class DatasetResponse(BaseModel):
    """Schema for returning dataset metadata."""
    id: str  # Assuming stringified ObjectId or UUID
    user_id: str
    dataset_name: str
    storage_url: str
    file_type: str
    target_column: str
    created_at: datetime

    model_config = {
        "populate_by_name": True
    }
