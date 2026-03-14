from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class DatasetModel(BaseModel):
    """
    Dataset metadata document in 'datasets' collection.
    """
    id: Optional[str] = Field(alias="_id", default=None)
    user_id: str
    dataset_name: str
    storage_url: str
    file_type: str
    target_column: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }
