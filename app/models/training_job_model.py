from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Any, Optional

class TrainingJobModel(BaseModel):
    """
    Training job document in 'training_jobs' collection.
    """
    id: Optional[str] = Field(alias="_id", default=None)
    user_id: str
    dataset_id: str
    model_config_id: str
    status: str = "Queued"  # "Queued" | "Running" | "Completed" | "Failed"
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    result_model_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }
