from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class TrainingJobCreate(BaseModel):
    """Schema for triggering a new training job."""
    dataset_id: str
    model_config_id: str

class TrainingJobStartResponse(BaseModel):
    """Schema for response after starting a training job."""
    training_job_id: str
    status: str
    message: str = "Training job successfully queued."

class TrainingJobStatusResponse(BaseModel):
    """Schema for checking the status of a training job."""
    id: str
    dataset_id: str
    model_config_id: str
    status: str
    metrics: List[Dict[str, Any]]
    result_model_url: Optional[str] = None
    created_at: datetime

    model_config = {
        "populate_by_name": True
    }
