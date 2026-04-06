from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Any, Optional

class ModelConfigModel(BaseModel):
    """
    Model configuration document in 'model_configs' collection.
    """
    id: Optional[str] = Field(alias="_id", default=None)
    user_id: str
    model_name: str
    task_type: str
    architecture: List[Dict[str, Any]]
    hyperparameters: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }
