from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class SubmissionModel(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    user_id: str
    dataset_url: str
    target_column: str
    use_case: str
    requirement: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }
