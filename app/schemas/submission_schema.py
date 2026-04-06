from pydantic import BaseModel, Field
from datetime import datetime

class SubmissionResponse(BaseModel):
    id: str = Field(alias="_id")  # Stringified Object ID
    user_id: str
    dataset_url: str
    target_column: str
    use_case: str
    requirement: str
    created_at: datetime

    model_config = {
        "populate_by_name": True
    }
