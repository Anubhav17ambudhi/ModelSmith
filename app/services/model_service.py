from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.schemas.model_config_schema import ModelConfigCreate, ModelConfigResponse
from app.models.model_config_model import ModelConfigModel
from typing import List

class ModelService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db["model_configs"]

    async def create_model_config(self, user_id: str, config_data: ModelConfigCreate) -> ModelConfigResponse:
        new_config = ModelConfigModel(
            user_id=user_id,
            model_name=config_data.model_name,
            task_type=config_data.task_type,
            architecture=config_data.architecture,
            hyperparameters=config_data.hyperparameters
        )

        config_dict = new_config.model_dump(by_alias=True, exclude_none=True)

        result = await self.collection.insert_one(config_dict)

        config_dict["_id"] = str(result.inserted_id)

        # ✅ FIX
        config_dict["id"] = config_dict.pop("_id")

        return ModelConfigResponse(**config_dict)

    async def get_user_models(self, user_id: str) -> List[ModelConfigResponse]:
        cursor = self.collection.find({"user_id": user_id})
        models = []
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            models.append(ModelConfigResponse(**doc))
        return models
