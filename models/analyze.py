from typing import Dict

from pydantic import BaseModel


class AnalyzeModel(BaseModel):
    """Pydantic model for image analysis request."""
    b64_img: str
    actions: Dict[str, bool] = {
        "age": True, "emotion": True, "gender": True, "race": True
    }
