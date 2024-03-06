from typing import Dict

from pydantic import BaseModel


class AnalyzeModel(BaseModel):
    b64_img: str
    actions: Dict[str, bool]
