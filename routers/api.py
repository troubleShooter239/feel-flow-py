from base64 import b64decode

from fastapi import APIRouter

from models.analyze import AnalyzeModel
from models.metadata import MetadataModel
from models.verify import VerifyModel
from services.face_analyze.feel_flow import analyze, get_image_metadata, verify


router = APIRouter()

#
#   TODO: Add try/except blocks for request
#
@router.post("/analyze")
def analyze_image(request: AnalyzeModel):
    data = request.model_dump()


@router.post("/metadata")
def img_metadata(request: MetadataModel):
    data = request.model_dump()
    try:                                    
        return get_image_metadata(b64decode(data["b64_img"]))
    except Exception:
        return {}


@router.post("/verify")
def verify_img(request: VerifyModel):
    data = request.model_dump()


