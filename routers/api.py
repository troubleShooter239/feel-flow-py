from base64 import b64decode
from typing import Dict

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
    return analyze(request.b64_img, request.actions)

@router.post("/metadata")
def img_metadata(request: MetadataModel):
    try:                                    
        return get_image_metadata(b64decode(request.b64_img))
    except Exception:
        return {}


@router.post("/verify")
def verify_img(request: VerifyModel):
    return verify(request.b64_img1, request.b64_img2, 
                  request.model_name, request.distance_metric)
