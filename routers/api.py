from base64 import b64decode

from fastapi import APIRouter

from models.analyze import AnalyzeModel
from models.metadata import MetadataModel
from models.verify import VerifyModel
from services.face_analyze.feel_flow import analyze, get_image_metadata, verify

router = APIRouter()


@router.post("/analyze")
async def analyze_image(request: AnalyzeModel):
    try:
        return await analyze(request.b64_img, request.actions)
    except Exception as e:
        return {"error": e}


@router.post("/metadata")
def img_metadata(request: MetadataModel):
    try:                                    
        return get_image_metadata(b64decode(request.b64_img))
    except Exception as e:
        return {"error": e}


@router.post("/verify")
async def verify_img(request: VerifyModel):
    try:
        return await verify(request.b64_img1, request.b64_img2, 
                            request.r_model_name, request.distance_metric)
    except Exception as e:
        return {"error": e}