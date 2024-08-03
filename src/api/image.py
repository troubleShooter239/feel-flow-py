from base64 import b64decode
from binascii import Error

from fastapi import APIRouter, HTTPException

from .schemas import Analyze, Metadata, Verify
from services.feel_flow.feel_flow import analyze, get_image_metadata, verify
from services.feel_flow.loaders.image_loader import load_image
from services.feel_flow.models.response_models import AnalyzeResult, MetadataResult, VerifyResult


router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResult)
def analyze_image(request: Analyze):
    """Analyze an image to detect facial attributes like age, emotion, gender, and race.

    Args:
        request (Analyze): Analyze instance containing the base64-encoded image and analysis options.

    Returns:
        AnalyzeResult: A list of dictionaries containing the analysis results for each detected face."""
    # try:
    #     img, _ = load_image(request.b64_img)     
    # except ValueError:
    #     raise HTTPException(400, "Not correct format of image.")
    return analyze(request.b64_img, request.actions)


@router.post("/metadata", response_model=MetadataResult)
def img_metadata(request: Metadata):
    """Extract metadata from an image.

    Args:
        request (Metadata): Metadata instance containing the base64-encoded image.

    Returns:
        MetadataResult: A dictionary containing metadata extracted from the image."""    
    try:
        img = b64decode(request.b64_img)     
        return get_image_metadata(img)
    except (Error, ValueError):
        raise HTTPException(400, "Not correct format of image.")                        


@router.post("/verify", response_model=VerifyResult)
def verify_img(request: Verify):
    """Verify the similarity between two images.

    Args:
        request (Verify): Verify instance containing the base64-encoded images, model name, and distance metric.

    Returns:
        VerifyResult: A dictionary containing the verification result and related information."""
    # try:
    #     img1, img2 = b64decode(request.b64_img1), b64decode(request.b64_img2) 
    # except Error:
    #     raise HTTPException(400, "Not correct format of image(images).")
    return verify(request.b64_img1, request.b64_img2, request.r_model_name, request.distance_metric)