from base64 import b64decode

from fastapi import APIRouter

from models.analyze import AnalyzeModel
from models.metadata import MetadataModel
from models.verify import VerifyModel
from services.face_analyze.feel_flow import analyze, get_image_metadata, verify

router = APIRouter()


@router.post("/analyze")
def analyze_image(request: AnalyzeModel):
    """Analyze an image to detect facial attributes like age, emotion, gender, and race.

    Args:
        request (AnalyzeModel): AnalyzeModel instance containing the base64-encoded image
                                and analysis options.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the analysis results for each detected face.

    Raises:
        Exception: If an error occurs during the analysis process."""
    try:
        return analyze(request.b64_img, request.actions)
    except Exception as e:
        return {"error": e}


@router.post("/metadata")
def img_metadata(request: MetadataModel):
    """Extract metadata from an image.

    Args:
        request (MetadataModel): MetadataModel instance containing the base64-encoded image.

    Returns:
        Dict: A dictionary containing metadata extracted from the image.

    Raises:
        Exception: If an error occurs during the metadata extraction process."""
    try:                                    
        return get_image_metadata(b64decode(request.b64_img))
    except Exception as e:
        return {"error": f"{e}"}


@router.post("/verify")
async def verify_img(request: VerifyModel):
    """Verify the similarity between two images.

    Args:
        request (VerifyModel): VerifyModel instance containing the base64-encoded images,
                               model name, and distance metric.

    Returns:
        Dict: A dictionary containing the verification result and related information.

    Raises:
        Exception: If an error occurs during the verification process."""
    try:
        return await verify(request.b64_img1, request.b64_img2, 
                            request.r_model_name, request.distance_metric)
    except Exception as e:
        return {"error": e}