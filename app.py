import base64

from io import BytesIO

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

from model import predict

from gradcam import create_gradcam


# ============================================================
# ============================================================
# FASTAPI APP
# ============================================================
# ============================================================

app = FastAPI(

    title="Pneumonia Detection API",

    description=(
        "DenseNet-121 Chest X-ray "
        "Pneumonia Detection API "
        "with Grad-CAM Explainability"
    ),

    version="3.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "Pneumonia Detection API",

        "status":
            "running",

        "model":
            "DenseNet-121",

        "explainability":
            "Grad-CAM",

        "version":
            "3.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy"
    }


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(
    image: Image.Image
):

    buffer = BytesIO()


    image.save(
        buffer,
        format="JPEG",
        quality=90
    )


    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


    return encoded


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def prediction(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    allowed_types = [

        "image/jpeg",

        "image/jpg",

        "image/png"
    ]


    if file.content_type not in allowed_types:

        raise HTTPException(

            status_code=400,

            detail=(
                "Only JPG and PNG "
                "images are supported."
            )
        )


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        contents = await file.read()


        image = Image.open(
            BytesIO(contents)
        )


        image.load()


    except Exception:

        raise HTTPException(

            status_code=400,

            detail="Invalid image file."
        )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    try:

        result = predict(image)


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "Model prediction failed: "
                f"{str(e)}"
            )
        )


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    try:

        gradcam_image, _ = create_gradcam(
            image
        )


        gradcam_base64 = (
            image_to_base64(
                gradcam_image
            )
        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "Grad-CAM generation failed: "
                f"{str(e)}"
            )
        )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "filename":
            file.filename,

        "prediction":
            result["prediction"],

        "pneumonia_probability":
            round(
                result[
                    "pneumonia_probability"
                ],
                6
            ),

        "normal_probability":
            round(
                result[
                    "normal_probability"
                ],
                6
            ),

        "threshold":
            result["threshold"],

        "gradcam_image":
            (
                "data:image/jpeg;base64,"
                + gradcam_base64
            )
    }