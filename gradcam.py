import cv2
import numpy as np
import torch
from PIL import Image

from model import model, transform, DEVICE


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._save_activation
            )
        )


    # --------------------------------------------------------
    # Save activation
    # --------------------------------------------------------

    def _save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

        # Register gradient directly on the activation tensor.
        # This is more robust than using a backward hook.
        if output.requires_grad:

            output.register_hook(
                self._save_gradient
            )


    # --------------------------------------------------------
    # Save gradient
    # --------------------------------------------------------

    def _save_gradient(self, gradient):

        self.gradients = gradient


    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    def generate(self, input_tensor):

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None


        # Forward pass
        output = self.model(input_tensor)


        # Our model outputs one logit:
        #
        # positive = pneumonia
        #
        target_score = output[:, 0]


        # Backpropagation
        target_score.backward()


        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )


        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )


        activations = self.activations
        gradients = self.gradients


        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )


        # ----------------------------------------------------
        # Weighted feature maps
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(dim=1)


        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = torch.relu(cam)


        cam = cam[0].detach().cpu().numpy()


        # ----------------------------------------------------
        # Normalize CAM
        # ----------------------------------------------------

        cam -= cam.min()


        max_value = cam.max()


        if max_value > 0:

            cam /= max_value


        return cam, output.detach()


    # --------------------------------------------------------
    # Remove hook
    # --------------------------------------------------------

    def remove_hooks(self):

        self.forward_handle.remove()


# ============================================================
# CREATE GRAD-CAM IMAGE
# ============================================================

def create_gradcam(image: Image.Image):

    # Original image
    original_image = image.convert("RGB")


    # Image used for display
    display_image = original_image.resize(
        (224, 224)
    )


    # Same preprocessing as model prediction
    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(DEVICE)


    # DenseNet final convolutional feature layer
    target_layer = model.features.norm5


    gradcam = GradCAM(
        model,
        target_layer
    )


    try:

        cam, output = gradcam.generate(
            input_tensor
        )

    finally:

        gradcam.remove_hooks()


    # --------------------------------------------------------
    # Pneumonia probability
    # --------------------------------------------------------

    probability = torch.sigmoid(
        output
    ).item()


    # --------------------------------------------------------
    # Resize CAM
    # --------------------------------------------------------

    cam = cv2.resize(
        cam,
        (
            display_image.width,
            display_image.height
        )
    )


    # --------------------------------------------------------
    # Convert CAM to heatmap
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * cam
    )


    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original_array = np.array(
        display_image
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_array,
        0.55,
        heatmap,
        0.45,
        0
    )


    overlay_image = Image.fromarray(
        overlay
    )


    return overlay_image, probability