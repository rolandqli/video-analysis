"""Image analysis tools using OpenAI vision."""
from openai import OpenAI

from utils.image_utils import load_image


def create_detect_objects(client: OpenAI):
    """Factory for detect_objects tool bound to an OpenAI client."""

    def detect_objects(image_path: str) -> dict:
        """
        Detect objects in an image and return a scene description with bounding boxes.
        """
        image_url = load_image(image_path)

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Detect objects and describe scene."},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
        )

        return {"analysis": response.output_text}

    return detect_objects


def create_summarize_scene(client: OpenAI):
    """Factory for summarize_scene tool bound to an OpenAI client."""

    def summarize_scene(scene_description: str) -> str:
        """
        Convert a scene description into structured metadata and tags.
        """
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Convert this scene description into structured tags: {scene_description}",
        )

        return response.output_text

    return summarize_scene
