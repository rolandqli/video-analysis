from mcp.server.fastmcp import FastMCP
import base64
import cv2
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


mcp = FastMCP("video-analysis")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}

def load_image(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lower()
    mime = MIME_TYPES.get(ext, "image/jpeg")
    return f"data:{mime};base64,{data}"

@mcp.tool()
def detect_objects(image_path: str) -> dict:
    """
    Detect objects in an image and return bounding boxes + summary.
    """
    image_url = load_image(image_path)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Detect objects and describe scene."},
                {
                    "type": "input_image",
                    "image_url": image_url
                }
            ]
        }]
    )

    return {
        "analysis": response.output_text
    }


@mcp.tool()
def summarize_scene(scene_description: str) -> str:
    """
    Convert detection output into structured metadata.
    """
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"Convert this scene description into structured tags: {scene_description}"
    )

    return response.output_text


if __name__ == "__main__":
    mcp.run()