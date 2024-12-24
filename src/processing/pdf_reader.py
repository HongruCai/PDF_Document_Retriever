import os
import sys
from openai import OpenAI
from typing import List, Tuple, Dict
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import base64
from src.config import SYS_PROMPT


class PDFReader:
    """
    A class to handle PDF file reading and extracting title, author, and abstract using OpenAI's GPT-4 API.
    """

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(10))
    def call_openai_api(self, messages: List[dict], model: str) -> str:
        """
        Calls OpenAI's GPT-4 API with retry logic.

        Args:
            messages (List[dict]): The messages to send to the OpenAI API.
            model (str): The model name to use.

        Returns:
            str: The content of the response message.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

    def pdf_to_image(self, file_path: str) -> Image:
        """
        Converts the first page of a PDF to an image.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            Image: The first page of the PDF as a PIL Image object.
        """
        images = convert_from_path(file_path, first_page=1, last_page=1)
        if images:
            return images[0]
        else:
            raise RuntimeError(f"Unable to convert PDF {file_path} to image.")

    def extract_metadata(self, pdf_image: Image) -> Dict[str, str]:
        """
        Extracts the title, author, and abstract from the first page of a PDF represented as an image.

        Args:
            pdf_image (Image): The first page of the PDF as a PIL Image object.

        Returns:
            Dict[str, str]: A dictionary containing the title, authors, and abstract.
        """
        buffered = BytesIO()
        pdf_image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": [
                {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                }
                ]},
        ]
        response = self.call_openai_api(messages, model="gpt-4o-mini")

        metadata = {
            "title": "",
            "authors": "",
            "abstract": ""
        }
        try:
            metadata.update(eval(response)) 
        except Exception as e:
            raise RuntimeError(f"Error parsing metadata: {e}")

        return metadata

    def read_pdf(self, file_path: str) -> Dict[str, str]:
        """
        Reads a PDF file and extracts title, authors, and abstract using OpenAI's GPT-4 API.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            Dict[str, str]: A dictionary containing the title, authors, and abstract.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            pdf_image = self.pdf_to_image(file_path)
            metadata = self.extract_metadata(pdf_image)
            return metadata
        except Exception as e:
            raise RuntimeError(f"Error reading PDF file {file_path}: {e}")


