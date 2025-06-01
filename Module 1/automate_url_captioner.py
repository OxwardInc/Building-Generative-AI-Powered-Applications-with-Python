import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration
from urllib.parse import urljoin

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"
headers = {"User-Agent": "Mozilla/5.0"}

# Download the page
response = requests.get(url, headers=headers, timeout=10)
# Parse the page with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all img elements
img_elements = soup.find_all('img')

# Open a file to write the captions
seen = set()
with open("captions.txt", "w", encoding="utf-8") as caption_file:
    # Iterate over each img element
    for img in img_elements:
        img_url = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy")
            or img.get("srcset")
        )
        if not img_url:
            continue

        # Skip if the image is an SVG or too small (likely an icon)
        if 'svg' in img_url or '1x1' in img_url:
            continue

         # Correct the URL if it's malformed
        img_url = urljoin(url, img_url)

        if img_url in seen:
            continue
        seen.add(img_url)

        try:
            # Download the image
            print(f"Processing: {img_url}")
            img_resp = requests.get(img_url, headers=headers, timeout=10)
            # Convert the image data to a PIL Image
            raw_image = Image.open(BytesIO(img_resp.content)).convert("RGB")

            if raw_image.size[0] * raw_image.size[1] < 400:
                continue

            # Process the image
            inputs = processor(raw_image, return_tensors="pt")
            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)
            # Decode the generated tokens to text
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Write the caption to the file, prepended by the image URL
            caption_file.write(f"{img_url}: {caption}\n")
            raw_image.close()
        except Exception as e:
            print(f"Error processing {img_url}: {e}")
