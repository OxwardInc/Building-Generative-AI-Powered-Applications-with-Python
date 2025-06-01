import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"
headers = {"User-Agent": "Mozilla/5.0"}

# Download and parse the page
response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.text, 'html.parser')
img_elements = soup.find_all('img')

seen = set()
with open("captions.txt", "w", encoding="utf-8") as caption_file:
    for img_element in img_elements:
        img_url = img_element.get('src') or img_element.get('data-src')
        if not img_url:
            continue

        if 'svg' in img_url or '1x1' in img_url:
            continue

        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif not img_url.startswith('http://') and not img_url.startswith('https://'):
            continue

        if img_url in seen:
            continue
        seen.add(img_url)

        try:
            print(f"Processing: {img_url}")
            response = requests.get(img_url, headers=headers, timeout=10)
            raw_image = Image.open(BytesIO(response.content)).convert("RGB")
            if raw_image.size[0] * raw_image.size[1] < 400:
                continue

            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)

            caption_file.write(f"{img_url}: {caption}\n")
            raw_image.close()
        except Exception as e:
            print(f"Error processing {img_url}: {e}")
            continue
