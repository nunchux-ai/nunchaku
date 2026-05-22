import argparse
import sys
import os
import base64
import time
import io
import requests
from PIL import Image

# Oh, hello there! Nikola here, ready to help this little client talk to the big server!
# It's like sending a messenger bird from our village to the capital!

def main():
    # Peeking at the arguments... gotta make sure we have all our supplies for the journey!
    parser = argparse.ArgumentParser(description="ImageGen Client - A little seeker tool!")
    parser.add_argument("--host", type=str, default="localhost", help="Where the server lives (Host)")
    parser.add_argument("--port", type=int, default=8000, help="The door to knock on (Port)")
    parser.add_argument("--num_images", type=int, default=1, help="How many pictures to paint?")
    parser.add_argument("--image_folder", type=str, default="generated_images", help="Where to keep our treasures")
    # Changing defaults to None so we can use input image size if needed!
    parser.add_argument("--width", type=int, default=None, help="Canvas width (default: 1024 or input image size)")
    parser.add_argument("--height", type=int, default=None, help="Canvas height (default: 1024 or input image size)")
    
    # New shiny tools for our quest!
    parser.add_argument("--input", type=str, default=None, help="Path to an input image (for image-to-image magic!)")
    parser.add_argument("--max-size", type=int, default=1024, help="Max size for the input image (we don't want it to get too heavy for the bird!)")
    
    args = parser.parse_args()

    # Reading the prompt from the spirits... I mean, stdin!
    # "What do you desire to see?" *sparkle*
    print("Waiting for a prompt from stdin... (Type something and press Ctrl+D!)")
    try:
        prompt = sys.stdin.read().strip()
    except Exception as e:
        print(f"Oh no! The spirits were silent (stdin error): {e}")
        return

    if not prompt:
        print("Aww, the prompt was empty! The canvas remains blank.")
        return

    print(f"Yay! We got a prompt: '{prompt}'")
    
    # Restoring the canvas size variables from the journey's start!
    final_width = args.width
    final_height = args.height

    # Prepare prompt and payload
    url_gen = f"http://{args.host}:{args.port}/v1/images/generations"
    url_edit = f"http://{args.host}:{args.port}/v1/images/edits"

    try:
        if args.input:
            print(f"Oh! You brought a reference image: {args.input}. Let's go to the Editing Shrine!")
            
            # Prepare for multipart upload
            # We need to open the image file effectively
            if not os.path.exists(args.input):
                print(f"Eek! I can't find the image at {args.input}")
                return

            # Open image to ensure it's valid and memory-friendly resize if needed
            with Image.open(args.input) as img:
                img = img.convert("RGB")
                w, h = img.size
                max_dim = max(w, h)
                if max_dim > args.max_size:
                    scale = args.max_size / max_dim
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    print(f"Resizing big image from {w}x{h} to {new_w}x{new_h}. Compact and cute!")
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                
                if final_width is None: final_width = img.width
                if final_height is None: final_height = img.height

                # Save to buffer for upload
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                buffered.seek(0)
                image_bytes = buffered.getvalue()

            # Construct multipart payload
            files = {
                'image': ('input.png', image_bytes, 'image/png')
            }
            data = {
                'prompt': prompt,
                'n': args.num_images,
                'size': f"{final_width}x{final_height}",
                'response_format': 'b64_json',
                'guidance_scale': 2.5 # Default specific to edit/kontext if needed
            }

            print(f"Sending input image to {url_edit}... *whoosh*")
            response = requests.post(url_edit, files=files, data=data)

        else:
            # Standard Generation
            print("Just a prompt? Off to the Creation Forge!")
            if final_width is None: final_width = 1024
            if final_height is None: final_height = 1024

            payload = {
                "prompt": prompt,
                "n": args.num_images,
                "size": f"{final_width}x{final_height}",
                "response_format": "b64_json"
            }
            
            print(f"Sending prompt to {url_gen}... *sparkle*")
            response = requests.post(url_gen, json=payload)

        response.raise_for_status()
        
        data = response.json()
        
        # Making sure we have a chest for our treasures
        if not os.path.exists(args.image_folder):
            print(f"Creating a new treasure chest at {args.image_folder}...")
            os.makedirs(args.image_folder)

        # Unpacking the magic
        images = data.get("data", [])
        print(f"Ooh! The server sent back {len(images)} masterpieces!")

        for i, img_data in enumerate(images):
            # Decoding the spell
            img_bytes = base64.b64decode(img_data["b64_json"])
            
            timestamp = int(time.time())
            filename = f"image_{timestamp}_{i}.png"
            filepath = os.path.join(args.image_folder, filename)
            
            with open(filepath, "wb") as f:
                f.write(img_bytes)
            
            print(f"Saved masterpiece #{i+1} to {filepath}! It sparkles!")

    except requests.exceptions.ConnectionError:
        print("Oh no! The server didn't answer. Is it sleeping? (Connection Refused)")
        print("Maybe check if the host and port are correct? We tried: " + url)
    except Exception as e:
        print(f"Eek! Something went wrong on the journey: {e}")
        # We'll give it a gentle hug and try to understand...
        print("Don't worry, we can try again later!")

if __name__ == "__main__":
    main()
