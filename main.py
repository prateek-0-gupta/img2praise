import os
import cv2
import time
import requests
from PIL import Image, UnidentifiedImageError
from llama_cpp import Llama
import moondream as md
from kokoro import KPipeline
import soundfile as sf
import numpy as np


pipeline = KPipeline(lang_code='a')

def capture_photo(output_path="image.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return None

    print("[INFO] Press 'SPACE' to capture photo or 'ESC' to exit.")

    photo_captured = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from the camera.")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            try:
                cv2.imwrite(output_path, frame)
                photo_captured = True
                print(f"[INFO] Saved photo to '{output_path}'.")
            except cv2.error as e:
                print(f"[ERROR] Could not save photo: {e}")
            break
        elif key == 27:  # ESC
            print("[INFO] Exiting without capturing a photo.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if photo_captured:
        return output_path
    return None


def load_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image '{image_path}' does not exist.")
        return None
    try:
        image = Image.open(image_path)
        print(f"[INFO] Image loaded: '{image_path}'")
        return image
    except (UnidentifiedImageError, IOError) as e:
        print(f"[ERROR] Could not open image: {e}")
        return None

def generate_caption(image, md_model):
    try:
        encoded_image = md_model.encode_image(image)
        caption = md_model.query(encoded_image, "describe their facial attributes and their appearance and their fashion choices")["answer"]
        print(f"[INFO] Moondream result: {caption}")
        return caption
    except Exception as e:
        print(f"[ERROR] Failed to generate caption: {e}")
        return None


def generate_compliment(caption, llm, max_tokens=256):
    prompt = (
        f"As a nice human kawaii girlfriend, and base on this image discription '{caption}' give user a very short, cute compliment describing their facial attributes and their appearance and their fashion choices:"
    )

    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["Q:"]
        )
        raw_text = output["choices"][0]["text"]
        clean_text = raw_text.replace("<|assistant|>", "").strip()
        print(f"[INFO] Generated compliment: {clean_text}")
        return clean_text
    except Exception as e:
        print(f"[ERROR] Failed to generate compliment: {e}")
        return None


def generate_speech(text, output_file="output.wav"):
    if not text:
        print("[ERROR] Cannot generate speech from empty text.")
        return False

    generator = pipeline(
    text, voice='af_nicole', # 
    speed=1, split_pattern=r'\n+'
    )
    audio_chunks = []
    for i, (gs, ps, audio) in enumerate(generator):
        print(f"[INFO] Processing chunk {i}")
        print(gs)  # graphemes/text
        print(ps)  # phonemes
        audio_chunks.append(audio)

    if audio_chunks:
        full_audio = np.concatenate(audio_chunks, axis=0)
        sf.write(output_file, full_audio, 24000)
        print(f"[INFO] Speech saved to '{output_file}'")
        return True
    else:
        print("[ERROR] No audio generated.")
        return False


def main():
   
    photo_path = capture_photo("image.jpg")
    if not photo_path:
        print("[INFO] No photo captured. Exiting.")
        return
    print("[INFO] Loading Moondream model...")
    try:
        md_model = md.vl(model="./models/moondream-0_5b-int8.mf")
    except Exception as e:
        print(f"[ERROR] Failed to load Moondream model: {e}")
        return
    image = load_image(photo_path)
    if image is None:
        return  
    caption = generate_caption(image, md_model)
    if not caption:
        return 
    PHI_model_path = "./models/phi/Phi-3-mini-4k-instruct-q4.gguf"
    print("[INFO] Loading text model...")
    try:
        llm = Llama(
            model_path=PHI_model_path,
            n_threads=4,
            n_ctx=4096,
            logits_all=False,
            vocab_only=False,
            use_mmap=True,
            use_mlock=False
        )
    except Exception as e:
        print(f"[ERROR] Failed to load text model: {e}")
        return
    compliment = generate_compliment(caption, llm)
    if not compliment:
        return  
    generate_speech(compliment)

if __name__ == "__main__":
    main()
