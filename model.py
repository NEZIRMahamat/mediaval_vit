
import os
import json
from tqdm import tqdm

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageColor


INPUT_DIR = "manuscrites" # images manuscrites à traiter
OUTPUT_DIR = "output" # résultats JSON (sortie)
OUTPUT_IMAGES_COLOREES_DIR = "output_images_colorees" # images avec les détections tracées (rectangles) (sortie)


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(OUTPUT_IMAGES_COLOREES_DIR):
    os.makedirs(OUTPUT_IMAGES_COLOREES_DIR)

COLORS_RECTAGLES_DETECTIONS = [
    "red", "blue", "green", "yellow", "cyan", "magenta", 
    "orange", "purple", "pink", "brown"
    ] # couleurs des classes rectangles détectés (en boucle si +sieurs classes)


# Load model from huggingface hub
def load_model_from_hf(repo_id, filename):
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        return YOLO(path, task='obb')
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None


# Save coordinates predicted in a json file
def save_coordinates_to_json(output_data):
    if not output_data:
        print("Aucune donnée à enregistrer.")
        return
    if (output_data.get("detections") is None):
        print("Données de détection manquantes, aucune détection trouvée.")
        return
    image_name = output_data.get("image_name", "unknown_image")
    image_name = os.path.splitext(image_name)[0]  # Remove file extension
    output_path = f"{OUTPUT_DIR}/{image_name}_output.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    return output_data

# Prediction and processing results with image name
def predict_and_process(model, image_path):
    results = model.predict(image_path)    
    image_name = os.path.basename(image_path)
    out_docs = {"image_name": image_name, "image_path": image_path, "classes" : [],
                "detections": []}
    docs = []

    for r in tqdm(results):
        for box in r.obb:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf) # confidence score (évaluation de la confiance de la détection)
            coordinates = box.xyxyxyxy.tolist()

            doc = {
                "class_name": class_name,
                "confidence": confidence,
                "coordinates": coordinates
            }
            docs.append(doc)

    if docs:
        out_docs["detections"] = docs
    else:
        out_docs["detections"] = None
    out_docs["classes"] = list(set([doc["class_name"] for doc in docs]))
    
    return save_coordinates_to_json(out_docs)

# Build rectangles on the input image and save it in output_images_colorees
def build_rectangles_on_image(output_data):
    if not output_data:
        print("Aucune donnée à traiter pour construire les rectangles.")
        return
    if (output_data.get("detections") is None):
        print("Données de détection manquantes, pas de détection")
        return
    image_path = output_data.get("image_path")
    image_name = output_data.get("image_name", "inconnu_image")
    detections = output_data.get("detections", [])
    
    if not detections:
        print(f"Aucune détection trouvée pour l'image {image_name}")
        return
    
    try:
        img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw_text = ImageDraw.Draw(img)

        for idx, detection in enumerate(detections):
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            coords = detection["coordinates"]
            color = COLORS_RECTAGLES_DETECTIONS[idx % len(COLORS_RECTAGLES_DETECTIONS)]
            pts = coords[0] if isinstance(coords[0][0], list) else coords
            polygon = [(x, y) for x, y in pts]
            color_rgb = ImageColor.getrgb(color)          # (R, G, B)
            color_transparent = color_rgb + (40,)         # (R, G, B, 40) => 15% opacité
            draw.polygon(polygon, outline=color_rgb + (255,), fill=color_transparent)
            draw_text.text((polygon[0][0], polygon[0][1] - 30),
                      f"{class_name} : {confidence:.2f}", fill=color, anchor="la")
        img = Image.alpha_composite(img, overlay).convert("RGB")

        output_image_path = f"{OUTPUT_IMAGES_COLOREES_DIR}/{image_name}"
        img.save(output_image_path)
        print(f"Image avec rectangles enregistrée : {output_image_path}")
    except Exception as e:
        print(f"Erreur lors de la construction des rectangles sur l'image {image_name} : {e}")

def main():
    # parcourir images du dossier d'entrée : traitement par image
    name_model = "magistermilitum/YOLO_manuscripts"
    model = load_model_from_hf(name_model, 'best.pt')
    for image_file in os.listdir(INPUT_DIR):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(INPUT_DIR, image_file)
            print(f"Traitement de l'image : {image_path}")            
            output_data = predict_and_process(model=model, image_path=image_path)
            build_rectangles_on_image(output_data)

if __name__ == "__main__":    
    main()




