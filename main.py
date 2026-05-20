import os

from model import (
    load_model_from_hf, predict_and_process, build_rectangles_on_image,
    INPUT_DIR
)


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