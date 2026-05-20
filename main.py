from ultralytics import YOLO 
name_model = "magistermilitum/YOLO_manuscripts" # 
from huggingface_hub import hf_hub_download 

path = hf_hub_download(repo_id='magistermilitum/YOLO_manuscripts', 
                       filename='best.pt' ) 
print(path)

# coords peut être [[[x1,y1],...]] ou [[x1,y1],...]
# print("Modèle chargé")