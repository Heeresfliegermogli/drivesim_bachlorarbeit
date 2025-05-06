from ultralytics import YOLO

# Initialisiere YOLO mit dem vortrainierten 'Medium'-Modell
model = YOLO('yolo11m.pt')  

# Trainiere das Modell mit dem bereitgestellten Datensatz
model.train(
    data='/home/tas/Desktop/Yolo_Enviroment/MergedDataset/dataset.yaml',  # Pfad zur dataset.yaml-Datei
    epochs=100,         # Anzahl der Trainingsdurchläufe
    imgsz=640,          # Zielbildgröße (Skalierung auf 640x640)
    batch=32,           # Batch-Größe (Bilder pro Trainingsschritt)
    device=0,           # GPU-Index (0 = erste GPU)
    workers=8,          # Anzahl paralleler Prozesse zum Datenladen
    classes=[0],        # Nur Klasse 0 trainieren (z. B. „Automobile“)
    project='runs/detect',  # Ausgabeordner
    name='<Name>',      # Name des Versuchs (Ordnerbezeichnung)
    verbose=True        # Ausgabe detaillierter Trainingslogs
)
