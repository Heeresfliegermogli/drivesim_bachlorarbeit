# Laden des trainierten Modells
model = YOLO('runs/detect/<Name>/weights/best.pt')

# Durchführung einer Vorhersage auf einem Testdatensatz
results = model.predict(source='<Pfad_zum_Testbilderordner>', save=True)

