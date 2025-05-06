
# NVIDIA DriveSim – Bachelorprojekt

Dieses Repository dokumentiert ein studentisches Projekt zur Evaluation der Simulationsumgebung **NVIDIA DriveSim** hinsichtlich ihrer Eignung zur Generierung synthetischer Bilddaten für KI-basierte autonome Fahrsysteme.

Die zugehörige Abschlussarbeit ist unter `report/Bachelorarbeit_Vincent_Mann_finished.pdf` zu finden.

---

## 🔧 Projektstruktur

```
drivesim-bachelorarbeit/
├── dataset/            # Beispiel-Datensätze
├── launch/             # Startskripte zur Verwendung von Replicator
├── postprocess/        # Voxel51-Konvertierung und Datensatzaufbereitung
├── replicator/         # Konfigurationsdateien & Asset-Randomisierung
├── training/           # YOLO-Trainingsskripte (Ultralytics)
├── report/             # Abgabeversion der Bachelorarbeit (PDF)
```

---

## 🚀 Vorgehensweise

Die Evaluation erfolgte in mehreren Schritten:

1. **Szenarien erstellen:**  
   In der E2E-Oberfläche von DriveSim wurden eigene Verkehrsszenarien manuell modelliert.

2. **Sensor-Setup & Datenerfassung:**  
   Über das integrierte SensorRig wurden virtuelle Kameras auf Ego-Fahrzeugen platziert.

3. **Synthetische Bilder generieren:**  
   Mit Replicator wurde automatisiert eine große Zahl annotierter RGB-Bilder erzeugt.  
   Der Start erfolgt über das Skript:

   ```bash
   launch/generate_SDG_headless.sh --scenario=sign --num-frames=50 ...
   ```

   Wichtige Replicator-Komponenten:

   - `generate_SDG_headless.sh` (Wrapper)
   - `generate_SDG_headless.py`  
   - `generate_SDG_headless_args.py`

   Diese befinden sich im Container typischerweise unter:

   ```
   /drivesim-ov/
   ```

4. **Randomisierungskonfigurationen:**  
   Die Konfigurationen zur Asset- und Szenenrandomisierung liegen im NVIDIA-Container unter:

   ```
   /Files/drivesim-ov/_build/linux-x86_64/ext/omni.drivesim.replicator.domainrand/omni/drivesim/replicator/domainrand/config/
   ```

   Die verwendeten Asset-Dateien liegen unter:

   ```
   /Files/drivesim-ov/_build/linux-x86_64/ext/omni.drivesim.replicator.domainrand/omni/drivesim/replicator/domainrand/config/assets/
   ```

5. **Postprocessing & Formatierung (VOXEL51):**  
   Die generierten `.json` + `.png`-Daten werden mit dem Python-Skript in `postprocess/` in das YOLO-Format überführt.

6. **Training des neuronalen Netzes (YOLOv11):**  
   Das Training erfolgt mit `training/yolo_train.py` innerhalb eines Ultralytics-Umfelds.  
   Als Basis dient ein Datensatz aus synthetisch erzeugten Bildern. Die Auswertung erfolgt auf realen Testdaten.

---

## 📋 Voraussetzungen

- NVIDIA DriveSim (Container)
- Voxel51 
- Python 3.11+
- Ultralytics YOLOv11

---

## 📄 Lizenz

Siehe [LICENSE](LICENSE) für weitere Informationen.

