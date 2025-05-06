#!/bin/bash

set -e
# User input for docker links
echo "Enter path to DRIVE SIM container from NGC"
read -p 'DRIVE SIM path: ' IMAGE
read -p 'Are you using an NGC hosted Nucleus?(y/n): ' NGC_NUCLEUS       #bei lokalen omniverse Server gebe n ein

case $NGC_NUCLEUS in
    [yY] ) docker run -it --rm --gpus all -v /usr/share/nvidia:/usr/share/nvidia/ --name=ds2 --network=host --ulimit nofile=65535:65535 --env "ACCEPT_EULA=Y" --env DISPLAY -e NGC_API_KEY=$NGC_API_KEY --env nuc=omniverse://c2981f53-74dc-4c0e-a8af-99fe16d81521.cne.ngc.nvidia.com/ -v ~/ovcache:/var/ovcache $IMAGE /bin/bash
esac        #Started den Container bei y Eingabe
    
case $NGC_NUCLEUS in
	[nN] ) read -p 'Enter Nucleus server Username:' USERNAME;      #Benutzername des Omniverse Server Nutzers
    read -p 'Enter Nucleus server Password:' PASSWORD;      #Passwort des Omniverse Server Nutzers
    OUT_DIR="/home/vima/nvidia_local_data/Output"       #Weg zum Output Ordner auf dem lokalen System

    docker run -it --rm --gpus all  \       #started den Container bei Eingabe n. --gpus all gibt dem docker das Recht auf die grafikkarte voll zuzugreifen
    --network=host --ulimit nofile=65535:65535 \
    -v /usr/share/nvidia:/usr/share/nvidia/ \
    -v ~/ovcache:/var/ovcache \
    --env "ACCEPT_EULA=Y" --env DISPLAY  \
    --env NVIDIA_DOC_PORT=8080 \
    -v "${OUT_DIR}":/tmp/replicator_out \       #Mounted unser vorher definierten lokalen Output Ordner in den Output Ordner des Containers. Erstellte Daten werden in diesen und damit automatisch in den Output Ordner im lokalen System gespeichert.
    -e OMNI_USER=${USERNAME} -e OMNI_PASS=${PASSWORD} \
    ${IMAGE} \
    bash
esac
\end{lstlisting}


