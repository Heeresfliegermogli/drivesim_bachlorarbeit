#!/bin/bash
current_dir=`pwd`

#User input for docker links
echo "Enter path to DRIVE SIM container from NGC"
read -p 'DRIVE SIM path: ' DRIVESIM_CONTAINER_PATH
read -p 'Do you need a new Asset pack?(y/n): ' NEW_ASSET_PACK
case $NEW_ASSET_PACK in
	[yY] ) read -p 'Enter path to asset pack:' PATH_TO_ASSET_PACK;
	PATH_TO_ASSET_PACK=`echo "$PATH_TO_ASSET_PACK" | cut -d'"' -f 2`;
	read -p 'Enter Nucleus server IP Address:' IP;
esac

ORG=`echo "$DRIVESIM_CONTAINER_PATH" | cut -d'/' -f 2`;
TEAM=`echo "$DRIVESIM_CONTAINER_PATH" | cut -d'/' -f 3`;

if [ -x "$(command -v docker)" ]; then
   echo "Docker already installed"

else 

echo "You do not have docker installed, please go to http://get.docker.com and install docker"


fi

#Validate Setup
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

echo "! VERIFY IF THE NVIDIA DRIVER VERSION IS CORRECT !"
#Login to NGC
sudo docker login nvcr.io

#Pull DRIVESIM docker
sudo docker pull $DRIVESIM_CONTAINER_PATH

#Create local cache directory
if [ -e ~/ovcache ]; then
	echo ""
else
	mkdir ~/ovcache
	chmod -R a+rw ~/ovcache

fi
#Post install steps for docker
case $NEW_ASSET_PACK in

	[yY] ) echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile && source ~/.bash_profile
	ngc config set --ace no-ace --format_type ascii --org $ORG --team $TEAM
	if [ $? -eq 0 ]; then
		echo $?
		ngc registry resource download-version "$PATH_TO_ASSET_PACK";
		find $PWD -name '*asset_*' | while read line; do
  			cd $line
   			echo $PWD
		unzip asset.zip
	
		current_dir=`pwd`
		echo $current_dir


		NUCLEUS_TOOLS_DOCKER_IMAGE="nvcr.io/omniverse/prerel/nucleus-tools:1.1.1"
		docker pull $NUCLEUS_TOOLS_DOCKER_IMAGE

		echo "uploading the content to Nucleus server"
		docker run -e "ACCEPT_EULA=Y" -v $current_dir/asset:/backup $NUCLEUS_TOOLS_DOCKER_IMAGE upload /backup/ $IP /Projects/ -p 123456
		done
	else

echo "Please download the asset pack from https://registry.ngc.nvidia.com/orgs/drive/teams/drivesim-ov/resources/asset_package"
	echo "Press any key after the download is complete"
	while [ true ] ; do
	read -t 3 -n 1
	if [ $? = 0 ] ; then
	if [ -e files.zip ]; then
		unzip files.zip
	fi

	if [ -e asset.zip ]; then
		unzip asset.zip
	fi
	
	current_dir=`pwd`
	echo $current_dir


	NUCLEUS_TOOLS_DOCKER_IMAGE="nvcr.io/omniverse/prerel/nucleus-tools:1.1.1"
	docker pull $NUCLEUS_TOOLS_DOCKER_IMAGE


	echo "uploading the content to Nucleus server"
	docker run -e "ACCEPT_EULA=Y" -v $current_dir/asset:/backup $NUCLEUS_TOOLS_DOCKER_IMAGE upload /backup/ $IP /Projects/ -p 123456

	else
	echo "waiting for the keypress"
	fi
	done
	fi
esac
echo "!ALL SET!"
