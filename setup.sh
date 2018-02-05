#!/bin/bash

echo "----------------------------------------------------------"
echo "Setting up Python Virtual Env"
echo "----------------------------------------------------------"
current_dir=`pwd`
virtualenv -p python3 --no-site-packages --distribute .meng_env
source ${current_dir}/.meng_env/bin/activate
pip install -r requirements.txt

echo "----------------------------------------------------------"
echo "Checking Starcraft Install"
echo "----------------------------------------------------------"

if [ -d ~/StarCraftII ];
then
    echo "Starcraft 2 already installed!"
    echo "Skipping..."
else
    echo "Downloading Starcraft 2..."
    wget --continue  --tries=0 http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.0.2.zip -O /tmp/sc2.zip
    echo "Password is 'iagreetotheeula' and binds you to the AI and Machine Learning License."
    unzip /tmp/sc2.zip -d ~/
    rm /tmp/sc2.zip

    echo "Downloading Map packs..."
    wget --continue http://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip -O /tmp/melee.zip
    echo "Password is 'iagreetotheeula' and binds you to the AI and Machine Learning License."
    unzip /tmp/melee.zip -d ~/StarCraftII/Maps/
    rm /tmp/melee.zip

    wget --continue https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip -O /tmp/mini_games.zip
    unzip /tmp/mini_games.zip -d ~/StarCraftII/Maps/
    rm /tmp/mini_games.zip
fi

echo "----------------------------------------------------------"
echo "Done!"
echo "----------------------------------------------------------"

echo "Run '. ./start.sh' to enter the Virtual Environment."
echo "Run 'python -m pysc2.bin.map_list' to check the maps have been installed correctly"
