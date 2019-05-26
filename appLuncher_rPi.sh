#!/bin/bash

echo "Switching to preconfigured python environment"
source ~/.profile
workon cv

echo "Starting up FecaStabilizer"
cd /home/pi/projects/faceCut
python show_stabalized_face.py --shape-predictor shape_predictor_68_face_landmarks.dat

echo "FecaStabilizer started"