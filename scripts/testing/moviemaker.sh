#!/bin/bash

ffmpeg -framerate 30 -pattern_type glob -i "*.png" out.avi