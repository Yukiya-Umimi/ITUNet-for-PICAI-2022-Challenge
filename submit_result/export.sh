#!/usr/bin/env bash

./build.sh

docker save picai_baseline_unet_processor | gzip -c > itunet.tar.gz
