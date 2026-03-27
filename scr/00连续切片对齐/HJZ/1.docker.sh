#!/bin/bash

## 进入docker容器内部
docker run -it --memory=200g \
    -v "$HOME:$HOME" \
    -v /public3/Xinyu:/public3/Xinyu \
    cdgatenbee/valis-wsi bash

