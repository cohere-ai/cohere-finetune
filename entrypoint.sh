#!/bin/bash

if [ -z "$TASK" ]; then
  echo "TASK is not set"
  exit 1
fi

case $TASK in
  "FINETUNE")
    python /opt/program/cohere_finetune/cohere_finetune_service.py
    ;;
  "INFERENCE")
    python /opt/program/cohere_finetune/cohere_inference_service.py
    ;;
  *)
    echo "Invalid TASK $TASK. Must be one of [FINETUNE, INFERENCE]"
    exit 1
    ;;
esac
