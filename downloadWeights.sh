#!/bin/bash

WEIGHTS_DIR="public/weights"
DOWNLOAD_APP="./utils/goodls"

declare -A WEIGHTS
WEIGHTS=(
  ["tiny"]="https://drive.google.com/file/d/1ttm1Rdfgk8T-jG1EJhwIxvT0coSdXnN5/view?usp=share_link"
  ["small"]="https://drive.google.com/file/d/1zTDbW_eKPqOwOhhLy4_RifC_fmxVgjxv/view?usp=share_link"
  ["base"]="https://drive.google.com/file/d/1l3ELPd5MRTG1mIc2UmU7QTBLy-7n71Nj/view?usp=share_link"
)

declare -A WEIGHTS_CONFIGS
WEIGHTS_CONFIGS=(
  ["tiny"]="https://drive.google.com/file/d/1ttm1Rdfgk8T-jG1EJhwIxvT0coSdXnN5/view?usp=share_link"
  ["small"]="https://drive.google.com/file/d/1zTDbW_eKPqOwOhhLy4_RifC_fmxVgjxv/view?usp=share_link"
  ["base"]="https://drive.google.com/file/d/1l3ELPd5MRTG1mIc2UmU7QTBLy-7n71Nj/view?usp=share_link"
)

chmod 777 ${DOWNLOAD_APP}

if [ ! -d "$WEIGHTS_DIR" ]; then
  mkdir -p "$WEIGHTS_DIR"
  echo "Directory $WEIGHTS_DIR created."
else
  echo "Directory $WEIGHTS_DIR found."
fi

for WEIGHT_NAME in "${!WEIGHTS[@]}"
do
  WEIGHTS_URL=${WEIGHTS[$WEIGHT_NAME]}
  WEIGHTS_FILE="${WEIGHT_NAME}.hdf5"

  if [ ! -f "$WEIGHTS_DIR/$WEIGHTS_FILE" ]; then
    echo "File $WEIGHTS_DIR/$WEIGHTS_FILE not found, downloading it..."
    ${DOWNLOAD_APP} -u "$WEIGHTS_URL" -d "$WEIGHTS_DIR"

    echo "File $WEIGHTS_DIR/$WEIGHTS_FILE downloaded."
  else
    echo "File $WEIGHTS_DIR/$WEIGHTS_FILE found."
  fi
done

