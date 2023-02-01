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
  ["tiny"]="https://drive.google.com/file/d/1D-eaZX95qRlBB3QUG1zznm8P6wSvtF0D/view?usp=share_link"
  ["small"]="https://drive.google.com/file/d/1mtAJb7Doi7zN91yETxdf2Yl5nzv4iLC5/view?usp=share_link"
  ["base"]="https://drive.google.com/file/d/1yftHSA-LSa_aPc-6opAmNFTzdq7EedT_/view?usp=share_link"
)

chmod 777 ${DOWNLOAD_APP}

if [ ! -d "$WEIGHTS_DIR" ]; then
  mkdir -p "$WEIGHTS_DIR/configs"
  echo "Directory $WEIGHTS_DIR created."
else
  echo "Directory $WEIGHTS_DIR found."
fi

for WEIGHT_NAME in "${!WEIGHTS[@]}"
do
  WEIGHTS_URL=${WEIGHTS[$WEIGHT_NAME]}
  WEIGHTS_CONFIG_URL=${WEIGHTS_CONFIGS[$WEIGHT_NAME]}
  WEIGHTS_FILE="${WEIGHT_NAME}.hdf5"

  if [ ! -f "$WEIGHTS_DIR/$WEIGHTS_FILE" ]; then
    echo "File $WEIGHTS_DIR/$WEIGHTS_FILE not found, downloading it..."
    ${DOWNLOAD_APP} -u "$WEIGHTS_URL" -d "$WEIGHTS_DIR"
    ${DOWNLOAD_APP} -u "$WEIGHTS_CONFIG_URL" -d "$WEIGHTS_DIR/configs" -f "${WEIGHT_NAME}.json"

    echo "File $WEIGHTS_DIR/$WEIGHTS_FILE downloaded."
  else
    echo "File $WEIGHTS_DIR/$WEIGHTS_FILE found."
  fi
done

