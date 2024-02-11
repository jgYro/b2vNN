#cat random.txt | xargs -I {} sh -c 'filename=$(basename "{}" ); python3 main.py {} --image-size 512 512 --output ./train/1/$filename.png'
cat random.txt | xargs -I {} -P 0 sh -c 'filename=$(echo {} | awk -F "/" "{print \$NF}" | awk -F "." "{print \$1}"); python3 main.py {} --image-size 512 512 --output ./train/1/$filename.png'
