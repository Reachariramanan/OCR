docker build -t dolphin-surya-app .
docker run --gpus all -p 7860:7860 dolphin-surya-app
