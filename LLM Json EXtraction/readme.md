# Build Docker image

docker build -t schema-md-json .

# Run Docker container

docker run -p 7860:7860 --env HF_API_TOKEN=<your_token_here> schema-md-json

HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_API_URL=https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct
HF_API_TOKEN=your_hf_token_ (Add"hf" before underscore in key)  = "_lssJtJJVIWNIhNxrmiMDsiewvsSdNUphSQ"
