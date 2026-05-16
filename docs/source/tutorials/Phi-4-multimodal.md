# **Phi-4-multimodal **
## **1. Introduction**

Phi-4-multimodal represents a significant advancement in compact multimodal models, integrating high-fidelity image understanding and text generation capabilities. This document outlines the validation steps for deploying Phi-4-multimodal on Ascend hardware, covering environment preparation, multi-node deployment, and performance evaluation.

This model is first supported in the vllm-ascend:v0.18.0rc1 version.
## **2. Supported Features**
### **Main Functional Features**

- **Multimodal Input** : Supports interleaved text and image input

- **High-Resolution Understanding** : Capable of processing highdefinition images and performing detailed object recognition
## **3. Environment Preparation**
### **3.1 Model Weights**

**Phi-4-multimodal (BF16 version)** : Requires either 1 Atlas 800 A3 (64G ×16) node or 1 Atlas 800 A2 (64G × 8) node, occupying one accelerator card.

**Phi-4-multimodal** : 

[Download model weight]: https://modelscope.cn/models/LLM-Research/Phi-4-multimodal-instruct

**Recommendation** : Download model weights to a shared directory across multiple nodes, such as /root/.cache/.

### **3.2 Multi-node Communication Verification (Optional)**

For multi-node deployment environments, it's necessary to verify the multi-node communication environment.


## **4. Installation**
### **4.1 Using Docker Image**

Use the image quay.io/ascend/vllm-ascend:v0.18.0rc1 (compatible with
Atlas 800 A2/A3):

```
export IMAGE=quay.io/ascend/vllm-ascend:v0.18.0rc1

docker run --rm \
	--name vllm-ascend \
	--shm-size=1g \
	--net=host \
	--device /dev/davinci0 \
	--device /dev/davinci1 \
	--device /dev/davinci2 \
	--device /dev/davinci3 \
	--device /dev/davinci4 \
	--device /dev/davinci5 \
	--device /dev/davinci6 \
	--device /dev/davinci7 \
	--device /dev/davinci_manager \
	--device /dev/devmm_svm \
	--device /dev/hisi_hdc \
	-v /usr/local/dcmi:/usr/local/dcmi \
	-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
	-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
	-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
	-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.inf \
	-v /etc/ascend_install.info:/etc/ascend_install.info \
	-v /root/.cache:/root/.cache \
	-it $IMAGE bash
```



## **5. Deployment**
### **5.1 Single-node Deployment**

Phi-4-multimodal can be deployed on either 1 Atlas 800 A3 (64G × 16) or1 Atlas 800 A2 (64G × 8), occupying one accelerator card.

```
export VLLM_USE_MODELSCOPE=true
export MODEL_PATH="/root/.cache/Phi-4-multimodal-instruct"
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

vllm serve ${MODEL_PATH} \
	--max-num-batched-tokens 16384 \
	--served-model-name Phi-4 \
	--trust-remote-code \
	--no-enable-prefix-caching \
	--mm-processor-cache-gb 0 \
	--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
	--additional-config '{"enable_cpu_binding":true}' \
	--port 8000
```


### **Parameter Description**

- **vllm serve ${MODEL_PATH}** : Service startup command followed bymodel path or name

- **--max-num-batched-tokens 16384** : Maximum tokens processed atonce; higher values increase throughput 

  but consume more memory

- **--served-model-name Phi-4** : Service alias, the model name specified when calling the API

- **--trust-remote-code** : Allows running custom code provided with the model 

  (typically required for deploying new models like Phi)

- **--no-enable-prefix-caching** : Disables prefix caching to save memory

- **--mm-processor-cache-gb 0** : Sets multimodal processor cache to 0, typically not needed for text-only models

- **--compilation-config ...** : Enables CUDA Graph optimization to accelerate the decoding process

- **--additional-config ...** : Binds CPU cores to reduce resource scheduling overhead and improve stability
## **6. Functionality Verification**

```
curl --location "localhost:8000/v1/chat/completions" \
	--header "Content-Type: application/json" \
	--data '{
	"model": "Phi-4",
	"messages": [
		{
			"role": "user",
			"content": [
				{
					"type": "text",
					"text": "What's in this image?"
				},
				{
					"type": "image_url",
					"image_url": {
						"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfpwisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsinmadison-the-nature-boardwalk.jpg"
					}
				}
			]
		}
	],
	"max_tokens": 300
}'
```

After the server starts, you can query the model with image-containing
prompts.
## **7. Performance Evaluation**
Using AISBench Refer to the performance evaluation documentation.
Reference Link:
[AISBench]: https://github.com/AISBench/benchmark
## **8. Summary**

This document provides a complete deployment and validation guide for the Phi-4-multimodal model on the Ascend platform, covering all key steps from environment preparation to performance evaluation. By following the guidance in this document, developers can efficiently deploy and optimize multimodal model inference services.



