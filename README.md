# cohere-finetune
Cohere-finetune is a tool that facilitates easy, efficient and high-quality fine-tuning of Cohere's models on users' own data to serve their own use cases.

Currently, we support the following base models for fine-tuning:
- [Cohere's Command R in HuggingFace](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
- [Cohere's Command R 08-2024 in HuggingFace](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024)
- [Cohere's Command R Plus in HuggingFace](https://huggingface.co/CohereForAI/c4ai-command-r-plus)
- [Cohere's Command R Plus 08-2024 in HuggingFace](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024)
- [Cohere's Aya Expanse 8B in HuggingFace](https://huggingface.co/CohereForAI/aya-expanse-8b)
- [Cohere's Aya Expanse 32B in HuggingFace](https://huggingface.co/CohereForAI/aya-expanse-32b)

We also support any customized base model built on one of these supported models (see [Step 4](#step-4-submit-the-request-to-start-the-fine-tuning) for more details).

Currently, we support the following fine-tuning strategies:
- [Parameter efficient fine-tuning by LoRA](https://arxiv.org/pdf/2106.09685)
- [Parameter efficient fine-tuning by QLoRA](https://arxiv.org/pdf/2305.14314)

We will keep extending the base models and fine-tuning strategies we support, and keep adding more features, to help our users fine-tune Cohere's models more easily, more efficiently and with higher quality.

## 1. Prerequisites
- You need to have access to a machine with at least one GPU. The specific required number, memory and model of GPUs depend on your specific use case, e.g., the model to fine-tune, the batch size, the max sequence length in the data, etc.
- You need to install necessary apps, e.g., Docker, Git, etc. on the GPU machine.

To help you better decide the hardware resources you need, we list some feasible scenarios in the following table as a reference, where all the other hyperparameters that are not shown in the table are set as their default values (see [here](#step-4-submit-the-request-to-start-the-fine-tuning)).

| Hardware resources | Base model                                                    | Finetune strategy | Batch size | Max sequence length |
|:-------------------|:--------------------------------------------------------------|:------------------|:-----------|:--------------------|
| 8 * 80GB H100 GPUs | Command R, Command R 08-2024, Aya Expanse 8B, Aya Expanse 32B | LoRA or QLoRA     | 8          | 16384               |
| 8 * 80GB H100 GPUs | Command R, Command R 08-2024, Aya Expanse 8B, Aya Expanse 32B | LoRA or QLoRA     | 16         | 8192                |
| 8 * 80GB H100 GPUs | Command R Plus, Command R Plus 08-2024                        | LoRA or QLoRA     | 8          | 8192                |
| 8 * 80GB H100 GPUs | Command R Plus, Command R Plus 08-2024                        | LoRA or QLoRA     | 16         | 4096                |

## 2. Setup
Run the commands below on the GPU machine.
```
git clone git@github.com:cohere-ai/cohere-finetune.git
cd cohere-finetune
```

## 3. Fine-tuning
Throughout this section and the sections below, we use the notation `<some_content_you_must_change>` to denote some content that you must change according to your own use case, e.g., names, paths to files or directories, etc. Meanwhile, for any name or path that is not between the angle brackets, you must use it as it is, unless otherwise stated.

You can fine-tune a base model on your own data by following the steps below on the GPU machine (the host).

### Step 1. Build the Docker image
Run the command below to build the Docker image, which may take about 18min to finish if it is the first time you build it on the host.
```commandline
DOCKER_BUILDKIT=1 docker build --rm \
    --ssh default \
    --target peft-prod \
    -t <peft_prod_docker_image_name> \
    -f docker/Dockerfile \
    .
```

Alternatively, you may directly use the image we built for you: skip this step and use our image name `ghcr.io/cohere-ai/cohere-finetune:latest` as `<peft_prod_docker_image_name>` in the next step.

### Step 2. Run the Docker container to start the fine-tuning service
Run the command below to start the fine-tuning service.
```commandline
docker run -it --rm \
    --name <peft_prod_finetune_service_docker_container_name> \
    --gpus <gpus_accessible_by_the_container> \
    --ipc=host \
    --net=host \
    -v ~/.cache:/root/.cache \
    -v <finetune_root_dir>:/opt/finetuning \
    -e PATH_PREFIX=/opt/finetuning/<finetune_sub_dir> \
    -e ENVIRONMENT=DEV \
    -e TASK=FINETUNE \
    -e HF_TOKEN=<hf_token> \
    -e WANDB_API_KEY=<wandb_api_key> \
    <peft_prod_docker_image_name>
```

Some parameters are explained below:
- `<gpus_accessible_by_the_container>` specifies the GPUs the service can access, which can be, e.g., `'"device=0,1,2,3"'` (for GPUs 0, 1, 2, 3) or `all` (for all GPUs).
- By default, HuggingFace will cache all downloaded models in `~/.cache/huggingface/hub` and try to fetch the cached model from there when you want to load a model again. Therefore, it is highly recommended to mount `~/.cache` on your host to `/root/.cache` in the container, such that the container will have access to these cached models on your host and avoid going through the time-consuming model downloading process.
- `<finetune_root_dir>` is the root directory on your host to store all your fine-tunings, and `/opt/finetuning` is the corresponding fine-tuning root directory in your container (it can also be changed but you do not have to).
- `PATH_PREFIX` is an environment variable that specifies the fine-tuning sub-directory in your container, where `<finetune_sub_dir>` can be an empty string, i.e., the fine-tuning sub-directory can be equal to the fine-tuning root directory.
- `ENVIRONMENT` is an environment variable that specifies the mode of your working environment, which is mainly used to determine the level of logging. If you explicitly set it as `DEV`, more debugging information will be printed, but if you do not set it or set it as any other value, these debugging information will not be printed.
- `HF_TOKEN` is an environment variable that specifies your [HuggingFace User Access Token](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables#hftoken).
- `WANDB_API_KEY` is an environment variable that specifies the [authentication key of your Weights & Biases account](https://docs.wandb.ai/guides/track/environment-variables/). If you are not going to use Weights & Biases for logging during the fine-tuning, you do not have to set it.

If you want the service to run in the background, now you can [detach from the Docker container](https://docs.docker.com/reference/cli/docker/#default-key-sequence-to-detach-from-containers).

### Step 3. Prepare the training and evaluation data
Put one and only one file as the training data in the directory `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>/input/data/training`, where this file must be one of the followings:
- A CSV file with file extension `.csv` in the prompt-completion format. The file must **not** have a header and must have two columns, where the first column consists of prompts and the second column consists of completions.
- A JSONL file with file extension `.jsonl` in the prompt-completion format. Each record (line) in the file must have two keys: `"prompt"` and `"completion"` for prompt and completion, respectively.
- A JSONL file with file extension `.jsonl` in the chat format. See [here](https://docs.cohere.com/docs/chat-preparing-the-data#data-format) for the specific requirements. See an example at [here](https://github.com/cohere-ai/notebooks/blob/main/notebooks/data/scienceQA_train.jsonl).

Optionally, you can also put one and only one file as the evaluation data in the directory `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>/input/data/evaluation`, where this file must be in one of the three formats above. If you do not provide any evaluation data, we will split the provided training data into training and evaluation sets according to the hyperparameter `eval_percentage` (see [Step 4](#step-4-submit-the-request-to-start-the-fine-tuning) below).

### Step 4. Submit the request to start the fine-tuning
Throughout this section and the sections below, we use cURL to send the requests, but you can also send the requests by Python's `requests` or in any other way you want. Also, you can send the requests from the host where the service is running, or from any other machine, e.g., your laptop (as long as you run, e.g., `ssh -L 5000:localhost:5000 -Nf <username>@<host_address>` for local port forwarding on that machine).

Run the following command to submit a request to start the fine-tuning.
```commandline
curl --request POST http://localhost:5000/finetune \
    --header "Content-Type: application/json" \
    --data '{
        "finetune_name": "<finetune_name>",
        "base_model_name_or_path": "command-r-08-2024",
        "parallel_strategy": "fsdp",
        "finetune_strategy": "lora",
        "use_4bit_quantization": "false",
        "gradient_checkpointing": "true",
        "gradient_accumulation_steps": 1,
        "train_epochs": 1,
        "train_batch_size": 16,
        "validation_batch_size": 16,
        "learning_rate": 1e-4,
        "eval_percentage": 0.2,
        "lora_config": {"rank": 8, "alpha": 16, "target_modules": ["q", "k", "v", "o"], "rslora": "true"},
        "wandb_config": {"project": "<wandb_project_name>", "run_id": "<wandb_run_name>"}
    }'
```

The `<finetune_name>` must be exactly the same as that used in [Step 3](#step-3-prepare-the-training-and-evaluation-data). If you are not going to use Weights & Biases for logging during the fine-tuning, the hyperparameter `"wandb_config"` can be removed. See table below for details about all the other hyperparameters we support, where some valid values or ranges below are based on best practices (you do not have to strictly follow them, but if you do not follow them, some validation codes also need to be changed or removed).

| Hyperparameter               | Definition                                                                                               | Default value        | Valid values or range                                                                                                                                     |
|:-----------------------------|:---------------------------------------------------------------------------------------------------------|:---------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------|
| base_model_name_or_path      | The name of the base model or the path to the checkpoint of a customized base model                      | "command-r-08-2024"  | "command-r", "command-r-08-2024", "command-r-plus", "command-r-plus-08-2024", "aya-expanse-8b", "aya-expanse-32b", "/opt/finetuning/<path_to_checkpoint>" |
| parallel_strategy            | The strategy to use multiple GPUs for training                                                           | "fsdp"               | "vanilla", "fsdp", "deepspeed"                                                                                                                            |
| finetune_strategy            | The strategy to train the model                                                                          | "lora"               | "lora"                                                                                                                                                    |
| use_4bit_quantization        | Whether to apply 4-bit quantization to the model                                                         | "false"              | "false", "true"                                                                                                                                           |
| gradient_checkpointing       | Whether to use gradient (activation) checkpointing                                                       | "true"               | "false", "true"                                                                                                                                           |
| gradient_accumulation_steps  | The gradient accumulation steps                                                                          | 1                    | integers, min: 1                                                                                                                                          |
| train_epochs                 | The number of epochs to train                                                                            | 1                    | integers, min: 1, max: 10                                                                                                                                 |
| train_batch_size             | The batch size during training                                                                           | 16                   | integers, min: 8, max: 32                                                                                                                                 |
| validation_batch_size        | The batch size during validation (evaluation)                                                            | 16                   | integers, min: 8, max: 32                                                                                                                                 |
| learning_rate                | The learning rate                                                                                        | 1e-4                 | real numbers, min: 5e-5, max: 0.1                                                                                                                         |
| eval_percentage              | The percentage of data split from training data for evaluation (ignored if evaluation data are provided) | 0.2                  | real numbers, min: 0.05, max: 0.5                                                                                                                         |
| lora_config.rank             | The rank parameter in LoRA                                                                               | 8                    | integers, min: 8, max: 16                                                                                                                                 |
| lora_config.alpha            | The alpha parameter in LoRA                                                                              | 2 * rank             | integers, min: 16, max: 32                                                                                                                                |
| lora_config.target_modules   | The modules to apply LoRA                                                                                | ["q", "k", "v", "o"] | Any non-empty subset of ["q", "k", "v", "o", "ffn_expansion"]                                                                                             |
| lora_config.rslora           | Whether to use rank-stabilized LoRA (rsLoRA)                                                             | "true"               | "false", "true"                                                                                                                                           |

Note that you can set `base_model_name_or_path` as either the name of a supported model or the path to the checkpoint of a customized base model. However, if it is a path, the following requirements must be satisfied:
- The customized base model must have the same architecture as one of the supported models (the weights can be different). For example, it can be a model obtained by fine-tuning a supported model like Command R 08-2024.
- The checkpoint of the customized base model must be a HuggingFace checkpoint like [this](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024/tree/main). It must contain a `config.json` file, as we will use it to infer the type of your model.
- The checkpoint must be put in `<finetune_root_dir>/<path_to_checkpoint>` on your host, and the `base_model_name_or_path` must be in the format of `/opt/finetuning/<path_to_checkpoint>` (recall that we mount `<finetune_root_dir>` on the host to `/opt/finetuning` in the container).

Also note that `finetune_strategy = "lora", use_4bit_quantization = "false"` corresponds to the fine-tuning strategy of LoRA, while `finetune_strategy = "lora", use_4bit_quantization = "true"` corresponds to the fine-tuning strategy of QLoRA.

After the fine-tuning is finished, you can find all the files about this fine-tuning in `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>`. More specifically, our fine-tuning service will automatically create the following folders for you:
- `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>/finetune` that stores all the intermediate results generated during fine-tuning, which contains the following sub-folders:
  - `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>/finetune/data` that stores the preprocessed data (the data split into train & eval and rendered by the Cohere template)
  - `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>/finetune/checkpoints` that stores the checkpoints of adapter weights during fine-tuning
  - `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>/finetune/logs` that stores the Weights & Biases logs
  - `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>/finetune/configs` that stores the configuration files used by fine-tuning
- `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>/output` that stores the final fine-tuning outputs, which contains the following sub-folder:
  - `<finetune_root_dir>/<finetune_sub_dir>/<finetune_name>/output/merged_weights` that stores the final model weights after we merge the fine-tuned adapter weights into the base model weights

At any time (before, during or after the fine-tuning), you can run the following command to check the status of the fine-tuning service.
```commandline
curl --request GET http://localhost:5000/status
```

After you finish the current fine-tuning, you can do another fine-tuning (probably with different data and/or hyperparameters) by doing [Step 3](#step-3-prepare-the-training-and-evaluation-data) and [Step 4](#step-4-submit-the-request-to-start-the-fine-tuning) again, but you must use a different `<finetune_name>`.

### Step 5. Terminate the fine-tuning service
When you do not want to do any more fine-tunings, you can run the following command to termintate the fine-tuning service.
```commandline
curl --request GET http://localhost:5000/terminate
```

### Next steps
Now you have one or more fine-tuned models. If you want to deploy them in production and efficiently serve a large number of requests, here are your options:
- Use [Cohere Bring Your Own Fine-tune](https://aws.amazon.com/marketplace/pp/prodview-5wt5pdnw3bbq6) if your fine-tuned model is [Cohere's Command R 08-2024](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024).
- Contact our sales team by email [sales@cohere.com](mailto:sales@cohere.com) or from [here](https://cohere.com/contact-sales) for on-premises deployments.

## 4. Inference

We also provide a simple inference service to facilitate quick experiments or small-scale evaluations with the fine-tuned models, but this service should not be used for large-scale inference in production.

### Step 1. Build the Docker image

Do [Step 1](#step-1-build-the-docker-image) if you have not done so. Otherwise, skip this step.

### Step 2. Run the Docker container to start the inference service
Run the command below to start the inference service. This command is similar to the one in [Step 2](#step-2-run-the-docker-container-to-start-the-fine-tuning-service). The main difference is that you need to set the environment variable `TASK=INFERENCE` to indicate now you want to do inference, not fine-tuning.
```commandline
docker run -it --rm \
    --name <peft_prod_inference_service_docker_container_name> \
    --gpus <gpus_accessible_by_the_container> \
    --ipc=host \
    --net=host \
    -v ~/.cache:/root/.cache \
    -v <finetune_root_dir>:/opt/finetuning \
    -e ENVIRONMENT=DEV \
    -e TASK=INFERENCE \
    -e HF_TOKEN=<hf_token> \
    <peft_prod_docker_image_name>
```

### Step 3. Submit the request to get model response

Run the following command to submit a request to the fine-tuned model and get model response. Note that this inference service is designed to be similar to [Cohere's Chat API](https://docs.cohere.com/v1/reference/chat), and the port for this inference service is `5001`, not `5000`.
```commandline
curl --request POST http://localhost:5001/inference \
    --header "Content-Type: application/json" \
    --data '{
        "model_name_or_path": "<model_name_or_path>",
        "message": "<message>",
        "chat_history": <chat_history>,
        "preamble": "<preamble>",
        "max_new_tokens": 1024,
        "do_sample": "false"
    }'
```
The parameters are explained below.
- The parameter `model_name_or_path` is required, where `<model_name_or_path>` can be any model name accepted by [CohereForCausalLM.from_pretrained](https://huggingface.co/docs/transformers/main/en/model_doc/cohere#transformers.CohereForCausalLM) or a path to the merged weights of any fine-tuned model in the format of `/opt/finetuning/<finetune_sub_dir>/<finetune_name>/output/merged_weights`.
- The parameter `message` is required, where its value can be any string.
- The parameter `chat_history` is optional, where `<chat_history>` is a list of messages in the format of `[{"role": "user", "content": "<user_content_1>"}, {"role": "assistant", "content": "<assistant_content_1>"}, ...]`.
- The parameter `preamble` is optional, where its value can be any string.
- You must explicitly set the parameter `max_new_tokens` as a large enough number; if you do not set it or set it as a small number, the model generation could be truncated and our inference service currently does not allow such an incomplete generation.
- You can also add any other parameters accepted by the [generate](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) method, e.g., `"do_sample": "false"`, etc.

A caveat is that if the inference service finds the model you want to use is different from the current model it is holding, it will spend some time loading the model you want. Therefore, please do not frequently switch models during inference; you want to finish all the inferences with one model before switching to another model.

You can also run the following command to get some information of the inference service, e.g., the current model it is holding, etc.
```commandline
curl --request GET http://localhost:5001/info
```

### Step 4. Terminate the inference service
When you do not want to do any more inferences, you can run the following command to termintate the inference service.
```commandline
curl --request GET http://localhost:5001/terminate
```

## 5. Development
If you want to write and run your own codes for fine-tuning by, e.g., Jupyter Notebook, you can use our tool in a development mode that provides you with more flexibility and more control on fine-tuning.

### Step 1. Build the Docker image for development
Run the command below to build the Docker image, which may take about 18min to finish if it is the first time you build it on the host.
```commandline
DOCKER_BUILDKIT=1 docker build --rm \
    --ssh default \
    --target peft-dev \
    -t <peft_dev_docker_image_name> \
    -f docker/Dockerfile \
    .
```

You can also edit the `peft-dev` stage in `docker/Dockerfile` to install any apps you need for development, and build your own Docker image for development.

### Step 2. Run the Docker container for development
Run the command below to enter the container for development, where you can mount any directory on the host `<dir_on_host>` to the directory in the container `<dir_in_container>` and do as many mounts as you want. For example, it could be helpful to do `-v ~/.ssh:/root/.ssh` if you want to use your host's `~/.ssh` in the container.
```commandline
docker run -it --rm \
    --name <peft_dev_docker_container_name> \
    --gpus <gpus_accessible_by_the_container> \
    --entrypoint=bash \
    --ipc=host \
    --net=host \
    -v ~/.cache:/root/.cache \
    -v <dir_on_host>:<dir_in_container> \
    -e HF_TOKEN=<hf_token> \
    -e WANDB_API_KEY=<wandb_api_key> \
    <peft_dev_docker_image_name>
```

Now you can start your development work and do anything you want there.
