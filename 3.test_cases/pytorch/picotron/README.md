# Picotron Test Cases

This test case demonstrates distributed training of [Picotron](https://github.com/huggingface/picotron), a distributed training framework for education and research experimentation. Picotron is designed for training speed, scalability, and memory efficiency.


## Tested Configurations

| Instance | GPUs | Status | Notes |
|----------|------|--------|-------|
| p5en.48xlarge | 8 x H200 80 GB | Untested | Expected to work |
| p5.48xlarge | 8 x H100 80 GB | Untested | Expected to work |
| p4de.24xlarge | 8 x A100 80 GB | Untested | Expected to work |
| g5.12xlarge | 4 x A10G 24 GB | Untested | May need smaller model sizes |

> See the [Instance Compatibility Guide](../../../docs/instance-compatibility.md)
> for parameter adjustments needed across instance types.

## Build Environment

The provided Dockerfile (`picotron.Dockerfile`) will set up the environment with all required dependencies:

```bash
docker build -t picotron -f picotron.Dockerfile .
```

First, create a Hugging Face account to retrieve a [token](https://huggingface.co/settings/tokens.). Log in to your account and create an access token from Hugging Face Tokens. 

Save the token onto the head node and download the Llama model:

## Get huggingface token

```bash
huggingface-cli login
```

You will be prompted to input the token. Paste the token and answer `n` when asked to add the token as a git credential.

```

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) n
Token is valid (permission: read).
Your token has been saved to /fsx/ubuntu/.cache/huggingface/token
Login successful
```

Then use the saved token `${HF_TOKEN}` to create configuration.
## Next Step 

Each subdirectory contains example configurations for training different types of models:

- `SmolLM-1.7B`: Training a 1.7B parameter LLM model on CPU/GPU using 3D parallelism. 

