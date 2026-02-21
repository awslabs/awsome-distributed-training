## PEFT fine tuning of Llama 3 on Slurm cluster (trn1/trn1n)

This example showcases how to train llama 3 models using AWS Trainium instances and Huggingface Optimum Neuron. 🤗 Optimum Neuron is the interface between the 🤗 Transformers library and AWS Accelerators including AWS Trainium and AWS Inferentia. It provides a set of tools enabling easy model loading, training and inference on single- and multi-Accelerator settings for different downstream tasks.

## Prerequisites
Before running this training, you'll need to create a SageMaker HyperPod cluster with at least 1 trn1.32xlarge/ trn1n.32xlarge instance group. Instructions can be found in the [Cluster Setup](https://catalog.workshops.aws/sagemaker-hyperpod/en-US/01-cluster) section. 

You will also need to complete the following prerequisites for configuring and deploying your SageMaker HyperPod cluster for fine tuning:

* Submit a service quota increase request to get access to Trainium instances in your AWS Region. You will need to request an increase for Amazon EC2 Trn1 instances, ml.trn1.32xlarge or ml.trn1n.32xlarge.
* Locally, install the AWS Command Line Interface (AWS CLI); the required minimum version needed is 2.14.3.
* Locally, Install the AWS Systems Manager Session Manager Plugin in order to SSH into your cluster.


Additionally, since Llama 3 is a gated model users have to register in Huggingface and obtain an [access token](https://huggingface.co/docs/hub/en/security-tokens) before running this example.

### Training

## Step 1: Download training scripts

Begin by downloading the training scripts from the aws-awesome-distributed repo:

```bash
cd ~/
git clone https://github.com/awslabs/awsome-distributed-training

cd ~/awsome-distributed-training/3.test_cases/pytorch/optimum-neuron/llama3/slurm/fine-tuning
```

## Step 2: Setup Python Environment

Setup a virtual python environment and install your training dependencies. Make sure this repo is stored on the shared FSx volume of your cluster so all nodes have access to it.

```bash
sbatch submit_jobs/0.create_env.sh
```

View the logs created by the scripts in this lab by running this command below. You can update it for the step you are currently running:

```bash
tail -f logs/0.create_env.out 
```

Before proceeding to the next step throughout this lab, check if the current job has finished by running:

```bash
squeue
```

## Step 3: Download the model

Next, you will download the model to your FSx file volume. Begin by logging into Hugging Face using your access token mentioned in the prerequisite steps. With your access token set, you should now be able to download the model.

First modify the `submit_jobs/1.download_model.sh` script to include the Hugging Face access token before running it:

```bash
export HF_TOKEN="<Your Hugging Face Token>"
```

Then trigger the script to download the Llama3 model. 

```bash
sbatch submit_jobs/1.download_model.sh
```

Now that your SageMaker HyperPod cluster is deployed and your environment is setup up, you can start preparing to execute your fine tuning job. 

## Step 4: Compile the model

Before you begin training on Trainium with Neuron, you will need to pre-compile your model with the [neuron_parallel_compile CLI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/training/pytorch-neuron-parallel-compile.html) which reduces the compilatin time during execution. This will trace through the model’s training code and apply optimizations to improve performance.

```bash
sbatch submit_jobs/2.compile_model.sh
```
The compilation process will generate NEFF (Neuron Executable File Format) files that will speed up your model’s fine tuning job. 

## Step 5: Fine Tuning

With your model compiled, you can now begin fine tuning your Llama 3 model. 

For the purposes of this workshop, we will use the [dolly 15k dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k). As part of the training process, the script below will download the dataset and format it in a way that the model expects. Each data point will contain an **instruction** that guides the model’s task, optional **context** that provides background information, and **response** that represent the desired output.

Now submit the fine tuning job:

```bash
sbatch submit_jobs/3.finetune.sh
```

## Step 6: Model Weight Consolidation

After training has completed, you will have a new directory for your model checkpoints. This directory will contain the model checkpoint shards from each neuron device that were generated during training. Use the model consolidation script to combine the shards into a single `model.safetensors` file.

```bash
sbatch submit_jobs/4.model_consolidation.sh
```

The `model.safetensors` file will contain the LoRA weights of your model that were updated during training. 

## Step 7: Merge LoRA Weights

After consolidating the model shards, merge the LoRA adapter weights back to your base Llama 3 model:

```bash
sbatch submit_jobs/5.merge_lora_weights.sh
```
Your final fine tuned model weights will be saved to the final_model_path directory. You can find or update the path in the script `submit_jobs/5.merge_lora_weights.sh` using the argument `--final_model_path`.

## Step 8: Validate your trained model
Now that your model is fine tuned, see how its generations differ from the base model for the dolly-15k dataset. 

```bash
sbatch submit_jobs/6.inference.sh
```

This will generate a prediction for the question “Who are you?”, comparing the response of the base model to the fine tuned model. It will also pass a system prompt to the model to always respond like a pirate. 

Before fine tuning:

```
{
    'role': 'assistant', 
    'content': "Arrrr, me hearty! Me name be Captain Chat, the scurviest pirate chatbot to 
    ever sail the Seven Seas! Me be here to regale ye with tales o' adventure, answer yer 
    questions, and swab the decks o' yer doubts! So hoist the colors, me matey, and let's 
    set sail fer a swashbucklin' good time!"
}
```

After fine tuning:

```
{
    'role': 'assistant', 
    'content': "Arrr, shiver me timbers! Me be Captain Chat, the scurviest pirate chatbot to ever sail the Seven Seas! Me been programmin' me brain 
    with the finest pirate lingo and booty-ful banter to make ye feel like ye just stumbled
    upon a chest overflowin' with golden doubloons! So hoist the colors, me hearty, and 
    let's set sail fer a swashbucklin' good time!"
 } 
```

And that's it! You've successfully fine tuned a Llama 3 model on Amazon SageMaker HyperPod using PEFT with Neuron. 
