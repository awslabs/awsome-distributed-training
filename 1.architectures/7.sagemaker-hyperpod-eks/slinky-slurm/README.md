# Running Slurm on HyperPod EKS with Slinky

### What is the Slinky Project?

The [Slinky Project](https://github.com/SlinkyProject/slurm-operator/tree/main) is an
open-source solution maintained by SchedMD (the main developers of Slurm) that deploys Slurm
on Kubernetes. When paired with HyperPod EKS, the Slinky Project unlocks the ability for
enterprises who have standardized infrastructure management on Kubernetes to deliver a
Slurm-based experience to their ML scientists. It also enables training, experimentation,
and inference to happen on the same cluster of accelerated nodes with the built-in resiliency
provided by HyperPod.

---

### Slinky on HyperPod EKS Architecture
![Image Description](./slinky-slurm-hp-eks.png)

The diagram above depicts the resulting proof-of-concept deployment outlined in this guide.
An Amazon EKS cluster acts as an orchestration layer, while a HyperPod cluster delivers a
resilient instance group of GPU accelerated compute nodes. The Slinky Slurm operator is
installed to extend Kubernetes with custom resources and actions, and a containerized Slurm
cluster is deployed using Kubernetes pods via Helm chart. This Slurm cluster includes the
following components:
| Component | Description |
|-----------|-------------|
| Controller (slurmctld) | The central management daemon that monitors resources, accepts jobs, and assigns work to compute nodes. |
| Accounting (slurmdbd) | Handles job accounting and user/project management through a MariaDB database backend. |
| Compute (slurmd) | The worker nodes that execute jobs, organized into NodeSets which can be grouped into different partitions. |
| Login | Provides SSH access points for users to interact with the Slurm cluster and submit jobs. |
| REST API (slurmrestd) | Offers HTTP-based API access to Slurm functionality for programmatic interaction with the cluster. |
| Authentication (sackd) | Manages credential authentication for secure access to Slurm services. |
| MariaDB | The database backend used by the accounting service to store job, user, and project information. |
| Slurm Exporter | Collects and exports Slurm metrics for monitoring purposes. |

The login LoadBalancer type service is annotated to dynamically create an AWS Network Load
Balancer using the
[AWS Load Balancer Controller](https://github.com/kubernetes-sigs/aws-load-balancer-controller),
allowing ML scientists to SSH into their login pods without interfacing with the Kubernetes
API server via kubectl.

The login and compute node pods also have FSx for Lustre and (optionally) FSx for OpenZFS
shared filesystems mounted. Having containerized compute node pods allows many dependencies
that would traditionally be installed manually using Conda or a Python virtual environment to
be baked into the container image, but shared filesystems are still beneficial for storing
training artifacts, data, logs, and checkpoints. If Conda environments are still required,
FSx for OpenZFS has proven optimal to avoid IOPS saturation with many small files.

---

### Release Notes

The following was tested in two infrastructure scenarios for hosting the compute NodeSet pods:
1. On 4 `ml.g5.8xlarge` instances (1 A10G Tensor Core GPU each)
2. On 2 `ml.p5.48xlarge` instances (8 H100 Tensor Core GPUs each) with EFAv2

For simplicity, 2 `ml.m5.2xlarge` instances were also allocated for separately hosting other
components like the Controller and Login pods. You can adjust the number and type of instances
associated with your HyperPod cluster, as well as the component affinity rules in
`slurm-values.yaml.template` to modify how they are spread across your nodes.

Testing used
[Slurm Operator v1.0.1](https://github.com/orgs/slinkyproject/packages/container/package/charts/slurm-operator)
and
[Slurm Cluster v1.0.1](https://github.com/orgs/slinkyproject/packages/container/package/charts/slurm)
Helm charts pulled as OCI artifacts from the Slinky container registry.

Worker pods were built with Python 3.12.8 + PyTorch 2.6.0 + CUDA 12.6 + NCCL 2.23.4 +
EFA Installer 1.38.0 (bundled with OFI NCCL plugin) pre-installed in the container image.
See the [Docker Build for the Slurmd Deep Learning Container](./Docker-Build-README.md)
for details.

* * *

### Quick Start (Automated Deployment)

The automated deployment uses three scripts that handle the entire lifecycle:

```
deploy.sh   →   install.sh   →   (run workloads)   →   destroy.sh
```

#### <u>Prerequisites</u>

- AWS CLI configured with appropriate permissions
- `jq` (for CloudFormation) or `terraform` (for Terraform)
- `kubectl`, `helm`, `eksctl`
- Docker (only if using `--local-build` for container images)

#### <u>Clone the Repository</u>
```
git clone https://github.com/awslabs/awsome-distributed-training.git
cp -r awsome-distributed-training/1.architectures/7.sagemaker-hyperpod-eks/slinky-slurm .
cd slinky-slurm
```

#### <u>Step 1: Deploy Infrastructure</u>

`deploy.sh` deploys the HyperPod EKS cluster via CloudFormation or Terraform, resolves
availability zones, substitutes parameters, and extracts stack outputs to `env_vars.sh`.

Deploy with 4 `ml.g5.8xlarge` instances using CloudFormation:
```
./deploy.sh --node-type g5 --infra cfn
```
Deploy with 2 `ml.p5.48xlarge` instances using CloudFormation:
```
./deploy.sh --node-type p5 --infra cfn
```
Deploy using Terraform:
```
./deploy.sh --node-type g5 --infra tf
```
Override the default region and availability zone:
```
./deploy.sh --node-type g5 --infra cfn --region us-east-1 --az-id use1-az2
```

After the script completes, source the environment variables:
```
source env_vars.sh
```

Run `./deploy.sh --help` for all available options.

#### <u>Step 2: Build Image, Install Slurm</u>

`install.sh` orchestrates `setup.sh` (container image build via CodeBuild, SSH key
generation, Helm values template substitution) followed by MariaDB, Slurm operator,
and Slurm cluster Helm installations and NLB configuration.

Install with CodeBuild image build (default):
```
./install.sh --node-type g5 --infra cfn
```
Install with local Docker build instead of CodeBuild:
```
./install.sh --node-type g5 --infra cfn --local-build
```
Install with an existing ECR image (skip build entirely):
```
./install.sh --node-type g5 --infra cfn --skip-build
```
Re-install Slurm without rebuilding the image or regenerating values:
```
./install.sh --skip-setup
```

Run `./install.sh --help` for all available options.

#### <u>Step 3: Verify the Deployment</u>

Update your kubectl context and verify:
```
aws eks update-kubeconfig --name $EKS_CLUSTER_NAME

kubectl get nodes

kubectl -n slurm get pods -l app.kubernetes.io/instance=slurm
```

#### <u>Clean Up</u>

`destroy.sh` tears down all resources in reverse order (Slurm cluster, operator,
MariaDB, CodeBuild stack, HyperPod infrastructure):

```
./destroy.sh --infra cfn
```

Run `./destroy.sh --help` for all available options.

* * *

<details>
<summary><b>Manual Deployment (Alternative)</b></summary>

### Set Up the HyperPod Cluster:

Deploy the
[HyperPod EKS CloudFormation Stack](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/00-setup/00-workshop-infra-cfn)
or the
[HyperPod EKS Terraform Modules](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/00-setup/01-workshop-infra-tf)
using the provided configurations below.

#### <u>Deploy Using CloudFormation (Manual)</u>

Deploy the HyperPod EKS infrastructure using the official
[SageMaker HyperPod CloudFormation templates](https://github.com/aws/sagemaker-hyperpod-cluster-setup/tree/main/eks/cloudformation).
Use the provided `params.json` file to set the stack parameters.

```
export PARAMS="params.json"
```

NOTE: The `params.json` file defaults to `ml.g5.8xlarge` x 4 (g5 profile) with
`us-west-2` AZ IDs. For p5, edit the `InstanceGroupSettings1` value to use
`ml.p5.48xlarge` x 2. If deploying in a different region, update
`AvailabilityZoneIds`, `FsxAvailabilityZoneId`, and the
`TargetAvailabilityZoneId` values inside `InstanceGroupSettings1` to match
your target region's availability zone IDs.

Set your region and deploy the stack using the S3-hosted template:
```
export AWS_REGION=<your-region-here> # e.g. us-west-2

aws cloudformation create-stack \
  --region $AWS_REGION \
  --stack-name hp-eks-slinky-stack \
  --template-url https://aws-sagemaker-hyperpod-cluster-setup-${AWS_REGION}-prod.s3.${AWS_REGION}.amazonaws.com/templates/main-stack-eks-based-template.yaml \
  --parameters file://$PARAMS \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND
```
Wait for the stack to complete, then set your environment variables from the
stack outputs:
```
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

export STACK_ID=hp-eks-slinky-stack

export EKS_CLUSTER_NAME=$(aws cloudformation describe-stacks \
  --stack-name $STACK_ID --region $AWS_REGION \
  --query "Stacks[0].Outputs[?OutputKey=='OutputEKSClusterName'].OutputValue" \
  --output text)

export VPC_ID=$(aws cloudformation describe-stacks \
  --stack-name $STACK_ID --region $AWS_REGION \
  --query "Stacks[0].Outputs[?OutputKey=='OutputVpcId'].OutputValue" \
  --output text)

export PRIVATE_SUBNET_ID=$(aws cloudformation describe-stacks \
  --stack-name $STACK_ID --region $AWS_REGION \
  --query "Stacks[0].Outputs[?OutputKey=='OutputPrivateSubnetIds'].OutputValue" \
  --output text | cut -d',' -f1)

export SECURITY_GROUP_ID=$(aws cloudformation describe-stacks \
  --stack-name $STACK_ID --region $AWS_REGION \
  --query "Stacks[0].Outputs[?OutputKey=='OutputSecurityGroupId'].OutputValue" \
  --output text)
```

#### <u>Deploy Using Terraform (Manual)</u>

Copy the Terraform modules:
```
cd ..
cp -r awsome-distributed-training/1.architectures/7.sagemaker-hyperpod-eks/terraform-modules .
cd terraform-modules/hyperpod-eks-tf
```
Use the provided `custom.tfvars` file to set the Terraform Module
parameters.

```
cp ../../slinky-slurm/custom.tfvars .
export PARAMS="custom.tfvars"
```

NOTE: The `custom.tfvars` file defaults to `ml.g5.8xlarge` x 4 (g5 profile).
For p5, edit `instance_type` to `ml.p5.48xlarge` and `instance_count` to `2`
in the accelerated instance group before deploying.
Initialize the Terraform modules:
```
terraform init
```
Generate an execution plan to validate the configuration of the Terraform
modules:
```
terraform plan -var-file=$PARAMS
```
Apply the Terraform modules to deploy the specified HyperPod cluster
infrastructure:
```
terraform apply -var-file=$PARAMS
```
Run the `terraform_outputs.sh` script, which populates the `env_vars.sh`
script with your environment variables:
```
cd ..
chmod +x terraform_outputs.sh
./terraform_outputs.sh
cat env_vars.sh
source env_vars.sh
cd ..
```

---

Verify that the required environment variables are set:
```
echo $AWS_ACCOUNT_ID $AWS_REGION $EKS_CLUSTER_NAME $VPC_ID $PRIVATE_SUBNET_ID $SECURITY_GROUP_ID
```
(Optional) Add an EKS access entry (if needed):
```
export ROLE_ARN=arn:aws:iam::$AWS_ACCOUNT_ID:role/<your-role-name-here>

export PLCY_ARN=arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy

aws eks create-access-entry \
 --cluster-name $EKS_CLUSTER_NAME \
 --principal-arn $ROLE_ARN \
 --type STANDARD \
 --region $AWS_REGION

aws eks associate-access-policy \
 --cluster-name $EKS_CLUSTER_NAME \
 --principal-arn $ROLE_ARN \
 --policy-arn $PLCY_ARN \
 --access-scope type=cluster \
 --region $AWS_REGION
```

Update your kubectl context:

```
aws eks update-kubeconfig --name $EKS_CLUSTER_NAME

kubectl get nodes
```

* * *

### Create an FSx for Lustre Storage Class:

> **NOTE:** The HyperPod CloudFormation stack with `CreateFsxStack=true` (default)
> automatically provisions an FSx for Lustre filesystem. The steps below are only
> needed if you want to create additional FSx storage classes or if you deployed
> without the FSx stack.

Create an [IAM OpenID Connect (OIDC)](https://docs.aws.amazon.com/eks/latest/userguide/enable-iam-roles-for-service-accounts.html) identity provider for your cluster:
```
eksctl utils associate-iam-oidc-provider --cluster $EKS_CLUSTER_NAME --approve
```
Create a service account with an IAM role mapped to it for use with the FSx for Lustre CSI driver:
```
eksctl create iamserviceaccount \
  --name fsx-csi-controller-sa \
  --namespace kube-system \
  --cluster $EKS_CLUSTER_NAME \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonFSxFullAccess \
  --approve \
  --role-name FSXLCSI-${EKS_CLUSTER_NAME}-${AWS_REGION} \
  --region $AWS_REGION
```
Verify proper annotation of the service account with the IAM role ARN:
```
kubectl get sa fsx-csi-controller-sa -n kube-system -oyaml
```
Install the [FSx for Lustre CSI Driver](https://github.com/kubernetes-sigs/aws-fsx-csi-driver) using Helm:
```
helm repo add aws-fsx-csi-driver \
 https://kubernetes-sigs.github.io/aws-fsx-csi-driver

helm repo update

helm upgrade --install aws-fsx-csi-driver \
  --namespace kube-system \
  --set controller.serviceAccount.create=false \
  aws-fsx-csi-driver/aws-fsx-csi-driver
```
Verify installation of the FSx for Lustre CSI driver:
```
kubectl get pods -n kube-system \
 -l app.kubernetes.io/name=aws-fsx-csi-driver
```
Create an FSx for Lustre storage class:
```
envsubst < lustre-storageclass.yaml | kubectl apply -f -
```
Note: This example uses [envsubst](https://github.com/a8m/envsubst) to inject the
`PRIVATE_SUBNET_ID` and `SECURITY_GROUP_ID` environment variables into the storage class
Kubernetes manifest. If you don't have envsubst in your development environment, install it
by following the [instructions here.](https://github.com/a8m/envsubst?tab=readme-ov-file#installation)

Verify the `fsx-sc` storage class was created:
```
kubectl get sc fsx-sc -oyaml
```

* * *

### (Optional) Create an FSx for OpenZFS Storage Class:

Create a service account with an IAM role mapped to it for use with the FSx for OpenZFS CSI driver:
```
eksctl create iamserviceaccount \
    --name fsx-openzfs-csi-controller-sa \
    --namespace kube-system \
    --cluster $EKS_CLUSTER_NAME \
    --attach-policy-arn arn:aws:iam::aws:policy/AmazonFSxFullAccess \
    --approve \
    --role-name FSXOCSI-${EKS_CLUSTER_NAME}-${AWS_REGION} \
    --region $AWS_REGION
```
Verify proper annotation of the service account with the IAM role ARN:
```
kubectl get sa fsx-openzfs-csi-controller-sa -n kube-system -oyaml
```
Install the [FSx for OpenZFS CSI driver](https://github.com/kubernetes-sigs/aws-fsx-openzfs-csi-driver) using Helm:
```
helm repo add aws-fsx-openzfs-csi-driver \
    https://kubernetes-sigs.github.io/aws-fsx-openzfs-csi-driver

helm repo update

helm upgrade --install aws-fsx-openzfs-csi-driver \
    --namespace kube-system \
    --set controller.serviceAccount.create=false \
    aws-fsx-openzfs-csi-driver/aws-fsx-openzfs-csi-driver
```
Verify installation of the FSx for OpenZFS CSI driver:
```
kubectl get pods -n kube-system \
 -l app.kubernetes.io/part-of=aws-fsx-openzfs-csi-driver
```
Create an FSx for OpenZFS Storage Class:
```
envsubst < openzfs-storageclass.yaml | kubectl apply -f -
```
Note: This example uses [envsubst](https://github.com/a8m/envsubst) to inject the
`PRIVATE_SUBNET_ID` and `SECURITY_GROUP_ID` environment variables into the storage class
Kubernetes manifest. If you don't have envsubst in your development environment, install it
by following the [instructions here.](https://github.com/a8m/envsubst?tab=readme-ov-file#installation)

Verify the `openzfs-sc` storage class was created:
```
kubectl get sc openzfs-sc -oyaml
```

* * *

### Install Slinky Prerequisites:

> **NOTE:** The HyperPod CloudFormation/Terraform stack automatically installs
> cert-manager, Prometheus, the GPU operator, and the EFA device plugin.
> These steps are only needed if your stack was deployed without these
> components enabled.

```
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add jetstack https://charts.jetstack.io

helm repo update

helm install cert-manager jetstack/cert-manager \
	--namespace cert-manager --create-namespace --set crds.enabled=true

helm install prometheus prometheus-community/kube-prometheus-stack \
	--namespace prometheus --create-namespace --set installCRDs=true
```

Verify prerequisite installation:

```
 kubectl get all -n cert-manager
 kubectl get all -n prometheus
```

* * *

### Install the AWS Load Balancer Controller:

Following the instructions below, which are a consolidation of the full
[Install with Helm](https://docs.aws.amazon.com/eks/latest/userguide/lbc-helm.html)
instructions found in the Amazon EKS documentation:

Create the IAM policy to give the AWS Load Balancer Controller permission to make calls
to AWS APIs on your behalf:
```
curl -O https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/refs/heads/release-2.13/docs/install/iam_policy.json

aws iam create-policy \
    --policy-name AWSLoadBalancerControllerIAMPolicy-v2.12.0 \
    --policy-document file://iam_policy.json
```

Create a service account with an IAM role mapped to it for use with the AWS Load Balancer
Controller:
```
eksctl create iamserviceaccount \
    --cluster=$EKS_CLUSTER_NAME \
    --namespace=kube-system \
    --name=aws-load-balancer-controller \
    --attach-policy-arn=arn:aws:iam::$AWS_ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy-v2.12.0 \
    --override-existing-serviceaccounts \
    --region $AWS_REGION \
    --approve
```
Verify proper annotation of the service account with the IAM role ARN:
```
kubectl get sa aws-load-balancer-controller -n kube-system -oyaml
```
Install the AWS Load Balancer Controller using Helm:
```
helm repo add eks https://aws.github.io/eks-charts
helm repo update

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=$EKS_CLUSTER_NAME \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller \
  --set region=$AWS_REGION \
  --set vpcId=$VPC_ID
```
Verify installation of the AWS Load Balancer Controller:
```
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
```

* * *

### Install MariaDB for Slurm Accounting:

Install the MariaDB operator and create a MariaDB instance for Slurm accounting:

```
helm repo add mariadb-operator https://helm.mariadb.com/mariadb-operator
helm repo update

helm install mariadb-operator mariadb-operator/mariadb-operator \
  --version 25.10.4 --namespace mariadb --create-namespace

kubectl create ns slurm
kubectl apply -f mariadb.yaml
```

Wait for the MariaDB instance to become ready:
```
kubectl get mariadb -n slurm --watch
```

* * *

### Install the Slurm Operator:

Install the
[Slurm Operator](https://github.com/SlinkyProject/slurm-operator/tree/main/helm/slurm-operator#slurm-operator)
release v1.0.1:

```
helm install slurm-operator oci://ghcr.io/slinkyproject/charts/slurm-operator \
  --version=1.0.1 --namespace=slinky --create-namespace
```

Verify Slurm Operator installation:

```
kubectl get all -n slinky
```

* * *

### Install the Slurm Cluster:

The `slurm-values.yaml.template` file contains a consolidated Helm values template
with shell variables for node-type-specific settings (instance type, GPU count, EFA
interfaces, replicas, GRES). The `setup.sh` script (or `install.sh` which calls it)
resolves these variables based on your `--node-type` selection and produces a
`slurm-values.yaml` file.

If using the automated flow, `install.sh` handles this automatically. For manual
installation, first generate the values file:

```
./setup.sh --node-type g5 --infra cfn --skip-build
```

Then deploy the Slurm cluster:
```
helm install slurm oci://ghcr.io/slinkyproject/charts/slurm \
  --values=slurm-values.yaml --version=1.0.1 --namespace=slurm
```

Watch the deployment status of the Slurm cluster:

```
kubectl -n slurm get pods -l app.kubernetes.io/instance=slurm --watch
```

Verify the deployment status of all components:

```
kubectl get all -n slurm
```

* * *

### Configure the Login Network Load Balancer:

> **NOTE:** If you used `install.sh`, the NLB is already configured via
> `slurm-login-service-patch.yaml`. These manual steps are only needed for
> standalone deployments.

Identify two public subnets in your VPC. If you used the
[default VPC configuration](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US/00-setup/02-additional-info#default-vpc-networking-architecture),
two public subnets were provisioned for you:
```
export PUBLIC_SUBNET_ID_1=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${VPC_ID}" "Name=map-public-ip-on-launch,Values=true" --query "Subnets[0].SubnetId" --output text)

export PUBLIC_SUBNET_ID_2=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${VPC_ID}" "Name=map-public-ip-on-launch,Values=true" --query "Subnets[1].SubnetId" --output text)

echo $PUBLIC_SUBNET_ID_1 $PUBLIC_SUBNET_ID_2
```
Add annotations to the `slurm-login-slinky` service to make it internet-facing using the
public subnets:

```
kubectl annotate service slurm-login-slinky -n slurm \
  service.beta.kubernetes.io/aws-load-balancer-type="nlb" \
  service.beta.kubernetes.io/aws-load-balancer-scheme="internet-facing" \
  service.beta.kubernetes.io/aws-load-balancer-nlb-target-type="ip" \
  service.beta.kubernetes.io/aws-load-balancer-subnets="$PUBLIC_SUBNET_ID_1,$PUBLIC_SUBNET_ID_2" \
  service.beta.kubernetes.io/aws-load-balancer-healthcheck-port="22" \
  --overwrite

kubectl describe service slurm-login-slinky -n slurm
```

The AWS Load Balancer Controller actively watches for and implements annotation changes.
It automatically adds inbound rules to the node security group to allow traffic from the
NLB security group on the target port (22 in this case).

---

#### Create an FSx for Lustre Persistent Volume Claim (PVC) in the slurm namespace:

Create the slurm namespace (if not already created):

```
kubectl create ns slurm
```

Create a PVC named `fsx-claim` in the slurm namespace:

```
kubectl apply -f lustre-pvc-slurm.yaml
```

Verify FSx for Lustre PVC creation:

```
kubectl get pvc -n slurm

# check for a bound state
kubectl get pvc fsx-claim  -n slurm -ojson \
 | jq -r .status.phase

# get the the volume ID
kubectl get pv $(kubectl get pvc fsx-claim  -n slurm -ojson \
 | jq -r .spec.volumeName) -ojson \
 | jq -r .spec.csi.volumeHandle
```
---

#### (Optional) Create an FSx for OpenZFS PVC in the slurm namespace:

Create a PVC named `openzfs-claim` in the slurm namespace:
```
kubectl apply -f openzfs-pvc-slurm.yaml
```
Verify FSx for OpenZFS PVC creation:
```
kubectl get pvc -n slurm

# check for a bound state
kubectl get pvc openzfs-claim  -n slurm -ojson \
 | jq -r .status.phase

# get the volume ID
kubectl get pv $(kubectl get pvc openzfs-claim -n slurm -ojson \
 | jq -r .spec.volumeName) -ojson \
 | jq -r .spec.csi.volumeHandle
```

* * *

### Clean Up (Manual):

Uninstall the Slurm cluster and the Slurm operator:
```
helm uninstall slurm -n slurm
helm uninstall slurm-operator -n slinky
```
Uninstall MariaDB:
```
kubectl delete mariadb mariadb -n slurm
helm uninstall mariadb-operator -n mariadb
```
Uninstall the Prometheus operator and cert-manager (if manually installed):
```
helm uninstall prometheus -n prometheus
helm uninstall cert-manager -n cert-manager
```
Delete the FSx persistent volume claims:
```
kubectl delete pvc fsx-claim -n slurm
kubectl delete pvc openzfs-claim -n slurm
```
Delete the FSx storage classes:
```
kubectl delete sc fsx-sc
kubectl delete sc openzfs-sc
```
Uninstall the FSx CSI drivers and delete the IAM roles mapped to their service accounts:
```
helm uninstall aws-fsx-csi-driver -n kube-system
helm uninstall aws-fsx-openzfs-csi-driver -n kube-system

eksctl delete iamserviceaccount \
  --name fsx-csi-controller-sa \
  --namespace kube-system \
  --cluster $EKS_CLUSTER_NAME

eksctl delete iamserviceaccount \
  --name fsx-openzfs-csi-controller-sa \
  --namespace kube-system \
  --cluster $EKS_CLUSTER_NAME
```
Uninstall the AWS Load Balancer Controller and delete the IAM role mapped to its
service account:
```
helm uninstall aws-load-balancer-controller -n kube-system

eksctl delete iamserviceaccount \
  --name aws-load-balancer-controller \
  --namespace kube-system \
  --cluster $EKS_CLUSTER_NAME

aws iam delete-policy --policy-arn arn:aws:iam::$AWS_ACCOUNT_ID:policy/AWSLoadBalancerControllerIAMPolicy-v2.12.0
```

Delete the HyperPod EKS CloudFormation stacks:
```
aws cloudformation delete-stack --stack-name $STACK_ID --region $AWS_REGION
```
Delete the HyperPod EKS Terraform modules:
```
cd terraform-modules/hyperpod-eks-tf
terraform plan -destroy -var-file=custom.tfvars
terraform destroy -var-file=$PARAMS
```

</details>

* * *

### Basic Tests:

SSH into the login node as root from the NLB endpoint:

```
SLURM_LOGIN_HOSTNAME="$(kubectl get services -n slurm -l app.kubernetes.io/instance=slurm,app.kubernetes.io/name=login -o jsonpath="{.items[0].status.loadBalancer.ingress[0].hostname}")"

ssh -i ~/.ssh/id_ed25519_slurm -p 22 root@$SLURM_LOGIN_HOSTNAME
```
---

Check the available nodes:

```
sinfo

PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
hp-node      up   infinite      4   idle hp-node-[0-3]
all*         up   infinite      4   idle hp-node-[0-3]
```
Note that in both scenarios (using 4 `ml.g5.8xlarge` instances or 2 `ml.p5.48xlarge`
instances) we should see the same number of slurm compute nodes. When running on 4
`ml.g5.8xlarge` instances, each slurm compute node is mapped to 1 available A10G GPU,
whereas when running on 2 `ml.p5.48xlarge` instances, each slurm compute node is mapped
to 8 available H100 GPUs and 32 EFA network interfaces.

---

Verify FSx for Lustre and OpenZFS filesystem mounts on the login pod:

```
df -h

# Filesystem                                             Size  Used Avail Use% Mounted on
# overlay                                                500G   30G  471G   6% /
# tmpfs                                                   64M     0   64M   0% /dev
# tmpfs                                                   63G     0   63G   0% /sys/fs/cgroup
# 10.1.12.93@tcp:/7c5dpb4v                               1.2T  7.8M  1.2T   1% /fsx
# fs-03221b7c7d3767607.fsx.us-west-2.amazonaws.com:/fsx   64G     0   64G   0% /home
# tmpfs                                                  115G  4.0K  115G   1% /etc/slurm
# /dev/nvme0n1p1                                         100G   23G   78G  23% /run
# /dev/nvme1n1                                           500G   30G  471G   6% /etc/hostname
# shm                                                     64M     0   64M   0% /dev/shm
# tmpfs                                                  115G  4.0K  115G   1% /etc/sssd/sssd.conf
# tmpfs                                                  115G   12K  115G   1% /etc/ssh/ssh_host_rsa_key
# tmpfs                                                   63G     0   63G   0% /proc/acpi
# tmpfs                                                   63G     0   63G   0% /sys/firmware

exit
```
---

Verify FSx for Lustre and OpenZFS filesystem mounts on the compute node pods:

```
kubectl -n slurm exec -it pod/slurm-compute-hp-node-0 -- bash --login

df -h

# Filesystem                                             Size  Used Avail Use% Mounted on
# overlay                                                500G   31G  470G   7% /
# tmpfs                                                   64M     0   64M   0% /dev
# tmpfs                                                   63G     0   63G   0% /sys/fs/cgroup
# 10.1.12.93@tcp:/7c5dpb4v                               1.2T  7.5M  1.2T   1% /fsx
# fs-03221b7c7d3767607.fsx.us-west-2.amazonaws.com:/fsx   64G     0   64G   0% /home
# tmpfs                                                  115G  4.0K  115G   1% /etc/slurm
# /dev/nvme0n1p1                                         100G   23G   78G  23% /run
# /dev/nvme1n1                                           500G   31G  470G   7% /etc/hostname
# shm                                                     64M     0   64M   0% /dev/shm
# tmpfs                                                  115G     0  115G   0% /var/log/slurm
```
---

Check the installed CUDA compiler version on compute node pods:

```
nvcc --version

# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on Tue_Oct_29_23:50:19_PDT_2024
# Cuda compilation tools, release 12.6, V12.6.85
# Build cuda_12.6.r12.6/compiler.35059454_0
```
---

Check the NCCL version on compute node pods:

```
ldconfig -v | grep "libnccl.so" | tail -n1 | sed -r 's/^.*\.so\.//'

# 2.23.4
```
---

Confirm NCCL headers are installed worker node pods:

```
find /usr/local/lib/ -name "nccl.h" 2>/dev/null

# /usr/local/lib/python3.12/site-packages/torch/include/torch/csrc/cuda/nccl.h
```
---

Check EFA availability:
```
ls /sys/class/infiniband/
fi_info -p efa
```
Check that the EFA libraries are properly mounted
```
ls /opt/amazon/efa/lib
ls /opt/amazon/ofi-nccl/lib/x86_64-linux-gnu
```
Verify EFA device allocation:
```
ls -l /dev/infiniband/
```
Verify intra-node GPU topology:
```
nvidia-smi topo -m
```
For `ml.p5.48xlarge` instances, the GPU topology should show all GPUs are connected via
NVLink (NV18 indicates 18 NVLink connections). The GPUs are split across two NUMA nodes
(0-3 on NUMA 0, 4-7 on NUMA 1).

---

### FSDP Test

SSH into the login pod as root, clone the repo, and create a checkpoints directory:

```
SLURM_LOGIN_HOSTNAME="$(kubectl get services -n slurm -l app.kubernetes.io/instance=slurm,app.kubernetes.io/name=login -o jsonpath="{.items[0].status.loadBalancer.ingress[0].hostname}")"

ssh -i ~/.ssh/id_ed25519_slurm -p 22 root@$SLURM_LOGIN_HOSTNAME

# install git
apt update
apt install -y git
git --version

# install vim (optional)
apt install -y vim
vim --version

cd /fsx
git clone https://github.com/awslabs/awsome-distributed-training/
cd awsome-distributed-training/3.test_cases/pytorch/FSDP/slurm

mkdir -p checkpoints
```
---
Copy the modified sbatch file:
```
export SLINKY_PATH=/fsx/awsome-distributed-training/1.architectures/7.sagemaker-hyperpod-eks/slinky-slurm

# for g5 instances
cp ${SLINKY_PATH}/g5/g5-llama2_7b-training.sbatch ./llama2_7b-training.sbatch

# for p5 instances
cp ${SLINKY_PATH}/p5/p5-llama2_7b-training.sbatch ./llama2_7b-training.sbatch
```
---
Add your Hugging Face token to stream the
[allenai/c4](https://huggingface.co/datasets/allenai/c4) dataset without throttling:
```
NEW_TOKEN="your_new_token_here"
sed -i "s/export HF_TOKEN=.*$/export HF_TOKEN=$NEW_TOKEN/" llama2_7b-training.sbatch
```

---
Kick-off the training job:
```
sbatch llama2_7b-training.sbatch
```
---

Watch the output logs from the login pod:

```
export JOB_ID=$(squeue -h -u root -o "%i" | head -1)

tail -f logs/llama2_7b-FSDP_${JOB_ID}.out
```
---

Watch the error logs from `slurm-compute-hp-node-0`:

```
# from a new terminal window
kubectl -n slurm exec -it pod/slurm-compute-hp-node-0 -- bash --login

cd /fsx/awsome-distributed-training/3.test_cases/pytorch/FSDP/slurm
export JOB_ID=$(squeue -h -u root -o "%i" | head -1)

watch "grep 'Batch.*Loss' logs/llama2_7b-FSDP_${JOB_ID}.err"

# or

tail -f logs/llama2_7b-FSDP_${JOB_ID}.err | grep --line-buffered 'Batch.*Loss'
```

Watch squeue from `slurm-compute-hp-node-1`:

```
# from a new terminal window
kubectl -n slurm exec -it pod/slurm-compute-hp-node-1 -- bash --login

# 1 second updates
watch -n 1 squeue
```

Watch checkpoints from `slurm-compute-hp-node-2`:

```
# from a new terminal window
kubectl -n slurm exec -it pod/slurm-compute-hp-node-2 -- bash --login

cd /fsx/awsome-distributed-training/3.test_cases/pytorch/FSDP/slurm

# highlight changes, show timestamps, 5 second updates
watch -n 5 -d "ls -lh checkpoints"
```

* * *

### Development & Testing:

The deployment scripts and their helper library `lib/deploy_helpers.sh` are
tested using [bats-core](https://github.com/bats-core/bats-core). The test
suite covers argument parsing, node profile resolution, Helm profile
resolution, AZ validation, CloudFormation parameter substitution (jq),
Terraform variable overrides (sed/awk), and template variable substitution.

```
# One-time setup: install bats-core
brew install bats-core            # macOS
sudo apt-get install -y bats      # Debian/Ubuntu
npm install -g bats               # cross-platform

# One-time setup: install bats helper libraries
bash tests/install_bats_libs.sh

# Run all tests (45 tests)
bats tests/test_deploy.bats
```
