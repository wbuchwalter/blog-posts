---
layout:     post
title:      Creating a Kubernetes Cluster with GPU support on Azure for ML Training and Predictions (With Autoscaling)
date:       2016-03-23 23:02:18
summary:    
categories: container docker machine learning kubernetes gpu training
---

## Creating a Kubernetes cluster using `acs-engine`

First, we need to create a Kubernetes cluster that supports GPUs.  
To do so, we will use `acs-engine`, a tool that will generate the ARM template we need to deploy our cluster with everything configured.    
  
Kubernetes 1.6 introduced multi-gpu support, so that's the version we are going to need, but currently, `acs-engine` doesn't support 1.6 (PR in progress [here](https://github.com/Azure/acs-engine/pull/470)).  
So, in the mean time we are going to use [this fork of acs-engine](https://github.com/wbuchwalter/acs-engine/tree/k8s-gpu), where k8s has been updated.    

Also, with this fork, the NVIDIA drivers are automatically installed for you on every node, making autoscaling much easier (this will be possible soon on the official repository, [with this PR](https://github.com/Azure/acs-engine/pull/415)).  

Clone the repo, and checkout the `k8s-gpu` branch.

``` 
> git clone https://github.com/wbuchwalter/acs-engine
> git checkout k8s-gpu
```

Then we need to specify what our cluster should look like. To do so we are going to edit `example\kubernetes.json` by filling the different parameters.    

The interesting part is the `agentPoolProfiles` section.  
There you can define a number of different pools. Each pool can have a different VM size, and can scale up to 100 nodes.   
You should define a separate pool for every different VM size you intend to use.    

For example, if you plan to use your cluster both for training with GPU and inference with CPU, you should at least specify two pools.     

The number of agent isn't really important because we are going to enable autoscaling, so you can keep everything to 1.    

Azure has68 different VM sizes that have GPU support, you can see the details [here](https://azure.microsoft.com/en-us/blog/azure-n-series-preview-availability/).  

Here is what your `kubernetes.json` should look like more or less:  
```json
{
  "apiVersion": "vlabs",
  "properties": {
    "orchestratorProfile": {
      "orchestratorType": "Kubernetes"
    },
    "masterProfile": {
      "count": 1,
      "dnsPrefix": "mygpucluster",
      "vmSize": "Standard_D2_v2"
    },
    "agentPoolProfiles": [
      {
        "name": "agentpool1",
        "count": 1,
        "vmSize": "Standard_NC6",
        "availabilityProfile": "AvailabilitySet"
      },     
      {
        "name": "agentpool3",
        "count": 1,
        "vmSize": "Standard_D2_v2",
        "availabilityProfile": "AvailabilitySet"
      }
    ],
    "linuxProfile": {
      "adminUsername": "azureuser",
      "ssh": {
        "publicKeys": [
          {
            "keyData": "somekey"
          }
        ]
      }
    },
    "servicePrincipalProfile": {
      "servicePrincipalClientID": "xxxxx",
      "servicePrincipalClientSecret": "xxxxx"
    }
  }
}
```

Let's generate the ARM template, you'll need docker to be installed on your machine.

```
> ./script/devenv.sh
> make build
> acs-engine example/kubernetes.json
```

This should have generated a bunch of files under the `_output/Kubernetes-XXXXX` directory, including the ARM template and parameters that we want.  
    
To deploy them, you can create a new `template deployment` in the Azure portal and copy paste `azurdeploy.json` and `azuredeploy.parameters.json`.  
**Make sure you choose a region that has N-Series VM available! South Central US is one of them.**  
Or use [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli):

```
> cd _output/Kubernetes-XXXXX
> az group create --location southcentralus --name myk8scluster
> az group deployment create --template-file azuredeploy.json --parameters @azuredeploy.parameters.json --resource-group myk8scluster
```

This should take between 5 and 10 minutes to deploy.  
Do not delete the generated `azurdeploy.json` and `azuredeploy.parameters.json` as we will need them later for autoscaling.  
  
Once the deployment is completed, we are going to grab the Kubernetes config file of our cluster to be able to issue commands locally.

```
> scp azureuser@<dnsname>.southcentralus.cloudapp.azure.com:.kube/config ~/.kube/config
```

If you don't have `kubectl` installed, now is the time: [Installing and Setting Up kubectl](https://kubernetes.io/docs/tasks/kubectl/install/)

## Testing the cluster  

Let's check our new cluster
```
> kubectl get nodes
NAME                        STATUS                     AGE
k8s-agentpool1-19661165-0   Ready                      1m
k8s-agentpool2-19661165-0   Ready                      23s
k8s-master-19661165-0       Ready,SchedulingDisabled   1m
```
One master, and two agents, one for each pool.
Let's describe one of our agents:

```
> kubectl describe node k8s-agentpool1-19661165-0
[...]
Capacity:
 alpha.kubernetes.io/nvidia-gpu:	1
 cpu:					6
 memory:				57703024Ki
 pods:					110
[...]
```
We can see that the drivers have been correctly installed since kubernetes has been able to find our GPU device.  
Let's run `nvidia-smi` a deployment to prove that GPU support is working correctly:

Download this [kubernetes template](https://github.com/wbuchwalter/acs-k8s-gpu/blob/master/nvidia-smi.yaml) somewhere and deploy it on your cluster (don't worry about the details of the `.yaml` file, we will get into this later on):
```
> kubectl create -f nvidia-smi.yaml
```
The deployment will take some time to finish, but you should eventually see:
```
> kubectl get pods --show-all
NAME                       READY     STATUS      RESTARTS   AGE
nvidia-smi-fcg8j           0/1       Completed   0          1m
```  

And if we check the logs:  

```
> kubectl logs nvidia-smi-fcg8j
Thu Apr 13 22:52:56 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.20                 Driver Version: 375.20                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | AEC6:00:00.0     Off |                    0 |
| N/A   45C    P0    73W / 149W |      0MiB / 11471MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Creating an image to run on our cluster

Here are the things I wanted my container to be able to do (you might have different requirements, that's fine):
* Ability to train our model, or predict using the same container depending on a flag.
* The app should save model checkpoints when the training is done to an azure blob storage, and restore the model before serving prediction.
* Ability to run on CPU or GPU.

You might already have an image that you want to use on your cluster for training/inference, in that case you can skip ahead.  
  
If not, you can use this example repository: [wbuchwalter/tf-app-container-sample](https://github.com/wbuchwalter/tf-app-container-sample).  
It is a very simple model in TensorFlow based on the MNIST sample. You can specify `--train` to start the container in training mode, otherwise a Flask application will serve (random) predictions on the `/predict` route to demonstrate serving.  

## Training our model on the cluster

To train our model on our Kubernetes cluster, we are going to define a [job](https://kubernetes.io/docs/concepts/jobs/run-to-completion-finite-workloads/) template.  
Because we want to train using GPUs, we have to specify how many of them we need: 
```yaml
resources:
    limits:
      alpha.kubernetes.io/nvidia-gpu: 2
```
We also need to expose the NVIDIA drivers from the host into the container. This is a bit tricky for now.
The correct mount paths will depend on many things: how you installed the drivers, which host OS you are using etc.  

Because the official Tensorflow image is based on Ubuntu 16.04, just like our host VMs in ACS/ACS-engine, we can simply map the `bin` and `lib` directories right away without too much issues.  

Here is what the template looks like if you used the `tf-app-container-sample` container on acs-engine (note the use of `--train` flag): 

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  labels:
    app: tensorflow-trainer
  name: tensorflow-trainer
spec:
  template:
    metadata:
      name: tensorflow-trainer
    spec:
      volumes:
      - name: binaries
        hostPath:
          path: /usr/bin/
      - name: libraries
        hostPath:
          path: /usr/lib/x86_64-linux-gnu
      containers:
      - args:
        - --train
        image: wbuchwalter/tf-app-container-sample:gpu
        name: tensorflow-trainer
        env:
        - name: STORAGE_ACCOUNT_NAME
          value: <key>
        - name: STORAGE_ACCOUNT_KEY
          value: <key>
        resources:
          limits:
            alpha.kubernetes.io/nvidia-gpu: 2
        volumeMounts:
        - mountPath: /usr/bin/
          name: binaries
        - mountPath: /usr/lib/x86_64-linux-gnu
          name: libraries
      restartPolicy: Never
```

As you can see, we are directly mounting `/usr/bin/` and `/usr/lib/x86_64-linux-gnu` into the container. This two directories contain everything needed to communicate with the GPU.

To create this job, simply `kubectl create -f template.yaml`.
If we look at the logs (`kubectl logs -f <pod-id>`), sure enough:

```
[...]
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: b25e:00:00.0)
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:1) -> (device: 1, name: Tesla K80, pci bus id: b8c3:00:00.0)
[...]
Extracting /app/MNIST_data/train-images-idx3-ubyte.gz
Extracting /app/MNIST_data/train-labels-idx1-ubyte.gz
Extracting /app/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting /app/MNIST_data/t10k-labels-idx1-ubyte.gz
0.9169
Model saved in file: /tmp/ckp/model
```

## Autoscaling

But what happens if we want to train multiple models in parallel?  
Currently, once our cluster is busy, any subsequent training would be scheduled sequentially.
This isn't great when trying to compare multiple hypothesis, or when a team of data scientist are working on different stuff on the same cluster.  
We could just create a cluster with a lot of nodes instead, but the price of a GPU capable VM is pretty steep, so this isn't a possibility for many companies.  

Instead, we are going to use an autoscaler. This autoscaler will run inside the cluster and monitor the different pods that get scheduled. Whenever a pod is pending because of a lack of resources, the autoscaler will create an adequate number of new VMs to match this need. And when VMs are idle, the autoscaler will delete them.  

That way we can achieve the flexibility we want, while still keeping costs down.

You can find the autoscaler here: [wbuchwalter/kubernetes-acs-autoscaler](https://github.com/wbuchwalter/kubernetes-acs-autoscaler) (a fork of [OpenAI's great work](https://github.com/openai/kubernetes-ec2-autoscaler)).

Follow the instructions in the [`README`](https://github.com/wbuchwalter/Kubernetes-acs-autoscaler/blob/master/README.md) to get the necessary credentials and create a Kubernetes secret.  
Once this is done, you'll need to tweak `scaling-controller.yaml`:
* Provide the resource group name with `--resource-group`
* Remove the `--container-service-name` parameter, because we are using `acs-engine` and not ACS.
* Provide a link to the `azuredeploy.json` file that we generated earlier via `--template-file-url`. You can use a secret [Gist](https://gist.github.com/) or an Azure storage link.
* Do the same thing for `azuredeploy.parameters.json` and provide the link to `--parameters-file-url`.

Once this is done, create the controller with `kubectl create -f scaling-controller.yaml`.

If you look at the logs of the pod, you should see something similar to this:

```
2017-04-17 19:51:38,515 - autoscaler.cluster - INFO - ++++++++++++++ Running Scaling Loop ++++++++++++++++
2017-04-17 19:51:39,113 - autoscaler.cluster - INFO - Pods to schedule: 0
2017-04-17 19:51:39,113 - autoscaler.cluster - INFO - ++++++++++++++ Scaling Up Begins ++++++++++++++++
2017-04-17 19:51:39,114 - autoscaler.cluster - INFO - Nodes: 2
2017-04-17 19:51:39,114 - autoscaler.cluster - INFO - To schedule: 0
2017-04-17 19:51:39,114 - autoscaler.cluster - INFO - Pending pods: 0
2017-04-17 19:51:39,114 - autoscaler.cluster - INFO - ++++++++++++++ Scaling Up Ends ++++++++++++++++
2017-04-17 19:51:39,114 - autoscaler.cluster - INFO - ++++++++++++++ Maintenance Begins ++++++++++++++++
2017-04-17 19:51:39,114 - autoscaler.cluster - INFO - ++++++++++++++ Maintaining Nodes ++++++++++++++++
2017-04-17 19:51:39,114 - autoscaler.cluster - INFO - node: k8s-agentpool1-14254244-0                                                   state: spare-agent
2017-04-17 19:51:39,115 - autoscaler.cluster - INFO - node: k8s-agentpool2-14254244-0                                                   state: spare-agent
2017-04-17 19:51:39,115 - autoscaler.cluster - INFO - ++++++++++++++ Maintenance Ends ++++++++++++++++
```

Now, if we schedule 2 training jobs at the same time on the cluster, and look at the logs once again:

```
2017-04-17 19:54:01,044 - autoscaler.cluster - INFO - ++++++++++++++ Running Scaling Loop ++++++++++++++++
2017-04-17 19:54:01,639 - autoscaler.cluster - INFO - Pods to schedule: 1
2017-04-17 19:54:01,639 - autoscaler.cluster - INFO - ++++++++++++++ Scaling Up Begins ++++++++++++++++
2017-04-17 19:54:01,639 - autoscaler.cluster - INFO - Nodes: 2
2017-04-17 19:54:01,639 - autoscaler.cluster - INFO - To schedule: 1
2017-04-17 19:54:01,639 - autoscaler.cluster - INFO - Pending pods: 1
2017-04-17 19:54:01,639 - autoscaler.cluster - INFO - ========= Scaling for 1 pods ========
[...]
2017-04-17 19:54:01,640 - autoscaler.cluster - INFO - New capacity requested for pool agentpool2: 2 agents (current capacity: 1 agents)
2017-04-17 21:13:09,654 - autoscaler.deployments - INFO - Deployment started
```

We can see that a new VM is being created to accommodate our second training job. 
After a few minutes, the new VM is up, and our second job starts running.  
  
Once the jobs are completed, our cluster is idle.    
The autoscaler will notice this and adjust the cluster size accordingly.  

First, idle VMs will be cordoned and drained, allowing Kubernetes to reschedule any running pod on another VM if needed.

```
autoscaler.cluster - INFO - node: k8s-agentpool1-32238962-1                                                   state: under-utilized-drainable
autoscaler.kube - INFO - cordoned k8s-agentpool1-32238962-1
autoscaler.kube - INFO - Deleting Pod kube-system/kube-proxy-ghr3z
autoscaler.kube - INFO - drained k8s-agentpool1-32238962-4
```

After some time, the cordoned node will get deleted

```
autoscaler.cluster - INFO - node: k8s-agentpool1-32238962-1                                                   state: idle-unschedulable
autoscaler.container_service - INFO - deleting node k8s-agentpool1-32238962-1
autoscaler.container_service - INFO - Deleting VM
autoscaler.container_service - INFO - Deleting NIC
autoscaler.container_service - INFO - Deleting OS disk
```


> If you see any mistake in this post, or have any question, feel free to contribute or to open an issue on the [GitHub repo](https://github.com/wbuchwalter/blog-posts).
