---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [x] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [x] Create a trigger workflow for automatically building your docker images
* [x] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:
> 3

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s194245, s194262, s194264, s173955, s222968

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:


For the machine learning aspect of the project, Pytorch was utilized to define the model architecture and implement the dataloaders. To reduce the amount of boilerplate code, the high-level Pytorch Lightning framework was employed, simplifying the training process by using the Trainer object. Huggingface was used to import a pretrained DistilBert model, which greatly simplified the model implementation.

Hydra was employed to ensure reproducibility. This approach separated the definition of hyperparameters from the model itself, making version control of the hyperparameters much easier. Additionally, Weights and Biases was used as a logging tool, providing insights into each training run, such as reporting losses.

Docker played a crucial role in the deployment phase. Marrying the building of images and containers in the cloud ensured system-level reproducibility of the model pipeline. 

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

Miniconda was used to manage the depenencies for this project. To automatically generate the list of dependencies ```pip freeze > requirements.txt``` can be used to extract all install dependencies from an environment. However, in order to limit the size of the requirements file it was manually updated.

Assuming you already have Anaconda installed and cloned the repository, run the following commands to get a complete copy of the development environment:

```bash
conda create -n mlops python=3.11
```

Activate the environment:

```bash
conda activate mlops
```

And install the project dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```
Additionally, make sure to install Pytorch based on you computer compatibilities following this guide: https://pytorch.org/get-started/locally/

In order too pull down the data, please authorize with the following link to Google Drive, https://drive.google.com/drive/folders/1apqcOMgmfkuDp4VGnCcmN6Bwx3dE1GC- and run this command afterwards:
```bash
dvc pull
```

Lastly, for the user's specific wandb configuration change the `wandb` part of the `configs/config.yaml` to reflect the correct project, team and entity. Without any renaming, the defaults should suffice. Otherwise, refer to the wandb documentation.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

In our project, we adopted the cookiecutter data-science template as the foundational structure. Key files such as predict_model.py and train_model.py were developed, along with additional utility files, for instance, ml_ops_detect_ai_generated_text /data/make_dataset.py, which is responsible for dataset construction. We have included a folder named 'tests', which contains our unit tests related to the data and the model. We have not used the "vizulation" folder, however this would be naturally populated by key metrics etc. in a normal project. Additionally, a '.dvc' folder has been created to hold metadata files associated with data versioning, utilizing cloud services.



### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

Ruff was utilized as both a linter and formatter using GitHub Actions to ensure PEP8 compliancy. By implementing it this way these checks are made automatically when pushing to the main branch. In our case multiple redundant import statements were removed by this check. 

In larger projects with a big team collaborating, ensuring a cohesive formatting helps with code readability and alleviates compatibility issues. Furthermore, introducing docstrings and typing would help end users understand the code in a standardized format. 


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we have implemented 2 tests. One concerns the training of the model and the other concerns the dataset size. Another test could have been implemented with regards to the predict module, and all of these things together are cornerstones of the application output.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

We have a coverage of 100% when running our tests. However, this may be misleading as it tests the `tests` folder alone using pytest. Coverage of a 100% does not coincide with no corner cases being unhandled by the source code of the application, rather just how much of the code is executed at runtime. It is a good indicator however of proper testing. 

### Question 9

> **Did your workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

For this project both branches, forks, and pull requests were used. Implementing new features and working on distinct issues helps keep overview of the development of the pipeline. Upon completion of the given feature the branch was merged back into the main branch after review. Ideally, this is done by assigning another team member as a reviewee, however this was not always done for this project. This promotes a collaborative environment among the project contributors and maintainers and ensures that the code merged into the main "production" branch is thoroughly vetted beforehand.


### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

The project data was stored with DVC in Google Drive, however when experimenting with the Google Cloud Platform (GCP) we used their Data Storage Buckets still managed with DVC. The outcome is in essence the same. The version control of the data enables one to track changes, similarly to GitHub, which can help in development and production to maintain uniformity among developers, and enables one to quickly find and fix new bug instances based on the historical records. Moreover, it addresses the challenges of storing large data sets in git, and instead merely store a pointer. 


### Question 11

> **Discuss you continuous integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:


The organization of our CI has been split into 2 files: the pytests.yml file runs all the tests and computes the coverage. The codecheck.yml file ensures that the code in the repository is PEP8 compliant and performs linting using the ruff package. Currently, these tests are merely executed on an Ubuntu Linux OS, but could be extended to macOS and Windows easily. We opted to not do it initially due to a ceiling of 2.000 minutes on GitHub Actions with a free account. Another idea could be to implement a workflow, which runs a docker instance on the GCP and ensures that the building and pushing of the repository in the container to the cloud is succesful. Due to time constraints and the requirement of a secrets file, we opted to use our time on other pressing matters.  An example of the GitHub Actions workflow triggers and its history can be seen [here](https://github.com/DanielHolmelund/ml_ops_detect_ai_generated_text/actions/workflows/codecheck.yml). We do also not test multiple Python Versions, which is probably a good practice for an application within reasonable limits, however we deliberately specify a Python version in the setup of the repository, hence this is in the end a intentional choice for the project.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

The configuration pipeline is managed using Hydra, which creates a config folder, wherein different experiments can be defined. The Hydra decorater is input in the model training file and points to a config file, which in turn points to a specific experiment that contains a set of hyper parameters for the execution thereof. This enables a streamlined and reproducable approach with logging enabled for future users. In order to initiate an experiment, assuming one has followed the setup of the repository and virtual environment requirements, one simply executes the following line in the terminal `python ml_ops_detect_ai_generated_text/train_model.py`.



### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Configuration files play a crucial role in our setup. These files help in clearly linking hyperparameters with the corresponding model performance, leaving no ambiguity. Additionally, DVC is implemented to ensure data consistency across different machines. Furthermore, the use of Docker facilitates the creation of a stable environment, maintaining specific library versions, dependencies, and packages. GCP or cloud computing can help configure the specific hardware setup, if the information and resources are readily available. This setup is vital to prevent any variations in library behaviors from impacting our results. For the Huggingface and PyTorch frameworks we set a specific seed as a hyperparameter in the config files to ensure reproducability in terms of batch shuffling, etc. 


### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

![image](https://github.com/DanielHolmelund/ml_ops_detect_ai_generated_text/assets/114672733/43d1aec0-9f3f-45f7-ab2e-58c8cf92aa7d)

In our experiments with Weights and Biases (WandB), we meticulously tracked crucial metrics during the training and evaluation phases of our DistilBERT model. The metrics captured include training loss, validation loss, and accuracy. These metrics serve as vital indicators of our model's performance and its ability to classify AI-generated texts from Language Model Models (LLMs).

The training loss is a measure of how well the model is learning from the training data. A decreasing trend in training loss indicates that the model is effectively minimizing errors during training, while validation loss provides insights into the model's generalization performance on unseen data. A consistent decrease in both training and validation loss signifies that the model is learning relevant patterns without overfitting to the training set. Monitoring accuracy helps us gauge the model's capability to make correct predictions, which is especially significant in the context of our classification task.

While the primary focus is on MLOps, tracking these metrics in WandB provides a valuable diagnostic tool. Any anomalies or unexpected patterns in the loss graphs or accuracy trends can signal potential issues with the model's training process, guiding decisions on hyperparameter adjustments or further refinements.


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

In our experiments, Docker played a pivotal role in containerizing our machine learning applications, providing a consistent and reproducible environment for both local development and deployment on Google Cloud Platform (GCP). We employed two Dockerfiles, each serving a distinct purpose in our MLOps pipeline.

The `trainer.dockerfile` defined an image tailored for local model training. This image was utilized to train our DistilBERT model using existing data. The trained model's state was then saved in a shared folder, allowing for easy access and further analysis. On the other hand, the `predict.dockerfile` image was designed for local deployment as well, with its primary focus on loading the trained model, running tests on existing data, and evaluation the model's performance. 

For deployment on GCP, we orchestrated the containerization process using the `cloudbuild.yaml` file. This configuration file triggered the build process, which involved executing docker compose up to create the containerized application. Subsequently, the resulting Docker image was pushed to the Google Container Registry, facilitating seamless deployment on GCP.

This Docker-based approach ensures that ML experiments are executed in an isolated and reproducible environment. It also streamlines the deployment process, allowing for easy scaling and deployment on cloud infrastructure.  Here is a link to the [trainer.dockerfile](https://github.com/DanielHolmelund/ml_ops_detect_ai_generated_text/blob/main/dockerfiles/train_model.dockerfile)


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

Debugging methods may be dependent on the group member, but generally the debugging tool in VSCode was used due to the familiarity with its uses in terms of stepping through the code. The python debugger pdb was also used once or twice. We used the PyTorch profiler, which said the attention mechanisms of the model used the most cpu time, which makes sense. Otherwise, there are probably room for improvements somewhere, as we did not extensively profile all of the codebase. 

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

On the Google Cloud Platform, we utilized a variety of services. For remote storage of our data and Docker images, we relied on Cloud Storage service. Here, we established a GCP Bucket, designated as the remote repository for our data version control, allowing us to efficiently push to and pull data from the cloud. Furthermore, we also used it for saving the models for deployment. The Cloud Engine was used to create VM's and run code. 

We configured triggers to automate the building of trainer and predictor Docker images. These triggers were activated by any push to the main branch of our cloud repository, which mirrors our git repository. All the container images created throughout the project were stored and are accessible in the Container Registry (Soon only Artifact Registry).

For building and deploying the inference API, we tried employing Cloud Run and Cloud Functions, but due to extensive complications we could not make it work for the project, only simple scripts. This was partly due to ports and memory issues. 


### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

The Virtual Machines (VMs) can be defined according to the users needs, which in our case primarily meant implementing a ubuntu debian-based VM with the latest PyTorch version available. This is offered for CPUs and in some instances and regions also GPUs. The Compute Engine service was used to run the code remotely and perform the tasks of the application, ie. training and evaluation of the model. Moreover, we created an instance from the docker container in the Cloud Storage, which is triggered each time a new push to main is executed. Ideally, the Vertex AI service could have been used to run custom jobs and use their logging services, but this proved to be a challenge to implement.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

The Data bucket:
![image](https://github.com/DanielHolmelund/ml_ops_detect_ai_generated_text/assets/57216460/0e3c6468-5aa5-44d6-8149-8230cde8ef25)
The trained model bucket:
![image](https://github.com/DanielHolmelund/ml_ops_detect_ai_generated_text/assets/57216460/1a47bc03-0ea5-4760-be87-0c67a6cd8d61)


### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![image](https://github.com/DanielHolmelund/ml_ops_detect_ai_generated_text/assets/57216460/6728d265-80cb-4fbf-9380-fbe879d9f416)


### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![image](https://github.com/DanielHolmelund/ml_ops_detect_ai_generated_text/assets/114672733/1936c4ce-eba0-43e0-bc04-9695ade2c7c9)



### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

For deployment we utilized the Streamlit framework to create an interactive app. Steamlit allows us to connect the main branch of the repository ensuring an updated app with the newest developed features. Using the google API our pretrained model is loaded from the cloud bucket, which allows for fast inference time of the end user. Streamlit allows for 1GB RAM usage for each app, which both allows for the model to be loaded into the application and cache alleviating wait time for the end user. The deployed model can either be accessed locally by running:
```bash
streamlit run app/streamlitapp.py
```
or accessed remotely in [here](https://detectaigeneratedtextmlops.streamlit.app/). Since we used this approach the FastAPI module was not further explored in combination with either Cloud Run or Cloud Functions services.

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

In general, this subject wasn't given top priority because Google Cloud appears to be quite comprehensive in its monitoring capabilities, covering aspects like security, user activity, and performance. Moreover, there's an option to establish a maximum on the number of requests. Additionally, "Cloud Armor Network Security" offers safeguards against threats such as DDoS attacks. The reason for no monitoring is primarily due to time constraints since the implementation of our application on the cloud proved difficult.

Other monitoring could include data drifting, wherein re-training models are required as the features created by the model would become un-representative of the present as newer data comes in. 


### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

Primarily Cloud Storage and Engine was the most pricey service. 

s194264 used 21$ of the alloted 50$.

s194245 - 7.91$ 

s222968 - 14.02$

s173955 - 3.86$

s194262 - 6.41$


## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![image](https://github.com/DanielHolmelund/ml_ops_detect_ai_generated_text/assets/57216460/40e2b177-a23d-45a1-9dc2-54f6fce873b2)

The process begins with a local setup, where the project is cloned from Git into a virtual environment, ensuring a controlled development environment.

For effective collaboration and version control, Data Version Control (DVC) is employed alongside Git, providing a systematic approach to manage both data and code versions. Google Cloud Platform (GCP) is utilized for storage and as a robust cloud infrastructure, facilitating scalability and accessibility.

Continuous Integration (CI) is implemented using GCP's Cloud Build, triggered by new pushes to the main Git branch. The build process is defined in the cloudbuild.yaml file, utilizing Docker and Docker Compose for containerization. This ensures reproducibility and consistency across different environments. The CI workflow enforces PEP8 compliance, code linting, and unit testing, enhancing code quality and reliability.

To streamline the deployment process, the system integrates with Streamlit. While manual deployment is currently in place, the architecture could readily support automation through triggers, allowing for a more efficient and automated deployment process, but due to time constraints this venture was not fully explored.

Furthermore, key tools like PyTorch, Anaconda, VSCode, Hydra, Weights and Biases (WandB), and GitHub were seamlessly integrated, enhancing the development experience and enabling efficient collaboration, tracking and reproducibility. 


### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Configuring our deployment setup and getting everything up and running on Google Cloud with Docker turned out to be the toughest and most time-consuming task for us. A lot of this difficulty came from our team's relative unfamiliarity with Google Cloud's features and how they operate with docker, etc. Even just getting the hang of the interface was quite a task. We had to rely heavily on a mix of trial and error, along with a lot of detailed reading of tutorials and guides, to get our whole pipeline operational. It was a bit of a journey, but in the end, we managed to get everything working as intended, but leaving quite a lot of unexplored potential in terms of monitoring and other tasks presented in later exercises, including but not limited to the exhaustive check list, which we would have liked to explore given the opportunity.


### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

As mandated by the formal requirement, which emphasizes the need for transparency in individual contributions to ensure fairness, the group and its members wish to affirm that each member has participated and delivered a commendable and closely equal contribution across the different sections of the project and repository. Generally, each person contributed side-by-side in regards to debugging several different aspects of the development cycle. Should further clarification be required, please feel free to contact any of the group members.
