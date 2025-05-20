# ✅ Deep-Learning-MLOps

## 🔢 Steps

1. Data Ingestion   # Download, extract, and prepare raw data
2. Create the base model   # Download the VGG16 with Conv. layers only, add the customized dense layers to it, and save both models
3. Train the base model   # Train the model on processed data
4. Evaluate the base model with MLflow  # Log metrics, params, and model using MLflow
5. Create the prediction pipeline   # Build serving logic for inference (e.g., API/UI integration)
6. Develop the App
7. Setup the AWS EC2, ECR, IAM and Jenkins
8. Setup the GitHub Actions secrets
9. Trigger the Pipeline

## Workflow for each step, from step 1 to 5

1. Update the config.yaml. # changeble vars and urls
2. Update the params.yaml & read it. # Specify hyperparameters and tunable settings
3. Update the entity & read it. # Create dataclasses to structure config _return-type_ of functions
4. Update the configuration manager in src config. # Parse YAMLs, instantiate entity configs
5. Update the components. # Write the logic for this step (e.g., download, train, evaluate)
6. Update the pipeline. # Sequence component calls using the pipeline class
7. Update the main.py. # Create entry/end point to trigger the pipeline with logging
8. Update the dvc.yaml. # Track dependencies, outputs, and automate pipeline with DVC

## How to run?

```bash
conda create -n chest python=3.8 -y
```

```bash
conda activate chest
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

## Git commands

```bash
git add .

git commit -m "Updated"

git push origin main
```

### Why DVC (Data Version Control)?
In your setup (image classifier, training via EC2 and Jenkins, Docker + ECR), DVC helps by:

🛠️ Structuring your ML workflow into stages (e.g., data prep → training → evaluation)

📦 Storing large files (datasets, models) outside Git (in S3, GDrive, etc.) while still tracking versions

📈 Making experiments reproducible — anyone can re-run your full pipeline with dvc repro

🔁 Helping Jenkins or other automation tools track whether files or stages changed

🔍 Tracking hyperparameters and model performance — using params.yaml and metrics.yaml for transparent experimentation and tuning

### DVC cmd
```bash
  dvc init  # Initialize DVC in your repo
  dvc repro # Re-run pipeline stages as needed
  dvc dag   # Visualize pipeline dependencies graphically
```
## AWS and Jenkins Setup

1. Create EC2-1 machine for Jenkins (Ubuntu 22, RAM >= 4GB, Disk >= 32GB) + set Elastic IP + Update/upgrade + AWS access key configuration
2. Create IAM user (Add AdministratorAccess permission)
3. Create ECR Repository for the App
4. Install Jenkins and Docker on EC2-1
5. Install SSH Agent plugin on Jenkins
6. Setup the Credincials (Here, 5 Credintials as are included in the Jenkinsfile)
7. Create the Pipeline in Jenkins and link it to your Github repo (plus the Jenkinsfile path, e.g., .jenkins/Jenkinsfile)
8. Create EC2-2 machine for the App (Ubuntu 22, t2.large, RAM >= 8GB, Disk >= 32 GB ) + Update/upgrade + AWS access key configuration
9. Install Docker + setup
10. Add required Secrets in Github
11. Trigger the Pipeline
