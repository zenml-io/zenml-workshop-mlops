# ZenML Pipeline Conversion Workshop

Convert traditional ML notebooks into production-ready ZenML pipelines! This workshop teaches you how to transform messy, unstructured ML code into clean, reproducible, and scalable MLOps workflows.

## 🎯 Workshop Objectives

By the end of this workshop, you will:

- **Understand the problems** with traditional notebook-based ML workflows
- **Learn ZenML fundamentals** including steps, pipelines, and artifacts
- **Convert messy ML code** into clean, structured ZenML pipelines
- **Implement both training and inference pipelines** with proper separation of concerns
- **Experience the benefits** of MLOps best practices including versioning, tracking, and reproducibility

## 📁 Workshop Structure

```
workshop-scaffold/
├── 📊 data/
│   ├── customer_churn.csv          # Sample dataset (generated)
│   └── generate_sample_data.py     # Script to create dataset
├── 📓 workshop_notebook.ipynb      # Traditional ML workflow (BEFORE)
├── 🔧 training_pipeline_scaffold.py    # Training pipeline template (TODO)
├── 🔮 inference_pipeline_scaffold.py   # Inference pipeline template (TODO)
├── 📦 requirements.txt             # All necessary dependencies
└── 📖 README.md                   # This file
```

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install uv
uv pip install -r requirements.txt

# Initialize ZenML
zenml init
zenml login
zenml integration install gcp github -y
zenml stack set zenml-workshop-local-stack
```

### 2. Explore the Traditional Workflow

Open and run `workshop_notebook.ipynb` to see a typical data scientist's workflow:

```bash
# Start Jupyter
jupyter notebook workshop_notebook.ipynb
```

**🔴 Notice the Problems:**
- Hardcoded file paths
- Mixed concerns in single cells
- Poor model versioning (`model_final_v2_actually_final_BEST.pkl`)
- Manual preprocessing steps
- No experiment tracking
- Difficult to reproduce

## 📚 Workshop Activities

### Activity 1: Analyze the Traditional Workflow (10 minutes)

Run through `workshop_notebook.ipynb` and identify:
- What could go wrong in production?
- How hard would it be to collaborate on this code?
- What happens when you need to retrain the model?

### Activity 2: Convert to ZenML Training Pipeline (30 minutes)

Work on `training_pipeline_scaffold.py`:

### Activity 3: Create your production stack (10 minutes)

Now that your pipeline ran locally, let's try running it on a remote orchestrator - to do this, you need to create a new stack:

![Create Stack](assets/create_stack.png)

Go for the Manual stack creation
![Manual Stack Creation](assets/manual.png)

Pick the orchestrator, artifact store, image builder and container registry.

![Pick Components](assets/pick.png)

Finally go into your Terminal and set this new stack with:

```
zenml stack set <STACK NAME>
```

Now run your training pipeline again, and see what happens.

### Activity 4: Create a Run Template

Congratulations, you have run your pipeline in the remote environment. You can now create what is called a `Run Template` in ZenML. You can read more about them [here](https://docs.zenml.io/concepts/templates)

To create a run template, head on over to the dashboard and find the training pipeline that you just ran on your production stack.

![Select Run](assets/select_run.png)

Now just click on the "New Template" button on top, give it a name and create it.

![Create New Template](assets/new_template.png)

Congrats, you now have a Run Template. You can now tie this into larger workflows by calling the ZenML API. You can also go through the dashboard, change the configuration as needed, an run the template. 

First navigate to the Run Templates section in your project ...

![Select it](assets/select_template.png)

... open your template ...

![Open it](assets/click_button.png)

.. and click on `Run Template`

![Run it](assets/run_template.png)

1) You can now adjust the step parameters
2) ... and run it


### Activity 5: Convert to ZenML Inference Pipeline (25 minutes)

Work on `inference_pipeline_scaffold.py`:

### Activity 6: Compare and Reflect (10 minutes)

Run both approaches and discuss:
- What's better about the ZenML version?
- How does artifact tracking work?
- What would deployment look like?

## 🏆 Key Learning Points

### Traditional ML Workflow Problems

| Problem | Example | Impact |
|---------|---------|---------|
| **Hardcoded Paths** | `pd.read_csv('data/file.csv')` | Breaks when files move |
| **Mixed Concerns** | Training + evaluation in one cell | Hard to debug/modify |
| **Poor Versioning** | `model_final_v2_BEST.pkl` | Can't track what changed |
| **Manual Steps** | Copy-paste preprocessing | Inconsistent between train/inference |
| **No Tracking** | Print statements for metrics | Can't compare experiments |

### ZenML Solutions

| ZenML Feature | Benefit |
|---------------|---------|
| **Steps** | Single responsibility |
| **Pipelines** | Clear workflow DAG |
| **Artifacts** | Automatic versioning |
| **Type Hints** | Better lineage tracking |
| **Caching** | Skip unchanged steps |

## 🔍 Expected Outputs

After completing the workshop:

### ✅ Working Training Pipeline
- Clean, modular steps
- Automatic artifact storage
- Experiment tracking
- Reproducible runs

### ✅ Working Inference Pipeline  
- Consistent preprocessing
- Model loading from registry
- Batch prediction capability
- Timestamped outputs

### ✅ Better ML Practices
- Version control friendly code
- Easy collaboration
- Production deployment ready
- Monitoring capabilities

## 🎓 Solutions

If you get stuck, check the `solutions`

- `training_pipeline_complete.py` - Fully implemented training pipeline
- `inference_pipeline_complete.py` - Fully implemented inference pipeline

## 🚀 Running the Solutions

```bash
# Run the complete training pipeline
python training_pipeline_complete.py

# Run the complete inference pipeline  
python inference_pipeline_complete.py

# View your pipeline runs
zenml pipeline runs list

# Explore artifacts
zenml artifact list
```

## 🔧 ZenML Commands Reference

```bash
# Initialize ZenML
zenml init

# Login to ZenML
zenml login

# Set ZenML Stack
zenml stack set zenml-workshop-stack

# View pipelines
zenml pipeline list

# View pipeline runs
zenml pipeline runs list

# View artifacts
zenml artifact list

# View models
zenml model list

# Start ZenML dashboard
zenml up
```

## 📈 Next Steps After Workshop

1. **Add More Steps**: Data validation, feature engineering, model comparison
2. **Integrate MLflow**: For enhanced experiment tracking
3. **Add Deployment**: Using ZenML's deployment capabilities
4. **Set Up Monitoring**: Track model performance over time
5. **Cloud Integration**: Deploy to AWS, GCP, or Azure
6. **Team Collaboration**: Share pipelines and artifacts

## 💡 Production Considerations

### For Real-World Usage:

**Data Validation**
```python
@step
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    # Check schema, data quality, distributions
    return df
```

**Model Monitoring**
```python
@step  
def monitor_predictions(predictions: pd.DataFrame) -> dict:
    # Track prediction distributions, detect drift
    return monitoring_metrics
```

**A/B Testing**
```python
@step
def compare_models(model_a: Model, model_b: Model) -> Model:
    # Statistical comparison, champion/challenger
    return best_model
```

## 🆘 Troubleshooting

**Common Issues:**

1. **Import Errors**: Make sure you've installed all requirements
2. **File Not Found**: Run the data generation script first
3. **ZenML Not Initialized**: Run `zenml init`
4. **Permission Errors**: Check file permissions in working directory

**Getting Help:**
- ZenML Documentation: https://docs.zenml.io/
- ZenML Discord: https://zenml.io/slack-invite
- GitHub Issues: https://github.com/zenml-io/zenml/issues

## 🎉 Workshop Completion

Congratulations! You've learned how to:

✅ Identify problems in traditional ML workflows  
✅ Structure ML code using ZenML steps and pipelines  
✅ Implement artifact versioning and experiment tracking  
✅ Create production-ready training and inference pipelines  
✅ Experience the benefits of MLOps best practices  

**Keep learning**: Try applying these concepts to your own ML projects! 