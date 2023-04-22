# Forest cover type classification


The project concerns classification of forest cover type from cartographic variables using classical machine learning methods and neural networks.

The repository contains:
1. Python script for data analysis, model trainings, visualisation: cover_type.py
2. Jupyter-notebook with a step-by-step data analysis and models training: cover_type.ipynb
3. Files containing data from the process of finding optimal hyperparameters for Neural Networks: history_training_balanced_data.pkl and history_training_unbalanced_data.pkl
4. Files with trained models: dummy_best.joblib, decision_tree_best.joblib, random_forest_best.joblib and neural_network_best.h5
5. Deployment proposal using Flask: flask_script.py
6. Dockerfile
7. requirement.txt for Docker image creation

Docker hub link:
https://hub.docker.com/layers/nikolajanik/covertype/latest/images/sha256:52278556f629f04cf01cb7e3f3301a7a5d09cf3be185dbde3d5ce75dc56f935f

Comments for Docker image generation, push and model deployment are included in the jupyter notebook.

