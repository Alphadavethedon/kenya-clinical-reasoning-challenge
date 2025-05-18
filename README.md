# 🇰🇪 Kenya Clinical Reasoning Challenge

## Overview

This repository contains the solution to the [Kenya Clinical Reasoning Challenge](https://zindi.africa/competitions/kenya-clinical-reasoning-challenge), hosted on Zindi. The competition involves predicting clinicians' responses to 400 authentic clinical vignettes, simulating real-world decision-making in rural Kenyan healthcare settings.

## Problem Statement

Participants are tasked with predicting the clinician's response based on a given prompt, which includes a nurse's background and a complex medical scenario. The goal is to develop models that can match real clinician reasoning in resource-constrained environments.

## Dataset

The dataset consists of:

- **400 clinical vignettes**: Each vignette combines a nurse's background with a complex medical scenario.
- **Features**: Includes nurse demographics, patient information, and medical history.
- **Target**: The clinician's response to the vignette.

## Approach

1. **Data Preprocessing**: Clean and preprocess the dataset to handle missing values, encode categorical variables, and normalize numerical features.
2. **Model Selection**: Experiment with various machine learning models, including:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting Machines
   - Neural Networks
3. **Evaluation**: Use appropriate metrics to evaluate model performance and select the best-performing model.

## Installation

Clone this repository:

```bash
git clone https://github.com/Alphadavethedon/kenya-clinical-reasoning-challenge.git
cd kenya-clinical-reasoning
