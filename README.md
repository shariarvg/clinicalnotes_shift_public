# **MIMshift: A Benchmark and Framework for Dataset Shift Estimation in Clinical Notes**

Monitoring the robustness of deep prediction models to concept drift is challenging when the exogenous variables are unstructured, particularly in the case of text. Clinical notes provide a natural setting for text drift quantification because clinician note-writing practices are constantly evolving and degradation in prediction model performance can literally be a life-or-death problem. We seek to describe shifts that emerged in the MIMIC-IV-Note database between 2008 and 2020, identify shifts that cause degradation in extraction and prediction models, and formulate a framework for predicting model degradation from notes alone. For the last objective, we jointly build a benchmark of real (temporal) and synthetic (perturbation) shifts, and evaluate the performance of existing high-dimensional statistical metrics on these shifts. We go beyond the classic representations of clinical notes by understanding how post-training refinements for sparsity and style invariance can both affect model robustness and provide a glimpse into features that the model is learning from notes. 

This repository contains my work **in progress** on analyzing clinical notes. I share some of the tools I've built for inducing and detecting shifts, as well as for measuring the degradation these shifts cause. Some good prerequisites for understanding this are
- Data Use: [The MIMIC Database](https://archive.physionet.org/physiobank/database/mimicdb/)
- Primary metric that I evaluate: [Maximum Mean Discrepancy](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html)
- The base model for evaluation is [NVIDIA/UF Gatortron-Base](https://www.nature.com/articles/s41746-022-00742-2)
- My primary interpretability framework (work in progress) is motivated by [HypotheSAE](https://arxiv.org/pdf/2502.04382)

## Overview

Clinical notes and future outcomes arrive at different times, but estimating degradation in model performance is extremely vital for a robust algorithm pipeline. We create a framework for estimating uncertainty at the time of discharge, using only information about the new batch of notes and the historical notes training set.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5226e73-484e-4db8-a66b-3ece9eaefc53" width="600">
</p>

Our primary approach for obtaining an estimate of shift is by featurizing datasets through transformer embeddings, extracting the last hidden state, and applying existing tools from the high-dimensional statistics literature, such as the 
- Maximum Mean Discrepancy [Gretton](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html)
- MAUVE (https://arxiv.org/abs/2102.01454)

<p align="center">
  <img src="[https://github.com/user-attachments/assets/e5226e73-484e-4db8-a66b-3ece9eaefc53](https://github.com/user-attachments/assets/289ca989-e20f-4ec7-a53a-badf3c54f6b1)" width="600">
</p>


Most of this README was written by an LLM tools, and most of the work is still actively in progress. I'll upload my preprint link, presentation slideshow, and several hundred commits by start of May.

## **Notable Files**

### Python Tools
- **`pythontools/mmd_tools.py`**
  - Contains methods for embedding documents with a large language model.
    - Word-to-document embedding is achieved through averaging across words.
  - Includes implementations for performing MMD (Maximum Mean Discrepancy) calculations.
  - Provides methods for various permutation tests, including tests under the null assumption.

- **`pythontools/mimic_tools.py`**
  - Defines the `MIMICEndpoint` class, which:
    - Downloads clinical notes.
    - Merges notes with admissions and patient data to estimate admission times.
    - Provides access to notes from a specific diagnosis using the `get_notes_diagnosis` method.

- **`pythontools/mimic_source.py`**
  - A class wrapper for certain methods in `mimic_tools`.
  - Allows for conceptualizing different "sources" of data, such as:
    - Notes from hospital admissions with a single diagnosis (`get_notes_K_diagnosis` method).
    - Notes from specific diagnosis codes like `I10` or `Z515` (`get_notes_diagnosis` method).
    - Notes from a mixture of diagnoses (`get_notes_mixture` method).

- **`pythontools/attack_tools.py`**
  - Implements various attack methodologies for analyzing model robustness.
  - Contains two main types of attacks:
    - **Filtering Attacks**: Remove specific notes based on criteria (e.g., `TabularUnivariateFilteringAttack`, `KeywordFilteringAttack`).
    - **Perturbation Attacks**: Modify note content (e.g., `ProbabilisticRemoveFirstSentenceAttack`, `ProbabilisticSwitchToKeywordAttack`).
  - Provides a framework for evaluating how different attacks affect model performance and distributional properties.

## **Experimental Methodologies**

### Attack Experiments

The attack experiments framework allows for systematic evaluation of model robustness against various data perturbations:

1. **Setup**: Define training and testing sources, model architecture, and attack methodology.
2. **Execution**: Apply attacks to training or testing data and measure performance degradation.
3. **Analysis**: Compare model behavior on attacked vs. unattacked data to identify vulnerabilities.

Key components:
- **Attack Types**: Filtering attacks remove specific notes, while perturbation attacks modify note content.
- **Metrics**: Evaluate both task-specific performance (e.g., classification accuracy) and distributional properties.
- **Interpretability**: Analyze which features or patterns are most affected by attacks.

### Source Degradation Prediction

This methodology predicts how model performance degrades when trained on one source and tested on another:

1. **Source Pair Selection**: Choose pairs of sources with different characteristics (e.g., different diagnoses, age groups).
2. **Performance Measurement**: Train models on one source and evaluate on both sources.
3. **Degradation Analysis**: Quantify performance differences between same-source and cross-source testing.

The `source_pairs_degradation_script.py` implements this methodology, generating results that help understand:
- Which source characteristics lead to the largest performance degradation
- How different model architectures handle distribution shifts
- Strategies for improving robustness to source variations

### Source Recovery Experiments

These experiments investigate whether we can identify the source of a document based on its embedding:

1. **Reference Dataset Creation**: Generate embeddings from known sources.
2. **Evaluation Dataset**: Create embeddings from a randomly selected source.
3. **Source Identification**: Use metrics like MMD (Maximum Mean Discrepancy) or MAUVE to identify which reference source the evaluation dataset most closely matches.

The `source_recovery.py` module implements this methodology, which helps:
- Understand the distinctiveness of different data sources
- Evaluate the effectiveness of embedding methods for source identification
- Develop techniques for detecting out-of-distribution data

### Distributional Comparison

This methodology identifies dimensions in sparse autoencoder representations that differ most between train and test distributions:

1. **Data Collection**: Obtain balanced train and test sets from different sources.
2. **Model Training**: Train a classifier and a sparse autoencoder on the train set.
3. **Sparse Encoding**: Generate sparse encodings for both train and test sets.
4. **Dimension Analysis**: Find dimensions with the largest distributional differences.
5. **Interpretation**: Identify notes with the largest activations for the most different dimensions.

The `distributional_comparison.py` script implements this approach, which helps:
- Identify systematic differences between data sources
- Interpret what specific dimensions in the sparse representation capture
- Understand how models might fail when deployed on different distributions

## **Structure**

- **`notebooks/`**: Contains Jupyter notebooks for analysis.
- **`pythonscripts/`**: Includes Python scripts for data preprocessing and analysis.
- **`bashfiles/`**: Bash scripts used for running Python scripts and organizing workflows.
- **`pythontools/`**: Core Python modules with reusable functionality.
- **`experimenttools/`**: Tools for running specific experimental methodologies.

## **To-Do**
- Improve documentation for easier navigation.
- Add detailed examples and workflows for new users.
- Expand the attack experiment framework with new attack types.
- Develop visualization tools for interpreting distributional differences.

Feel free to explore the repository and contribute!
