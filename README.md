# stimulus-py

[![ci](https://github.com/mathysgrapotte/stimulus-py/workflows/ci/badge.svg)](https://github.com/mathysgrapotte/stimulus-py/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://mathysgrapotte.github.io/stimulus-py/)
[![Build with us on slack!](http://img.shields.io/badge/slack-nf--core%20%23deepmodeloptim-4A154B?labelColor=000000&logo=slack)](https://nfcore.slack.com/channels/deepmodeloptim)

<!-- [![pypi version](https://img.shields.io/pypi/v/stimulus-py.svg)](https://pypi.org/project/stimulus-py/) -->

## ⚠️ Development Warning

> **Warning**  
> This package is in active development and breaking changes may occur. The API is not yet stable and features might be added, modified, or removed without notice. Use in production environments is not recommended at this stage.

We encourage you to:

- 📝 Report bugs and issues on our [GitHub Issues](https://github.com/mathysgrapotte/stimulus-py/issues) page

- 💡 Suggest features and improvements through [GitHub Discussions](https://github.com/mathysgrapotte/stimulus-py/discussions)

- 🤝 Contribute by submitting pull requests

We are actively working towards release 1.0.0 (see [milestone](https://github.com/mathysgrapotte/stimulus-py/milestone/1)), check the slack channel by clicking on the badge above where we are actively discussing. Build with us every wednesday at 14:00 CET until 18:00 CET on the nf-core gathertown (see slack for calendar updates i.e. some weeks open dev hours are not possible)




## Introduction

Most (if not all) quality software is thouroughly tested. Deep neural networks seem to have escaped this paradigm. 

In the age of large-scale deep learning, it is critical that early-stage dl models (prototypes) are tested to ensure costly bugs do not happen at scale.

Here, we attempt at solving the testing problem by proposing an extensive library to test deep neural networks beyond test-set performance. 

Stimulus provides those functionalities

1. **Data Perturbation Testing**:  
   Modify training data to test model's robustness to perturbations and uncover which pre-processing steps increase performance

2. **Hyperparameter Optimization**:  
   Perform tuning on model architecture with user-defined search spaces using Ray[tune] to ensure comparable performance across data transformations

3. **Comprehensive Analysis**:  
   Generate all-against-all model report to guide data pre-processing decisions

Stimulus aims at providing those functionalities in a near future

4. **Model Architecture Testing**:  
   Run routine checks on model architecture and training process including type-checking, model execution, and weight updates

5. **Post-Training Validation**:  
   Perform comprehensive model validation including overfitting detection and out-of-distribution performance testing

6. **Informed Hyperparameter Tuning**:  
   Encourage tuning strategies that follow [Google's Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) [^1]

7. **Scaling Analysis**:  
   Generate scaling law reports to understand prototype model behavior at different scales

For large scale experiments, we recommend our [nf-core](https://nf-co.re) [deepmodeloptim](https://github.com/nf-core/deepmodeloptim) pipeline which is still under development and will be released alongside stimulus v1.0.0.

### Repository Organization  

```
src/stimulus/ 🧪
├── analysis/ 📊
│   └── analysis_default.py
├── cli/ 🖥️
│   ├── analysis_default.py
│   ├── check_model.py
│   ├── interpret_json.py
│   ├── predict.py
│   ├── shuffle_csv.py
│   ├── split_csv.py
│   ├── split_yaml.py
│   ├── transform_csv.py
│   └── tuning.py
├── data/ 📁
│   ├── csv.py
│   ├── experiments.py
│   ├── handlertorch.py
│   ├── encoding/ 🔐
│   │   └── encoders.py
│   ├── splitters/ ✂️
│   │   └── splitters.py
│   └── transform/ 🔄
│       └── data_transformation_generators.py
├── learner/ 🧠
│   ├── predict.py
│   ├── raytune_learner.py
│   └── raytune_parser.py
└── utils/ 🛠️
    ├── json_schema.py
    ├── launch_utils.py
    ├── performance.py
    └── yaml_model_schema.py
```

## Installation

stimulus is still under development, you can install it from test-pypi by running the following command:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple stimulus-py==0.0.10
```


### citations

[^1]: Godbole, V., Dahl, G. E., Gilmer, J., Shallue, C. J., & Nado, Z. (2023). Deep Learning Tuning Playbook (Version 1.0) [Computer software]. http://github.com/google-research/tuning_playbook