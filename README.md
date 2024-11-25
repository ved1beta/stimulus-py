# STIMULUS 
## Stochastic Testing with Input Modification for Unbiased Learning Systems.

[![ci](https://github.com/mathysgrapotte/stimulus-py/workflows/ci/badge.svg)](https://github.com/mathysgrapotte/stimulus-py/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://mathysgrapotte.github.io/stimulus-py/)
[![Build with us on slack!](http://img.shields.io/badge/slack-nf--core%20%23deepmodeloptim-4A154B?labelColor=000000&logo=slack)](https://nfcore.slack.com/channels/deepmodeloptim)

<!-- [![pypi version](https://img.shields.io/pypi/v/stimulus-py.svg)](https://pypi.org/project/stimulus-py/) -->

> WARNING:
> This package is in active development and breaking changes may occur. The API is not yet stable and features might be added, modified, or removed without notice. Use in production environments is not recommended at this stage.
>
> We encourage you to:
>
> - 📝 Report bugs and issues on our [GitHub Issues](https://github.com/mathysgrapotte/stimulus-py/issues) page
>
> - 💡 Suggest features and improvements through [GitHub Discussions](https://github.com/mathysgrapotte/stimulus-py/discussions)
>
> - 🤝 Contribute by submitting pull requests
>
> We are actively working towards release 1.0.0 (see [milestone](https://github.com/mathysgrapotte/stimulus-py/milestone/1)), check the slack channel by clicking on the badge above where we are actively discussing. Build with us every wednesday at 14:00 CET until 18:00 CET on the nf-core gathertown (see slack for calendar updates i.e. some weeks open dev hours are not possible)



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

For large scale experiments, we recommend our [nf-core](https://nf-co.re) [deepmodeloptim](https://github.com/nf-core/deepmodeloptim) pipeline which is still under development and will be released alongside stimulus v1.0.0.

📹 Stimulus was featured at the nextflow summit 2024 in Barcelona, which is a nice intoduction to current package capabilities, you can watch the talk [here](https://www.youtube.com/watch?v=dC5p_tXQpEs)



Stimulus aims at providing those functionalities in a near future, stay tuned for updates!

4. **Model Architecture Testing**:  
   Run routine checks on model architecture and training process including type-checking, model execution, and weight updates

5. **Post-Training Validation**:  
   Perform comprehensive model validation including overfitting detection and out-of-distribution performance testing

6. **Informed Hyperparameter Tuning**:  
   Encourage tuning strategies that follow [Google's Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) [^1]

7. **Scaling Analysis**:  
   Generate scaling law reports to understand prototype model behavior at different scales


## User guide

### Repository organization  

Stimulus is organized as follows, we will reference to this structure in the following sections

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


### Data encoding

Data in stimulus can take many forms (files, text, images, networks...) in order to support this diversity, stimulus relies on the [encoding module](https://mathysgrapotte.github.io/stimulus-py/reference/stimulus/data/encoding/encoders/#stimulus.data.encoding.encoders.AbstractEncoder). List of available encoders can be found [here](https://mathysgrapotte.github.io/stimulus-py/reference/stimulus/data/encoding/encoders/#stimulus.data.encoding.encoders).

If the provided encoders do not support the type of data you are working with, you can write your own encoder by inheriting from the `AbstractEncoder` class and implementing the `encode`, `decode` and `encode_all` methods. 

- `encode` is currently optional, can return a `NotImplementedError` if the encoder does not support encoding a single data point
- `decode` is currently optional, can return a `NotImplementedError` if the encoder does not support decoding
- `encode_all` is called by other stimulus functions, and is expected to return a [`np.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html) . 

### Expected data format

Data is expected to be presented in a csv samplesheet file with the following format: 

| input1:input:input_type | input2:input:input_type | meta1:meta:meta_type | label1\:label:label_type | label2\:label:label_type |
| ----------------------- | ----------------------- | -------------------- | ----------------------- | ----------------------- |
| sample1 input1          | sample1 input2          | sample1 meta1        | sample1 label1          | sample1 label2          |
| sample2 input1          | sample2 input2          | sample2 meta1        | sample2 label1          | sample2 label2          |
| sample3 input1          | sample3 input2          | sample3 meta1        | sample3 label1          | sample3 label2          |

Columns are expected to follow this name convention : `name:type:data_type`

- name corresponds to the column name, this should be the same as input names in model batch definition (see model section for more details)

- type is either input, meta or label, typically models predict the labels from the input, and meta is used to perform downstream analysis

- data_type is the column data type.

> NOTE:
> This rigid data format is expected to change once we move to release v1.0.0, data types and information will be defined in a yaml config and only column names will be required in the data, see [this github issue](https://github.com/mathysgrapotte/stimulus-py/issues/24)

### Connecting encoders and datasets

Once we have our data formated and our encoders ready, we need to explicitly state which encoder is used for which data type. This is done through an experiment class. 

To understand how experiment classes are used to connect data types and encoders, let's have a look at a minimal DnaToFloat example : 

```python
class DnaToFloat(AbstractExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.dna = {
            "encoder": encoders.TextOneHotEncoder(alphabet="acgt"),
        }
        self.float = {
            "encoder": encoders.FloatEncoder(),
        }
```

Here we define the `data_type` for the dna and float types, note that those `data_type` are the same as the ones defined in the samplesheet dataset above, for example, a dataset on which this experiment would run could look like this: 

| mouse_dna:input:dna | mouse_rnaseq\:label:float |
| ------------------- | ------------------------ |
| ACTAGGCATGCTAGTCG   | 0.53                     |
| ACTGGGGCTAGTCGAA    | 0.23                     |
| GATGTTCTGATGCT      | 0.98                     |

Note how the `data_type` for the mouse_dna and mouse_rnaseq columns match exactly the attribute names defined in the `DnaToFloat` minimal class above. 

stimulus-py ships with a few basic experiment classes, if you need to write your own experiment class, simply inherit from the base `AbstractExperiment` class and overwrite the class `__init__` method like shown above.

> NOTE:
> This has the drawback of requiring a build of the experiment class each time a new task is defined (for instance, let's say we want to use dna and protein sequences to predict rna).
>
> Once we move to release v1.0.0, `type` (i.e. input, meta, label) and `data_type` will be defined in the data yaml config, and the relevant experiment class will be automatically built.


### Loading the data

Finally, once we have defined our encoders, the experiment class and the samplesheet, stimulus will transparently load the data using the [csv.py module](https://mathysgrapotte.github.io/stimulus-py/reference/stimulus/data/csv/#stimulus.data.csv)

csv.py contains two important classes, `CsvLoader` and `CsvProcessing`

`CsvLoader` is responsible for naïvely loading the data (without changing anything), it works by performing a couple of checks on the dataset to ensure it is correctly formated, and then uses the experiment class in conjunction with the column names to call the proper encoders and output inputs, labels, and meta dictionary objects. 

`CsvLoader` is used by the `handlertorch` module to load data into pytorch tensors. 

> TIP:
> So, to recap,
> when you load a dataset into a torch tensor, 
>
> 1. `handlertorch` will call `CsvLoader` with the csv samplesheet and the experiment class
>
> 2. `CsvLoader` will use the experiment class to fetch the proper encoder `encode_all` method for each data column
>
> 3. `CsvLoader` will use the `encode_all` method to encode the data and output dictionary objects for inputs, labels and meta data
>
> 4. `handlertorch` will convert the contents to torch tensors
>
> 5.  `handlertorch` will feed the `input` torch tensor to the model, use the `label` torch tensor for loss computation and will store the `meta` tensor for downstream analysis
>
>Great, now you know how stimulus transparently loads your data into your pytorch model! While this seems complicated, the only thing you really have to do, is to format your data correctly in a csv samplesheet and define your experiment class with the proper encoders (either by using the provided encoders or by writing your own).

### Data transformation

Measuring the impact of data transformations (noising, down/upsampling, augmentation...) on models at training time is a major feature of stimulus.

Data transformations materialize as `DataTransformer` classes, and should inherit from the `AbstractDataTransformer` class (see [docs](https://mathysgrapotte.github.io/stimulus-py/reference/stimulus/data/encoding/encoders/#stimulus.data.encoding.encoders.AbstractEncoder))

> NOTE:
> Writing your own `DataTransformer` class is the same as writing your own `Encoder` class, you should overwrite the `transform` and `transform_all` methods

> WARNING:
> Every `DataTransformer` class has to have `seed` in `transform` and `transform_all` methods parameters, and `np.random.seed(seed)` should be called in those methods.

> WARNING:
> Every `DataTransformer` class should have an `add_row` argument set to either `True` or `False` depending on if it is augmenting the data (adding rows) or not.

### Connecting transformations and dataset

Just like encoders, data transformations are defined in the `Experiment` class alongside encoders. Let's upgrade our `DnaToFloat` minimal class defined above to reflect this.

```python
class DnaToFloat(AbstractExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.dna = {
            "encoder": encoders.TextOneHotEncoder(alphabet="acgt"),
            "data_transformation_generators": {
                "UniformTextMasker": data_transformation_generators.UniformTextMasker(mask="N"),
                "ReverseComplement": data_transformation_generators.ReverseComplement(),
                "GaussianChunk": data_transformation_generators.GaussianChunk(),
            },
        }
        self.float = {
            "encoder": encoders.FloatEncoder(),
            "data_transformation_generators": {"GaussianNoise": data_transformation_generators.GaussianNoise()},
        }
```

As you can see, our `data_type` arguments get an other field, `"data_transformation_generators"`, there we can initialize the `DataTransformer` classes with their relevant parameters. 

In the `csv` module, the `CsvProcessing` class will call the `transform_all` methods from the classes contained in `"data_transformation_generators"` based on the column type and a list of transformations. 

i.e., if we give the `["ReverseComplement","GaussianChunk"]` list to the `CsvProcessing` class `transform` method the data contained in the `mouse_dna:input:dna` column in our minimal example above will be first reverse complemented and then chunked. 

> TIP:
> Recap :
> To transform your dataset,
>
> - define your own `DataTransformer` class or use one we provide
>
> - add it to your experiment class
>
> - load your data through `CsvProcessing` 
>
> - set a list of transforms
>
> - call `CsvProcessing.transform(transform_list)`




## Installation

stimulus is still under development, you can install it from test-pypi by running the following command:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple stimulus-py==0.0.10
```


## citations

[^1]: Godbole, V., Dahl, G. E., Gilmer, J., Shallue, C. J., & Nado, Z. (2023). Deep Learning Tuning Playbook (Version 1.0) [Computer software]. http://github.com/google-research/tuning_playbook