# mol-inspector - Inspect Molecular ML Models Using SHAP
[![mol-inspector](https://img.shields.io/pypi/v/mol-inspector.svg?label=mol-inspector&color=brightgreen)](https://pypi.org/project/mol-inspector/)
[![Python versions](https://img.shields.io/pypi/pyversions/mol-inspector?style=flat&logo=python&logoColor=white)](https://pypi.org/project/mol-inspector/)[![PyPI - Python Version](https://img.shields.io/pypi/v/mol-inspector.svg)](https://pypi.org/project/mol-inspector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project is built on [SHAP](https://shap.readthedocs.io/en/latest/index.html) - a game theoretic approach to 
explain the output of machine learning models with figures be generated using [Seaborn](https://seaborn.pydata.org). 
While this project should be able to work with any data generated from SHAP values, it was built with molecular 
fingerprints in mind. As a result, simple methods are also included that will render molecular fingerprint bits using 
RDKit.

## Tutorials
Tutorials can be found under the [Tutorials folder](/tutorials). The aim of mol-inspector is for ease of use. Generating
SHAP values and plots can be done as follows:
```python
from mol_inspector import Inspector, Plots

# generating SHAP values
inspector = Inspector(model=model, train_feats=train_feats, model_type="auto")
values = inspector.values(test_feats)

# generating plots
plot = Plots(values, train_feats, test_feats)
plot.summary_plot()
```
Users can also import and generate shap values natively, then feed the shap values into the Plots() class to generate 
the Seaborn plots. 

## Installation
Project can be pip installable:
```
pip install mol-inspector
```
> [!NOTE]
> As my background is in cheminformatics, RDKit will also be installed. This allows for methods to draw molecular 
> fingerprint bits. I may "split" this code upon user request.

## Motivations
When experimenting with the SHAP package, I found difficulties in generating customized color palettes for my figures. 
This project seeks to remedy that by extracting the generated SHAP values and using the data to generate figures using
Seaborn. This would allow easier customization of color palettes for presentations. 

## Future Contributions
Currently only three plot types have been converted. This is due to how I implement SHAP into my projects. I welcome 
contributors or requests for additional plot options to be converted into seaborn. Just let me know!