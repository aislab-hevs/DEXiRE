# DEXIRE
<!-- trunk-ignore(markdownlint/MD033) -->
<p align="center">
<img src="images/logo/logo_dexire_small.png" alt="logo_dexire" width="100"/>
</p>

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and license info here --->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DEXiRE stands Deep Explanations and Rule extractor is an XAI tool to explain deep learning models extracting rule sets from their hidden layers.

This project allows to explain supervised learning models (e.g., classification and regression).

## Prerequisites

Before you begin, ensure you have met the following requirements:

<!--- These are just example requirements. Add, duplicate or remove as required --->

- You have a `<Windows/Linux/Mac>` machine. State which OS is supported/which is not.
- You have installed `python 3.11` or `python 3.12`.
- It is recommended to create an environment with conda or venv to isolate the execution and avoid version conflict.

## Installing DEXiRE

To install DEXiRE, follow these steps:

Windows, Linux and macOS:

```
python -m pip install --upgrade setuptools wheel twine
```

In the root directory DEXIRE execute the following command with the active environment  activated:

```
pip install .
```

Or in the main folder with the environment activated  execute the following command in the terminal:

```
python setup.py install
```

## Installing with wheels

The package can be compile to a wheel fire and the easy installed. To build a wheel execute the following command in the terminal and localized in the DEXIRE main folder:

For Unix/Linux/macOS build:

```
python3 -m pip install --upgrade build
python3 -m build
```

For Windows:

```
py -m pip install --upgrade build
py -m build
```

The wheel installer will be appear in the dist subdirectory. Localize in the dist subdirectory execute the following command:

```
pip install dexire-0.1-py3-none-any.whl
```

The wheel installer (.whl file) cna be distributed to install in other environments.

## Using DEXiRE

Once DEXIRE have been successfully installed can be used following the next steps:

1. Train a tensorflow model using the functional or sequential API.
2. Create and configure the DEXiRE object in a python notebook or script:
   
    ```python
    dexire = DEXiRE(model=model)
    ```

3. Execute the rule extraction process, with the following method:
   
    ```python
        rule_set = dexire.extract_rules(X_train, y_train)
    ```

4. Visualize and use rules to predict:
    ```python
    y_pred = rule_set.predict(X_test)
    ```


## Contributing to DEXiRE

<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->

To contribute to DEXiRE, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## project manager

Thanks to the project manager [Davide Calvaresi](mailto:davide.calvaresi@hevs.ch), who support the development of this project.

## Contributors

Thanks to the following people who have contributed to this project:

- [@victorc365](https://github.com/victorc365) 📖
- [@ilmaro8](https://github.com/ilmaro8)
- [@lorafanda](https://github.com/lorafanda)

## Acknowledge  

To acknowledge the contributions of DEXiRE or cite the original paper use the following bibtex:

```
@article{contreras2022dexire,
  title={A dexire for extracting propositional rules from neural networks via binarization},
  author={Contreras, Victor and Marini, Niccolo and Fanda, Lora and Manzo, Gaetano and Mualla, Yazan and Calbimonte, Jean-Paul and Schumacher, Michael and Calvaresi, Davide},
  journal={Electronics},
  volume={11},
  number={24},
  pages={4171},
  year={2022},
  publisher={MDPI}
}
```

This work is supported by the Chist-Era grant CHIST-ERA19-XAI-005, and by the Swiss National Science Foundation (G.A. 20CH21\_195530).

## Contact

If you want to contact me you can reach me at <victorc365@gmail.com>.

## License

<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [MIT](https://opensource.org/license/mit).
