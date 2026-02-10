from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# Requirements for installing the package
REQUIREMENTS = [
    'numpy==2.4.2',
    'tensorflow==2.20.0',
    'pandas==3.0.0',
    'scikit-learn==1.8.0',
    'sympy==1.14.0',
    'matplotlib==3.10.8',
    'seaborn==0.13.2',
    'graphviz==0.21',
    'dill==0.4.1',
]

# Some details 
CLASSIFIERS = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
]

setup(
    name='dexire',
    version='0.1',
    description='Deep Explanation and Rule Extractor (DEXiRE)\
        is a rule extractor explainer to explain Deep learning modules\
        through rule sets.',
    author='Victor Hugo Contreras and Davide Calvaresi',
    author_email='victorc365@gmail.com',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=REQUIREMENTS,
    classifiers=CLASSIFIERS,
    python_requires='>=3.11',
)