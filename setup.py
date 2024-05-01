from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# Requirements for installing the package
REQUIREMENTS = [
'numpy==1.26.4',
'tensorflow==2.15.0',
'pandas==2.2.1',
'scikit-learn==1.4.1.post1',
'sympy==1.12',
'pytest==8.1.1',
'matplotlib==3.8.3',
'seaborn==0.13.2',
'graphviz==0.20.3'
]

# Some details 
CLASSIFIERS = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
]

setup(
    name='dexire',
    version='0.0.1',
    description='Deep Explanation and Rule Extractor (DEXiRE)\
        is a rule extractor explainer to explain Deep learning modules\
        through rule sets.',
    author='Victor Hugo Contreras and Davide Calvaresi',
    author_email='victorc365@gmail.com',
    packages=find_packages(),
    install_dependencies=REQUIREMENTS,
    classifiers=CLASSIFIERS,
    python_requires='>=3.9',
)