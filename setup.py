from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name='dexire',
    version='0.0.1',
    description='Deep Explanation and Rule Extractor (DEXiRE)\
        is a rule extractor explainer to explain Deep learning modules\
        through rule sets.',
    author='Victor Hugo Contreras and Davide Calvaresi',
    author_email='victorc365@gmail.com',
    packages=find_packages(),
    install_dependencies=[
'numpy==1.26.4',
'tensorflow==2.15.0',
'pandas==2.2.1',
'scikit-lear==1.4.1.post1',
'sympy==1.12',
'pytest==8.1.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)