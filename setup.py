from setuptools import setup, find_packages

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
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow'
    ],
)