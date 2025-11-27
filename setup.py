from setuptools import setup, find_packages

# This setup file makes the module installable with:
#   pip install .
# or
#   pip install git+https://github.com/Javier-fl/CyAnno_module

setup(
    name="cyanno_pipeline",
    version="0.1.0",
    description="Simplified CyAnno classifier for cytometry cell annotation",
    packages=find_packages(),  # automatically finds cyanno_pipeline/
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "fcsparser",
    ],
)
