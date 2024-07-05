import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SSVEPAnalysisToolbox",
    version="0.0.3",
    author="Ze Wang",
    author_email="pikipityw@gmail.com",
    description="Python package for SSVEP datasets and algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pikipity/SSVEP-Analysis-Toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Required packages
    install_requires=[
        'numpy<=1.23,>1.0',
        'scipy<=1.13,>1.0',
        'py7zr<=0.21',
        'pooch<=1.8,>1.0',
        'joblib<=1.4,>1.0',
        'matplotlib<=3.8,>3.0',
        'tqdm<=4.66,>4.0',
        'scikit-learn<=1.3,>1.0',
        'mat73<=1.0'
    ],
    python_requires='>=3.9',
)
