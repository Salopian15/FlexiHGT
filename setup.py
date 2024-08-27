from setuptools import setup, find_packages

setup(
    name="FlexiHGT",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "biopython",
        "ete3",
    ],
    entry_points={
        "console_scripts": [
            "flexihgt=flexihgt.cli:main",
        ],
    },
    author="Jack A. Crosby",
    author_email="jac180@aber.ac.uk",
    description= (
        "A rapid HGT detection tool based off HGTPhyloDetect, uses diamond and ete3 offline databases and custom taxlevels to detect HGT events"
        ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://https://github.com/Salopian15/FlexiHGT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.10",
    scripts =[
        'scripts/install_dependencies.sh',
    ]
)
