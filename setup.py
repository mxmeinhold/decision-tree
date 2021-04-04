import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="decision-tree",
    version="0.1.0",
    author="Max Meinhold",
    author_email="mxmeinhold@gmail.com",
    description="A statically typed python decision tree module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/mxmeinhold/decision-tree",
    packages=["decision_tree"],
    python_requires=">=3.10",
    project_urls={
        "Bug Tracker": "https://github.com/mxmeinhold/decision-tree/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    keywords='decision-tree machine-learning',
)
