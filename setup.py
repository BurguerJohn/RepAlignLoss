from setuptools import setup, find_packages
import os

def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="repalignloss",
    version="0.1.0",
    description="RepAlignLoss: a PyTorch loss aligning internal representations",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Gabriel Poetsch",
    author_email="griskai.yt@gmail.com",
    url="https://github.com/BurguerJohn/RepAlignLoss",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Since you only have two modules, you can list them as py_modules
    py_modules=["RepAlignLoss", "TTAdamW"],
    install_requires=[
        "torch", 
    ],
    python_requires=">=3.6",
)
