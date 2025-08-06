from setuptools import find_packages, setup

with open("VERSION") as f:
    version = f.read().strip()

with open("onnxslim/version.py", "w") as f:
    f.write(f'__version__ = "{version}"\n')

setup(
    name="onnxslim",
    version=version,
    description="OnnxSlim: A Toolkit to Help Optimize Onnx Model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/inisis/OnnxSlim",
    author="inisis",
    author_email="desmond.yao@buaa.edu.cn",
    project_urls={
        "Bug Tracker": "https://github.com/inisis/OnnxSlim/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    install_requires=["onnx", "sympy>=1.13.3", "packaging", "colorama"],
    packages=find_packages(exclude=("tests", "tests.*")),
    entry_points={"console_scripts": ["onnxslim=onnxslim.cli:main"]},
    zip_safe=True,
    python_requires=">=3.6",
)
