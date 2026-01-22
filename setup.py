from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("fp32_to_int_quantizer/version.py", "r", encoding="utf-8") as fh:
    version = fh.read().strip().split("=")[1].strip().strip('"').strip("'")

setup(
    name="fp32-to-int-quantizer",
    version=version,
    author="Author",
    author_email="author@example.com",
    description="A high-performance FP32-to-INT8/INT4 quantization toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fp32-to-int-quantizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.4",
        "scipy>=1.7",
    ],
    extras_require={
        "torch": ["torch>=1.9"],
        "tensorflow": ["tensorflow>=2.6"],
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
        ],
    },
    include_package_data=True,
    package_data={
        "fp32_to_int_quantizer": ["*.py"],
    },
)
