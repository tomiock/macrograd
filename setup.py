import setuptools
import os

def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Macrograd: A minimal deep learning framework." # Fallback description

REQUIRED_PKGS = [
    "numpy",
]

EXTRAS_REQUIRED_PKGS = {
    "cuda": ["cupy-cuda11x"],
    "viz": ["graphviz"],
}

EXTRAS_REQUIRED_PKGS["all"] = list(set(sum(EXTRAS_REQUIRED_PKGS.values(), [])))

setuptools.setup(
    name="macrograd",
    version="0.1.0-dev",
    author="tomiock",
    author_email="ockier@gmail.com",
    description="Deep learing framework from scratch",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/tomiock/macrograd",
    packages=setuptools.find_packages(
        where=".", include=("macrograd*",)
    ),  # Adjust include pattern if needed
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRED_PKGS,
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
