from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dunhuang_dance_evaluation",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.9",
        "PyQt5>=5.15.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "fastdtw>=0.3.4",
        "matplotlib>=3.4.0"
    ],
    entry_points={
        "console_scripts": [
            "dunhuang_dance_evaluation=src.main:main",
            "dunhuang_dance_demo=demo:main",
        ],
    },
    author="Devin AI",
    author_email="devin-ai-integration[bot]@users.noreply.github.com",
    description="A system for evaluating Dunhuang flying dance movements in real-time",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dunhuang_dance_evaluation",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/dunhuang_dance_evaluation/issues",
        "Documentation": "https://github.com/yourusername/dunhuang_dance_evaluation/wiki",
        "Source Code": "https://github.com/yourusername/dunhuang_dance_evaluation",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="computer vision, dance, evaluation, real-time, dunhuang, motion analysis",
    python_requires=">=3.8",
    include_package_data=True,
)
