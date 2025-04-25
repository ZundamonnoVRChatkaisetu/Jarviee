from setuptools import setup, find_packages

setup(
    name="jarviee",
    version="0.1.0",
    description="自律的知識探求型AIアシスタント",
    author="Jarviee開発チーム",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt")
        if not line.startswith("#") and line.strip()
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "jarviee=src.interfaces.cli.jarviee_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: Japanese",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
