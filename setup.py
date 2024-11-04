from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-orchestrator",  # Changed to avoid conflicts
    version="0.1.0",
    author="Intuition Labs LLC",  # Match your LICENSE.md
    author_email="admin@terminals.tech",
    description="Intelligent AI task orchestration for applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wheattoast11/orchestrator",
    packages=find_packages(exclude=["tests*", "examples*"]),  # Exclude test/example files
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",  # Added relevant topic
    ],
    python_requires=">=3.8",
    install_requires=[
    "aiohttp>=3.8.0",
    "pydantic>=2.0.0",
    "python-dateutil>=2.8.2",
    "typing-extensions>=4.0.0",
    "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.19.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=0.990",
        ],
    },
)
