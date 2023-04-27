from setuptools import setup, find_packages

setup(
    name='airunner-lora',
    version="1.0.0",
    author="Capsize LLC",
    description="LoRA extension for AI Runner",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="ai, stable diffusion, artificial intelligence, art, ai art, application",
    license="AGPL-3.0",
    author_email="contact@capsize.gg",
    url="https://github.com/Capsize-Games/airunner-lora",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10.0",
    install_requires=[
        "airunner>=1.9.0",  # airunnerwindows >= 1.9.0
    ]
)
