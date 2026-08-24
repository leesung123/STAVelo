from setuptools import setup, find_packages

project_urls = {
  'STAVelo': 'https://github.com/zhanglabtools/STAVelo'
}

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="STAVelo",
    version="0.0.1",
    packages=find_packages(),
    author="Shang Li",
    author_email="lishang@amss.ac.cn",
    description="Spatial RNA velocity inference based on graph attention encoder-decoder network",
    long_description=long_description,
    long_description_content_type='text/markdown',
    project_urls = project_urls,
    python_requires=">=3.9, <3.10",
    install_requires=[
        'pandas==1.5.3',
        'numpy==1.23.0',
        'anndata==0.10.9',
        'tqdm==4.70.0',
        'scikit-learn==1.6.1',
        'scipy==1.11.4',
        'joblib==1.5.3',
        'statsmodels==0.14.6',
        'matplotlib==3.6.0',
        'seaborn==0.13.2',
        'umap-learn==0.5.12',
        'scvelo==0.2.5',
        'igraph==0.10.4',
        'scanpy==1.8.2',
        'loompy==3.0.8',
        'numba==0.60.0',
        'pynndescent==0.6.0',
    ]
)

