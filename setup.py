# This setup file is based on the one found here:
# https://towardsdatascience.com/create-your-custom-python-package-that-you-can-pip-install-from-your-git-repository-f90465867893

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='BAD',
    version='0.1.0',
    author='I.N.Tzortzis',
    author_email='i.n.tzortzis@gmail.com',
    description='BAD_tool installation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/itzortzis/breast_area_detection.git',
    project_urls = {
        "Code": "https://github.com/itzortzis/breast_area_detection.git",
        "Bug Tracker": "https://github.com/itzortzis/breast_area_detection/issues"
    },
    license='GPL-3.0',
    packages=['bad'],
    install_requires=['numpy', 'torch', 'tqdm', 'matplotlib', 'csv', 'sklearn.cluster', 'skimage.color', 'calendar', 'torchmetrics'],
)