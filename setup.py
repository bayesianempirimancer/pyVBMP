from setuptools import setup, find_packages

setup(
    name='pyVBMP',
    version='0.1',
    dexcription='Variational Bayesian Message Passing Modules for pyTorch',
    author='Jeff Beck',
    author_email='bayesian.empirimancer@gmail.com',
    url='None',
    packages=find_packages(),
#    packages=['pyVBMP'],
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'matplotlib',
        # add any other dependencies here
    ],
)

