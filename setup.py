from setuptools import find_packages, setup

setup(
    name='mpcrl',  
    version='0.1.0', 
    description='A brief description of your package',
    author='Zhenlin (Gavin) Xu',
    author_email='gavinxu66@gmail.com',
    url='https://github.com/Zhenlin-Xu/mpcrl',
    packages=find_packages(),
    install_requires=[
        'gymnasium>=1.0.0',
        'numpy>=2.1.2',
        'highway-env==1.10.1',
        'hydra-core>=1.3.2',
        'stable-baselines3==2.4'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',  
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
