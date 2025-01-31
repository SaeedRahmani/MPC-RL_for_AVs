from setuptools import find_packages, setup

setup(
    name='mpcrl',  
    version='0.2.0', 
    description='A brief description of your package',
    author='Saeed Rahmani, Gavin (Zhenlin) Xu, Gozde Korpe',
    author_email='s.rahmani@tudelft.com',
    url='https://github.com/SaeedRahmani/MPC-RL_for_AVs',
    packages=find_packages(),
    install_requires=[
        'gymnasium==0.29.1',
        'numpy==2.1.2',
        'highway-env==1.9.1',
        'casadi==3.6.6',
        'hydra-core==1.3.2',
        'stable-baselines3==2.3.1',
        'shapely==2.0.6',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',  
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
