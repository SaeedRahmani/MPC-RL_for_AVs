from setuptools import setup, find_packages

setup(
    name='mpcrl',  
    version='0.1.0', 
    description='A brief description of your package',
    author='Zhenlin (Gavin) Xu',
    author_email='gavinxu66@gmail.com',
    # url='https://github.com/yourusername/your-repo',
    packages=find_packages(),
    install_requires=[
        'gymnasium>=1.0.0a2',
        'numpy>=2.1.1',
        'highway-env==1.10.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',  
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
