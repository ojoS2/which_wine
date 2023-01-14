from setuptools import setup, find_packages
setup(  author='Ricardo S. P. Lopes',
        author_email='ojogodascontasdevidro@gmail.com',
        description='''A package to analyze wine descriptions
         to find out which wine veriety it is''',
        name='which_wine',
        version='0.1.0',
        packages=find_packages(include=['which_wine', 'which_wine.*']),
        install_requires=['pandas>=1.5','numpy>=1.23','matplotlib>=3.6',
        'seaborn>=0.12.1', 'sklearn', 'spacy'],
        python_requires='>=3.8',
        license='MIT',
        setup_requires=['pytest-runner'],
        tests_require=['pytest==4.4.1'],
        test_suite='tests',
    )