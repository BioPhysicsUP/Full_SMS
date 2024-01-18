import os

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

this = os.path.dirname(os.path.realpath(__file__))

def read(name):
    with open(os.path.join(this, name)) as f:
        return f.read()
setup(
    name='Full_SMS',
    version='0.6.2',
    description='An analysis suite developed by the Biophysics Group at the University of Pretoria, South Africa',
    long_description=readme,
    author='Joshua Botha, Bertus van Heerden',
    author_email='jbotha1951@gmail.com, bertus.mooikloof@gmail.com',
    url='https://github.com/BioPhysicsUP/Full_SMS',
    packages=['src'],
    install_requires=read('requirements.txt'),
    include_package_data=True,
    zip_safe=True,
    licence='GPLv3',
    keywords='single-molecule spectroscopy fluorescence change-point analysis',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English'
    ],
    entry_points={
            'console_scripts': [
                'main_app=src.main:main']
                },
    #scripts=['bin_old/main_app_old']
)