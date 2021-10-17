from setuptools import setup, find_packages

# pdoc -o ./html GSTag
setup(name='tagginggs',
      version='0.1.0',
      description='Easily Tag',
      url='',
      author='Saccaggi Giacomo',
      author_email='giacomo.saccaggi@gmail.com',
      license='Saccaggi Giacomo',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
            'Flask', 'Pillow', 'tensorflow', 'scikit-learn', 'matplotlib', 'tqdm', 'numpy'
      ],
      zip_safe=False
      )
