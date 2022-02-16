from setuptools import setup, find_packages

# pdoc -o ./html GSTag
setup(name='tagginggs',
      version='0.1.1',
      description='Easily Tag',
      url='',
      author='Saccaggi Giacomo',
      author_email='giacomo.saccaggi@gmail.com',
      license='Saccaggi Giacomo',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
            'Flask~=2.0.2', 'Pillow~=8.3.2', 'tensorflow~=2.6.0', 'scikit-learn~=1.0', 'matplotlib~=3.4.3', 'tqdm~=4.62.3', 'numpy~=1.19.5', 'flask_httpauth~=4.4.0','gunicorn~=20.1.0','spacy~=3.1.3'
      ],
      zip_safe=False
      )

