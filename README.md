
# TaggingGS

> Currently TaggingGS is an beta version

How to install TaggingGS from ZIP file:
- ***pip install https://github.com/GiacomoSaccaggi/TaggingGS/archive/refs/heads/main.zip***

How to install TaggingGS from GIT:
 
- ***pip install git+https://github.com/GiacomoSaccaggi/TaggingGS.git#egg=TaggingGS***



How to install TaggingGS with clone folder:
1. Unzip the archive
2. cd into the unzipped folder from the console
3. type in the console "pip install ." to install the package locally


> How to use tagginggs:
1. Import the package with the following code:

	<code>>>> from tagginggs import run_tagging_app</code>


	
2. Create flask app:

	<code>>>> app = run_tagging_app()</code>



3. Run flask app:

	<code>>>> app.run(port=8080)</code>

