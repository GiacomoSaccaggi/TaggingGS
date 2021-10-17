
# TaggingGS

> Currently TaggingGS is an beta version

How to install TaggingGS from GIT:
- ***pip install --upgrade https://github.com/jkbr/httpie/tarball/master***

or
 
- ***pip install git+https://github.com/jkbr/httpie.git#egg=httpie***



How to install TaggingGS:
1. Unzip the archive
2. cd into the unzipped folder from the console
3. type in the console "pip install ." to install the package locally


> How to use tagginggs:
1. Import the package with the following code:

	<code>>>> from tagginggs import run_tagging_app</code>


	
2. Define Knowledge Base as pandas.DataFrame or import demo_kb:

	<code>>>> app = run_tagging_app()</code>



3. Model creation:

	<code>>>> app.run(port=8080)</code>





> Run app in container:

1. Install Docker app
2. Download zip docker and extract
3. Run on shell <code>docker_push_and_run.ps1</code>
