# Overview
Keeping abreast of current trends, technologies, and best practices in statistics, visualization, and data analysis is becoming
increasingly difficult even for professional data scientists, and is a hopeless endeavor for domain experts lacking time and training.

We propose Lodestar, an interactive visualization sandbox that allows users to perform visual data analysis simply by
selecting from a list of recommendations. Choosing a recommendation adds the corresponding Python code to the notebook and
executes it, thus generating new output. The recommendation engine is inspired by autocomplete mechanisms, where a partial query is
used to show suggestions for how to finish it. 

In our implementation, we derive our recommendations from a directed graph of analysis
states: one manually curated from online data science tutorials, another by automatically analyzing the Python code from a corpus of
approximately 6,000 Jupyter notebooks on data science. We evaluated Lodestar through a two-phase evaluation sequence: a formative
study guiding our next set of improvements to the tool, followed by a summative study assessing its utility for novice data scientists.

# Structure
Lodestar is developed as a web-application. The front-end is written in HTML, CSS, and JavaScript and developed using Polymer, a JavaScript library.  The back-end is written in Python and developed using Flask, a web-framework for Python. 

The Pandas data analysis library is heavily used, especially for the Dataframe data structure. 


## ```src/``` 
The ```src/``` folder contains the front-end of the application. ```src/main-app/``` contains three files that control the front-end display and logic. 

```main-app.html``` contains the MainApp class and handles the HTML templating, CSS styling, and JavaScript functions throughout the front-end of the application.

```analysis-tab``` contains the AnalysisTab class and handles HTML templating, CSS styling, and JavaScript functions for each notebook cell. 

```suggest-panel``` contains the SuggestPanel class and handles HTML templating, CSS styling, and JavaScript functions related to the recommendation panel.

## ```server/``` 
The ```server/``` folder contains files for the back-end of the application. ```server/src/``` contains three files that control much of the application's execution. 

```run.py``` establishes the routes of the program and executes the Flask application. 

```analysis.py``` contains the Analysis class, which primarily handles executing the selected analysis methods using codeblocks, and also handles deleting and exporting dataframes. 

```recommender.py``` contains the Recommender class, which determines the expert and crowd recommendations given the current place in the data analysis process. 


# Lodestar Setup Instructions


## **Step 1**

Clone the repo:

`git clone https://gitlab.cs.umd.edu/leibatt/lodestar.git`

Before continuing, we have to download Node & npm: 

[Download](https://www.npmjs.com/get-npm)


**Note**: NPM version should be 10 

**Note**: In the future, we will want to use a Node Version Manager (also in link above)

We also want to install **Python 2.7**

**Mac OS**: Python 2.7 comes preinstalled and is the default version for Mac OS (if you have an older Mac machine you may want to check to see you have this installed). To check which version of python you have enter ```python``` into the terminal

**Windows**: [Python Download - choose 2.7 release!](https://www.python.org/downloads/)


## **Step 2**

Lodestar uses [PolymerJS](https://www.polymer-project.org/) for the client side. We can install Polymer-CLI by running the following command in the root directory of the repo: 
`npm install -g polymer-cli`

**Note**: 
The **g** in `npm install -g` is a flag signifying that you want to install that particular npm module system wide (globally). Without the g option, the module would be installed locally inside the current directory called `node_modules`

Lodestar uses many different packages for all of its client side components. To manage these, we use **Bower**, a ***package manager***

To install Bower: `npm install -g bower`

After installing Bower, run `bower install` in the ***root directory*** of the project and follow prompts that appear. 

## **Step 3**

Finally, we need to install all the ***backend dependencies*** that Lodestar needs in order to work. Projects like Lodestar can sometimes have MANY dependencies, so in order to make our job easier, we have all the dependencies in a list called `requirements.txt` located in the `server/` directory of the project. 

Navigate to `server/` and run `pip install -r requirements.txt` 

**Note**: If you are working on a Windows machine, you may need to install pip separately. To do so, visit [here](https://www.liquidweb.com/kb/install-pip-windows/) 

Now you are ready to run Lodestar on your machine!

To get it started do the following: 
1. From root directory, navigate to `server/src` and run `python run.py`
    * This starts the Flask server. This ***must*** be started *before* client side is run!

2. In a separate window, navigate to root directory and run `polymer serve` to run the front end. Right now, both windows must be running at the same time.

3. Copy the URL that appears after you run the front-end into a browser and now you should be running Lodestar!


To set up the Docker VM: 
- You need to set the IP address of the backend to 0.0.0.0:5000 
You need to be able to set up the front end to receive 0.0.0.0:5000 