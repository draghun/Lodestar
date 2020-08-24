# Lodestar Setup Instructions


## **Step 1**

Clone the repo:

`git clone https://gitlab.cs.umd.edu/leibatt/lodestar.git`

Before continuing, we have to download Node & npm: 

[For Windows](https://www.guru99.com/download-install-node-js.html)

[For Mac OS](https://treehouse.github.io/installation-guides/mac/node-mac.html)

[See here for more information on NPM](https://www.npmjs.com/get-npm)

**Note**: NPM version should be 10 

**Note**: In the future, we will want to use a Node Version Manager (also in link above)

We also want to install **Python 2.7**

**Mac OS**: Python 2.7 comes preinstalled and is the default version for Mac OS (if you have an older Mac machine you may want to check to see you have this installed)

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

Finally, we need to install all the ***backend dependencies*** that Lodestar needs in order to work. Projects like Lodestar can sometimes have MANY dependencies, so in order to make our job easier, we have all the dependencies in a list called `requirements.txt` located in the `/server` directory of the project. 

Navigate to `/server` and run `pip install -r requirements.txt` 

**Note**: If you are working on a Windows machine, you may need to install pip separately. To do so, visit [here](https://www.liquidweb.com/kb/install-pip-windows/) 

Now you are ready to run Lodestar on your machine!

To get it started do the following: 
1. From root directory, navigate to `server/src` and run `python run.py`
    * This starts the Flask server. This ***must*** be started *before* client side is run!

2.Navigate to root directory and run `polymer serve`

##### Lodestar should now be running on localhost!


To set up the Docker VM: 
- You need to set the IP address of the backend to 0.0.0.0:5000 
You need to be able to set up the front end to receive 0.0.0.0:5000 

# Lodestar
# Lodestar
