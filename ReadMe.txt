# CatBoostPerSegment


This application allows you to train and test your own CatBoostPerSegment model. You can configure your own dataset, tune data transformations, select model hyper-parameters and evaluate model performance according to chosen metrics and forecast visualizations.


## Prerequisites
Before you get started, you're going to need a few things:
- Pyhton 3.8 - Pyhton 3.10
- PIP
- conda
- macOS


## Step 1: Create new virtual environment


conda create --name name_you_like


## Step 2: Activate your new virtual environment


conda activate name_you_like


## Step 3: Upgrade pip


pip install --upgrade pip


## Step 4: Install the requirements
Use the cd command followed by the path to the root directory of your project. Replace <your-path> with your actual project path.


cd <your-path>


The required packages for this project are specified in the requirements.txt file which is in the root directory of your project. To install these packages, use the pip install -r command followed by the name of the requirements file. 


pip install -r requirements.txt


You may need to reactivate your virtual environment or start a new terminal session to ensure that changes to the environment, (newly installed packages) are properly recognized. 


## Step 5: Build your installer 
Run the following commands using terminal:


briefcase build
briefcase package --no-sign


After this, the dist folder in the project directory will contain CatBoostSeries.dmg. Drag the app file into the application file and you are ready to go.
IMPORTANT: When you start application for the first time after installation, for some reasons it may be needed to start your application several times before it starts working properly, probably due to some peculiarities of operational system or packaging manager.