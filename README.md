# Project folder structure
* **data**: contains the dataset and validation set, which are a split of the orginal given ratings. Also contains a sample submission. 
* **src**: code of the project for the individual models, for the ensemble learning method and utils.
* **report**: contains the report both in .tex and .pdf.
* **papers**: contains the papers .pdf that we used as references for our project.


# Running the code
First, you need to run every individual model. Doing this, will create a .npy file for base model, which is a complete user-item matrix with predictions for every entry.
Secondly, put all these .npy in a single folderm and specify the path of this folder in the Ensemble.py file for this method to combine the individual predictions.
Lastly, run the Ensemble.py, which will create a .csv file "EnsembleSubmission.csv". This is the file containing the final predictions that you wanto to upload to Kaggle. 

The models to be run are: 
-bayesianPMF.py
-KMeans.py
-NPCA.py
-RidgeSVD.py
-RSVD.py
-RunAutoEncoder.py
-Similarities.py

# Important notes
- If you don't want to run all of these models individually, you can ask us to provide you with all the completed .npy files from the individual models. We did not include them in this repository due to storage limitation issues.
- IOUtils.py is only used to provide IO functionalities to the models regarding the submission.
- CV.py is only used to split the dataset into a validation set and train set.
- AutoEncoder.py is the model in tensorflow for the autoencoder. It is used by the RunAutoEncoder.py script.
