As you go through this project, please keep in mind that it is currently a work in progress and not production ready. 

For this project, a Gradient Boosting machine learning model was created (XGBoost) and trained using Canada's public sector salary information. 
The model was then deployed using Flask (HTML + CSS) with the goal to allow users to new feature information into the model to output a predicted salary. 

The model has been trained and is working as expected. 

There are still a few updates to workout for the deployment step: 
- Need to fix the unicode error.
- To train the model, we convert categorical variables into numeric values. The Flask app currently asks for these inputs in their original numeric form. 
This needs to be updated to allow users to input the categorical values via a dropdown menu and have the app convert these values into numeric on the backend.
- Visualization updates via CSS.
