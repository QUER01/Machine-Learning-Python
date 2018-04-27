# Start Script
print("")
print("-------------------------------------------")
print("--     Starting __init__.py       ")
print("-------------------------------------------")
print("")


# load libraries
import pickle

path  = "/home/osboxes/Documents/hackathon2017/"

# Set Variables
model_variables = (["model_type","naive_bayes"],
                   ["model_name","MyModel"],
                   ["feature_list",['title']],
                   ["predictor",['predictor']])

print("-------------------------------------------")
print("                Model Variables            ")
print("                                           ")
print("model_type            :" +  str(model_variables[0][1]))
print("model_name            :" +  str(model_variables[1][1]))
print("feature_list          :" +  str(model_variables[2][1]))
print("predictor             :" +  str(model_variables[3][1]))
print("-------------------------------------------")


# Store variables
# Saving the objects:
f = open(path + 'model_variables.pckl', 'wb')
pickle.dump(model_variables, f)
f.close()
