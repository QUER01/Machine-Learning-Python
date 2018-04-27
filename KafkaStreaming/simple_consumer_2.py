#online_consumer.py
import numpy as np
from sklearn.linear_model import SGDClassifier
from kafka import KafkaConsumer
import json
import plotly
import plotly.tools 
import plotly.graph_objs as go
from scipy import stats

#############################################
# Create a plotly plot
#############################################

with open('/home/ventum/Syn-Repo-DataScience/Python/KAFKA_Streaming/config2.json') as config_file:
    plotly_user_config = json.load(config_file)

username = plotly_user_config['plotly_username']
api_key = plotly_user_config['plotly_api_key']
stream_token1 = plotly_user_config['plotly_streaming_tokens'][1]
stream_token2 = plotly_user_config['plotly_streaming_tokens'][2]
#stream_token = dict(token=stream_token, maxpoints=60)

print("Starting Streaming")

#############################################
# Initialize your plotly object
#############################################

p = plotly.plotly.sign_in(username, api_key)

x_data          = [] 
y_data_fit      = [] 
target      = []
prediction  = []
line  = []
line_history  = []
n_features = 1
count = 0
correct = 0


trace0 = go.Scatter(
    x = [],
    y = [],
    mode = 'markers',
    name = 'train data points'
    
)
trace1 = go.Scatter(
    x = [],
    y = [],
    mode = 'markers',
    name = 'trend line'
)
data = [trace0,trace1]
plotly.plotly.iplot(data,filename='Support-Vector-Machine-Kafka', fileopt='overwrite')

#plotly.plotly.iplot([{'x': [], 'y': [], 'type': 'scatter', 'mode': 'markers','stream': {'token': stream_token1, 'maxpoints': 100}},
#                     {'x': [], 'y': [], 'type': 'scatter', 'mode': 'line','stream': {'token': stream_token2, 'maxpoints': 100}}],  
#                     filename='Support-Vector-Machine-Kafka', fileopt='overwrite')

#############################################
# Open plotly stream
#############################################

s1 = plotly.plotly.Stream(stream_token1)
s1.open()

s2 = plotly.plotly.Stream(stream_token2)
s2.open()


#############################################
# Open Kafka connection and do incremental SVM fit
############################################# 
 
clf = SGDClassifier(loss="hinge", alpha=0.01, n_iter=20, fit_intercept=True)
 
consumer = KafkaConsumer('simple-test2',  
    bootstrap_servers="localhost:9091,localhost:9092")
 



for binary_message in consumer:

    #############################################
    # encode binary value back to the message
    #############################################
    msg=binary_message.value.decode('utf-8')
    
    #############################################
    # create a json file
    #############################################
    
    data = json.loads(msg)
    vec = np.zeros(len(data))
    for key,value in data.items():
        vec[int(key)]=value
    
    #############################################    
    # create a feature vector
    #############################################    
    
    features = vec[:n_features].reshape(1,n_features)
    
    #############################################    
    # Create a target vector
    #############################################
    
    target = [vec[n_features]]
    
    # if the stream is initiated we need to create an initial value count
    if count > 1:
        prediction = clf.predict(features)
        prediction =  np.array(prediction)
        correct += int(prediction[0]==int(target[0]))
    if count > 2:
        x_data.extend(features[0])
        y_data_fit.extend(prediction)
        
        #############################################
        # Generated linear fit
        #############################################

        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data,y_data_fit)
        line = slope*float(x_data[0])+intercept
        line_history.append(float(line))
        print("Score           :   " + str(correct/float(count-1)))
        print("Line of best fit:   " + str(line_history))
        print("xdata[0]        :   " + str(float(x_data[0])))
        print("xdata           :   " + str(x_data))
        print("prediction      :   " + str(prediction))
        print("features        :   " + str(features))
    #############################################
    # Fit the Support vector machine model (Stochastic Gradient Descent)
    #############################################
        
    clf.partial_fit(features, target,classes=[0,1])

    

    
    #############################################
    # Write the data to the plotly stream    
    #############################################
    
    s1.write({'x': x_data, 'y': target[0]})
    s2.write({'x': x_data, 'y': line_history})

    # increment the count
    count += 1

    
    
