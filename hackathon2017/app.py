import os
import pandas as pd
import time
import re
from flask import Flask, Response
from flask import url_for, redirect


app = Flask(__name__)

def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(1000)
    return rv


app.static_folder = 'static'

# Stream character and its index

@app.route('/')
def index():
    def generate():
        
        fake_sum = 0
        real_sum = 0
                # Change working directory
        path =os.getcwd()+'/'
        os.chdir(path)
        print(os.getcwd())
        #path  = "/run/media/manjaro/julian/99_Others/GitHub/hackathon2017/"        
        path = "/usr/src/app/"    
        data = pd.read_csv(path + 'data/raw/news.csv')    
        character=data[['title','description', 'url', 'prediction']].values
        #character=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O'] 
        for index, value in enumerate(character):
            
            rx = re.compile('\W+')
            value_full =  rx.sub(' ', str(value)).strip() 
            #value = value_full[:-0]
            prediction = value_full.strip()[-1]
            #time.sleep(20)
            print(prediction)
            
            
            if prediction == '1':
                prediction = 'Fake News'
                fake_sum = fake_sum + 1
            elif prediction == '0':
                prediction ='Real News'
                real_sum = real_sum + 1
            
            yield  prediction , value_full[:-1] ,fake_sum, real_sum
            
    return Response(stream_template('/production/index.html', data=generate()))
    
if __name__ == '__main__':
	app.run( 
        host="0.0.0.0",
        port=int("5000")
  )
