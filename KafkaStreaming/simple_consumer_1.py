#simple_consumer.py
from kafka import KafkaConsumer
from pyspark import SparkContext


# Create the connection to KAFKA topic
consumer = KafkaConsumer('simple-test',  
    bootstrap_servers="localhost:9091,localhost:9092")
 
for binary_message in consumer:
   # Message must be decoded again. Kafka can only sends binary data 
   print (binary_message.value.decode('utf-8'))
  
   #rdd = sc.parallelize(int(binary_message.value.decode('utf-8'))).cache()
   #print(rdd)



