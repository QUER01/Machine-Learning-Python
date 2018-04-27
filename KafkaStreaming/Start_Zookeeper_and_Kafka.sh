~/kafka/bin/zookeeper-server-start.sh ~/kafka/config/zookeeper.properties 
~/kafka/bin/kafka-server-start.sh  ~/kafka/config/server.properties 

# Create some topics
~/kafka/bin/kafka-topics.sh --zookeeper localhost:2181 --create --topic simple-test2 --partitions 2 --replication-factor 2
#~/kafka/bin/kafka-topics.sh --zookeeper localhost:2181 --create --topic online --partitions 2 --replication-factor 2
 


# Describe the topics
#~/kafka/bin/kafka-topics.sh --zookeeper localhost:2181 --describe -- topic online
~/kafka/bin/kafka-topics.sh --zookeeper localhost:2181 --describe --topic simple-test2 



# start python producer and consumer scripts
	
python ~/Syn-Repo-DataScience/Python/KAFKA_Streaming/simple_consumer_1.py	
python ~/Syn-Repo-DataScience/Python/KAFKA_Streaming/simple_producer_1.py



