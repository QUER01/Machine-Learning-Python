
#add Kafka user 

useradd kafka -m
passwd kafka 

adduser kafka sudo

su - kafka


# intsall java and zookeper
sudo apt-get update
sudo apt-get install default-jre
sudo apt-get install zookeeperd


# Test connection using telnet
telnet localhost 2181

# Download Kafka
mkdir -p ~/Downloads
wget "http://mirror.cc.columbia.edu/pub/software/apache/kafka/0.8.2.1/kafka_2.11-0.8.2.1.tgz" -O ~/Downloads/kafka.tgz

# Create Kafka directory and extract the tgz file
mkdir -p ~/kafka && cd ~/kafka
tar -xvzf ~/Downloads/kafka.tgz --strip 1

# modify .property file. Close and save bz pressing ESC and tzpe in :wq Add the following line of code 
#
#"delete.topic.enable = true"
#
vi ~/kafka/config/server.properties



# Copy server.properties, change port and broker.id and the log file folder
# create a new log file folder 
mkdir ~/kafka-log-2
~/kafka/bin/kafka-server-start.sh  ~/kafka/config/server2.properties

# install python library for kafka
pip install kafka-python







