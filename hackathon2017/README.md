# Hackathon 2017
* Source Code my Blog post about Munich hackathon 2017 
* Visit my Blog : http://mydatascienceblog.ddns.net

![Alt text](/reports/figures/screenshot.png?raw=true "Hackathon 2017 Fake News Classifier app")

## Getting started

The dependencies for the project can be installed using

    $ sudo apt-get install docker
	$ sudo apt-get install docker-compose

You can use ``Docker`` to build a machine with a Python instance running

    $ sudo docker build -t hackathon2017:latest .
	
In order ro start the built docker image, you can type in the following command

	$ sudo docker run -it  -p 5000:5000 -v [your path]:/usr/src/app  hackathon2017 python app.py

To retrive the IP address you have to find the container ID and its corresponding IP address.
	
	$ sudo docker ps
	$ sudo docker inspect --format '{{ .NetworkSettings.IPAddress }}' IP
	
To stop and delete all your docker images use:
	
	$ # stop all containers
	$ docker stop $(docker ps -a -q)
	$ # Delete all containers
	$ docker rm $(docker ps -a -q)
	$ # Delete all images
	$ docker rmi $(docker images -q)



