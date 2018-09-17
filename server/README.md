# linstt-offline-dispatch

This project aims to build a speech-to-text transcriber web service based on kaldi-offline-decoding.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

The project is divided into 3 modules:
- [worker_offline] is the module in charge of the ASR (automatic speech recognition).
- [master_server] is the webserver that provide the ASR service.
- [client] is a simple client meant to transcribe an audio file. 

### Prerequisites

#### Python 2.7
This project runs on python 2.7.
In order to run the [master_server] and the [client] you will need to install those python libraries: 
- tornado>=4.5.2
- ws4py

```
pip install ws4py 
pip install tornado
```
Or

```
pip install -r recquirements.txt
```
within the modules/server folder.

#### Kaldi model
The ASR server that will be setup here require kaldi model, note that the model is not included in the depository.
You must have this model on your machine. You must also check that the model have the specific files bellow :
- final.alimdl
- final.mat
- final.mdl
- splice_opts
- tree
- Graph/HCLG.fst
- Graph/disambig_tid.int
- Graph/num_pdfs
- Graph/phones.txt
- Graph/words.txt
- Graph/phones/*

#### Docker and Docker-compose
You must install docker on your machine. Refer to [docker doc](https://docs.docker.com/engine/installation) and
```
apt-get install docker
yoaourt -S docker
```
You must install docker-compose on your machine. Refer to [docker-compose](https://docs.docker.com/compose/install/)
```
curl -L https://github.com/docker/compose/releases/download/1.21.2/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

### Installing
You need to build the docker image first.
Go to modules/worker_offline and build the container.
```
cd modules/worker_offline
docker build -t linagora/stt-offline .
```

Or by docker-compose
You can build all docker image by using `docker-compose build` from the current folder It will automatically build all other docker.


## Running the tests

To run an automated test go to the test folder
``` 
cd tests
```
And run the test script: 
```
./deployement_test.sh <langageModelPath>
```
The test should display "Test succefull"
## Deployment

### Docker-Compose deployment

Here is the step for docker-compose

1. Configuration
Rename the .envdefault in .env then verify the data on .env file.
- MODEL_PATH : Path of the model to use
- OFFLINE_PORT : Port of the server (recommended : 8888)
- SWAGGER_PATH : Configuration folder for swagger to load (recommended : ./document)
- SWAGGER_JSON : Path where the docker-swagger will read the configuration file (recommended : /app/swagger/swagger.yml)

2. Build
Build the docker image
`docker-compose -f docker-compose.yml -f docker-optional-service.yml build`

3. Start
Run the LinStt-Service
`docker-compose up`
Run with documentation and swagger (localhost)
`docker-compose -f docker-compose.yml -f docker-optional-service.yml up`

### Docker Deployment

#### 1- Server
* Configure the server options by editing the server.conf file.
* Launch the server 

```
./master_server.py
``` 
 
#### 2- Worker
You can launch as many workers as you want on any machine that you want.
* Configure the worker by editing the server.conf file, provide the server IP adress ans server port.
* Launch the worker using the start_docker.sh command

```
cd modules/worker_offline
./start_docker.sh <langageModelPath>
```
For example if yout model is located at ~/speech/models/mymodel
With mymodel folder containing the following files:
- final.alimdl
- final.mat
- final.mdl
- splice_opts
- tree
- graphs/

```
cd modules/worker_offline
./start_docker.sh ~/speech/models/mymodel/
```

## Built With

* [tornado](http://www.tornadoweb.org/en/stable/index.html) - The web framework used
* [ws4py](https://ws4py.readthedocs.io/en/latest/) - WebSocket interfaces for python

## Authors

* **Abdelwahab Aheba** - *linstt-Offline-Decoding* - [Linagora](https://linagora.com/)
* **Rudy Baraglia** - *linstt-dispatch* - [Linagora](https://linagora.com/)


## License

See the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* The project has been vastly inspired by [Alumae](https://github.com/alumae)'s project [kaldi-gstreamer-server](https://github.com/alumae/kaldi-gstreamer-server) and use chunk of his code.

