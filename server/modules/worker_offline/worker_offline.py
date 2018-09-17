#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:10:23 2018
Updated on Wed Sep 12 15:18:00 2018

@authors: {rbaraglia, irebai}@linagora.com
"""

import os
import argparse
import threading
import logging
import json
import subprocess
import configparser
import re
import tenacity
import base64
from signal_trimming import *
import noise_reduce as nr

from ws4py.client.threadedclient import WebSocketClient

#LOADING CONFIGURATION
worker_settings = configparser.ConfigParser()
worker_settings.read('worker.cfg')
SERVER_IP = worker_settings.get('server_params', 'server_ip')
SERVER_PORT = worker_settings.get('server_params', 'server_port')
SERVER_TARGET = worker_settings.get('server_params', 'server_target')
TEMP_FILE_PATH = worker_settings.get('worker_params', 'temp_file_location')
INDICE_DATA = True if worker_settings.get('worker_params', 'confidence_score') == 'true' else False
NUM_JOBS = worker_settings.get('worker_params', 'number_jobs')
NUM_THREADS = worker_settings.get('worker_params', 'number_threads')


#Signal Processing
PREPROCESSING = True if worker_settings.get('signal_processing', 'preprocessing') == 'true' else False
NOISE = True if worker_settings.get('signal_processing', 'noise') == 'true' else False
SILENCE = True if worker_settings.get('signal_processing', 'silence') == 'true' else False
SILENCE_METHOD = worker_settings.get('signal_processing', 'silence_method')
NOISE_METHOD = worker_settings.get('signal_processing', 'noise_method')


#Decoder parameters applied for both GMM and DNN based ASR systems
decoder_settings = configparser.ConfigParser()
decoder_settings.read('systems/models/decode.cfg')
DECODER_SYS = decoder_settings.get('decoder_params', 'decoder')
DECODER_MAXACT = decoder_settings.get('decoder_params', 'max_active')
DECODER_BEAM = decoder_settings.get('decoder_params', 'beam')
DECODER_LATBEAM = decoder_settings.get('decoder_params', 'lattice_beam')
DECODER_ACWT = decoder_settings.get('decoder_params', 'acwt')
DECODER_IVEC = "true" if decoder_settings.get('decoder_params', 'ivector') == 'true' else "false"
if decoder_settings.get('decoder_params','type') == '':
   DECODER_TYPE = 'none'
else:
   DECODER_TYPE = decoder_settings.get('decoder_params', 'type')
DECODER_MFCC = decoder_settings.get('decoder_params', 'mfcc_config')
DECODER_VAD = decoder_settings.get('decoder_params', 'vad_config')


if "OFFLINE_PORT" in os.environ:
    SERVER_PORT = os.environ['OFFLINE_PORT']

class NoRouteException(Exception):
    pass
class ConnexionRefusedException(Exception):
    pass

class WorkerWebSocket(WebSocketClient):
    def __init__(self, uri):
        WebSocketClient.__init__(self, url=uri, heartbeat_freq=10)

    def opened(self):
        pass
    def guard_timeout(self):
        pass
    def received_message(self, m):
        try:
            json_msg = json.loads(str(m))
        except:
            logging.debug("Message received: %s" % str(m))
        else:
            if 'uuid' in json_msg.keys():
                self.client_uuid = json_msg['uuid']
                self.fileName = self.client_uuid.replace('-', '')
                self.file = base64.b64decode(json_msg['file'])
                self.filepath = TEMP_FILE_PATH+self.fileName+'.wav'
                with open(self.filepath, 'wb') as f:
                    f.write(self.file)
                logging.debug("FileName received: %s" % self.fileName)
                # preprocessing
                if PREPROCESSING:
                    logging.debug("Signal Processing")
                    if NOISE:
                        logging.debug("noise parameter activated. Method: "+NOISE_METHOD)
                        if NOISE_METHOD == 'noise_reduce':
                             nr.noise_reduce(self.filepath,self.filepath,'centroid_s','false')
                        elif NOISE_METHOD == 'rnnoise':
                             subprocess.call("./rnnoise.sh "+self.fileName+".wav ", shell=True)
                        else:
                             logging.debug("not recognized noise method")
                    else:
                        logging.debug("sox processing")
                        subprocess.call("sox "+TEMP_FILE_PATH+self.fileName+".wav -t wav -r 16000 -c 1 "+TEMP_FILE_PATH+self.fileName+"_tmp.wav; mv "+TEMP_FILE_PATH+self.fileName+"_tmp.wav "+TEMP_FILE_PATH+self.fileName+".wav", shell=True);

                    if SILENCE:
                        logging.debug("silence parameter activated. Method: "+SILENCE_METHOD)
                        if SILENCE_METHOD == 'signal_trimming':
                             trim_silence_segments(self.filepath,self.filepath, chunk_size=120, threshold_factor=0.85, side_effect_accomodation=2)
                        elif SILENCE_METHOD == 'noise_reduce':
                             nr.noise_reduce(self.filepath,self.filepath,'','True')
                        else:
                             logging.debug("not recognized silence method")
                else:
                    logging.debug("sox processing")
                    subprocess.call("sox "+TEMP_FILE_PATH+self.fileName+".wav -t wav -r 16000 -c 1 "+TEMP_FILE_PATH+self.fileName+"_tmp.wav; mv "+TEMP_FILE_PATH+self.fileName+"_tmp.wav "+TEMP_FILE_PATH+self.fileName+".wav", shell=True);

                # Offline decoder call
                logging.debug("Offline decoder call")
                subprocess.call("cd scripts; ./decode.sh --mfcc_config "+DECODER_MFCC+" --vad-config "+DECODER_VAD+" --ivector "+DECODER_IVEC+" --type "+DECODER_TYPE+" ../systems/models "+self.fileName+".wav "+str(INDICE_DATA)+" "+DECODER_MAXACT+" "+DECODER_BEAM+" "+DECODER_LATBEAM+" "+DECODER_ACWT+" "+DECODER_SYS+" "+NUM_JOBS+" "+NUM_THREADS, shell=True)

                # Check result
                if os.path.isfile('trans/decode_'+self.fileName+'.log'):
                    if INDICE_DATA:
                        with open('trans/decode_'+self.fileName+'.log', 'r', encoding='utf-8') as resultFile:
                            result = resultFile.read().strip()
                            logging.debug("Transcription with indice data is : %s" % result)
                            self.send_result(result)
                    else:
                        with open('trans/decode_'+self.fileName+'.log', 'r', encoding='utf-8') as resultFile:
                            result = resultFile.read().strip()
                            logging.debug("Transcription without indice data is: %s", result)
                            self.send_result(result)
                else:
                    logging.error("Worker Failed to create transcription file")
                    self.send_result("")

                # Delete temporary files
                for file in os.listdir(TEMP_FILE_PATH):
                    os.remove(TEMP_FILE_PATH+file)

    def post(self, m):
        logging.debug('POST received')

    def send_result(self, result=""):
        logging.debug("Transcription without indice data is: %s", result)
        msg = ""
        try:
            data = json.loads(result)
            if(data['message']==""):
                if INDICE_DATA:
                    msg = json.dumps({u'uuid': self.client_uuid, u'transcription':data['utterance'], u'avg_confedence_per_word':data['cw'], u'standard_deviation':data['std']})
                else:
                    msg = json.dumps({u'uuid': self.client_uuid, u'transcription':data['utterance'], u'confidence_score':'Desactivated'})
            else:
                msg = json.dumps({u'uuid': self.client_uuid, u'transcription':data['utterance'], u'message':data['message']})

        except Exception as e:
            msg = json.dumps({u'uuid': self.client_uuid, u'ERROR':'SYSTEM ERROR!!!!!'})
        self.client_uuid = None
        self.send(msg)

    def send_error(self, message):
        msg = json.dumps({u'uuid': self.client_uuid, u'error':message})
        self.send(msg)

    def closed(self, code, reason=None):
        pass

    def finish_request(self):
        pass

@tenacity.retry(
        wait=tenacity.wait.wait_fixed(2),
        stop=tenacity.stop.stop_after_delay(45),
        retry=tenacity.retry_if_exception(ConnexionRefusedException)
    )
def connect_to_server(ws):
    try:
        logging.info("Attempting to connect to server at %s:%s" % (SERVER_IP, SERVER_PORT))
        ws.connect()
        logging.info("Worker succefully connected to server at %s:%s" % (SERVER_IP, SERVER_PORT))
        ws.run_forever()
    except KeyboardInterrupt:
        logging.info("Worker interrupted by user")
        ws.close()
    except Exception as e:
        if "[Errno 113]" in str(e):
            logging.info("Failed to connect")
            raise NoRouteException
        if "[Errno 111]" in str(e):
            logging.info("Failed to connect")
            raise ConnexionRefusedException
        logging.debug(e)
    logging.info("Worker stopped")

def main():
    parser = argparse.ArgumentParser(description='Worker for linstt-dispatch')
    parser.add_argument('-u', '--uri', default="ws://"+SERVER_IP+":"+SERVER_PORT+SERVER_TARGET, dest="uri", help="Server<-->worker websocket URI")

    args = parser.parse_args()
    #thread.start_new_thread(loop.run, ())
    if not os.path.isdir(TEMP_FILE_PATH):
        os.mkdir(TEMP_FILE_PATH)

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)8s %(asctime)s %(message)s ")
    logging.info('Starting up worker')
    ws = WorkerWebSocket(args.uri)
    try:
        connect_to_server(ws)
    except Exception:
        logging.error("Worker did not manage to connect to server at %s:%s" % (SERVER_IP, SERVER_PORT))
if __name__ == '__main__':
    main()
