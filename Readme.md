# TwilBERT

TwilBERT is an adaptation of the BERT model trained with spanish tweets. This repository contains and code instructions for generating the dockerfile with an ELG compliant API. The original code can be found [here](https://github.com/jogonba2/TWilBert).

# Installation
## Initial Setup

Execute *setup_project.sh* at least once as some files are copied to their target path.

```
bash setup_project.sh
```

## Manual steps

### To finetune a model

Download the weights of the large model https://drive.google.com/open?id=14BW8JoyrWkOOAHw6tLCOlh0tgrd8wIUh and position them in the *weights/bert_large_twitter* folder of the original Twilbert repository

Optionally you can download the weights for the base model: https://drive.google.com/open?id=1ztTQBvPhuy7SFi1uHrLCCOZCirubAt-1


## Dockerfile

### Build

```
bash docker-build.sh
```

### Execute


#### With docker
```
docker run --rm -p 0.0.0.0:8866:8866 --name twilbert elg_twilbert
```

#### Locally

Simply launch the flask API (inside the TWilBert folder)

```
python3 serve.py
```

_NOTE: You must execute this command in the environment with *python3.6* and all the dependencies (e.g. the one created with the script *setup_porject.sh*)._

Verify the API works as expected.

```
curl -X POST  http://0.0.0.0:8866/predict_json -H 'Content-Type: application/json' -d '{"type": "text", "content":"Este es un texto de prueba"}'
```


## Test the system

```
python3 -u twilbert/applications/bert/test.py configs/microservs/config_large_server.json
```

## Finetuning for HateEval 2019

We are going to fine tuning the model for HateEval 2019 (Spanish) https://aclanthology.org/S19-2007/ (base model: skip this step if base weights were not downloaded)

```
python3 twilbert/applications/bert/single_finetune.py configs/microservs/config_single_hateeval19_base.json 
```

and the large model that will be finally used

```
python3 twilbert/applications/bert/single_finetune.py configs/microservs/config_single_hateeval19_large.json 
```

### Testing the finetuned model

Try the following script to test the finetuned model (base model: skip this step if base weights were not downloaded):

```
python3 -u twilbert/applications/bert/single_labeling.py configs/microservs/config_labelling_single_hateeval19_base.json 
```

and the same for the large model

```
python3 -u twilbert/applications/bert/single_labeling.py configs/microservs/config_labelling_single_hateeval19_large.json
```

# Test API (ELG Format)

In the folder `test` you have the files for testing the API according to the ELG specifications.

It uses an API that acts as a proxy with your dockerized API that checks both the requests and the responses.
For this process, follow the instructions:

1) Configure the .env file with the data of the image and your API
2) Launch the test: `docker-compose up`
3) Make the requests, instead of to your API's endpoint, to the test's endpoint:
   ```
      curl -X POST  http://0.0.0.0:8866/processText/service -H 'Content-Type: application/json' -d '{"type": "text", "content":"Este es un texto de prueba"}'
   ```
4) If your request and the API's response is compliance with the ELG API, you will receive the response.
   1) If the request is incorrect: Probably you will don't have a response and the test tool will not show any message in logs.
   2) If the response is incorrect: You will see in the logs that the request is proxied to your API, that it answers, but the test tool does not accept that response. You must analyze the logs.

## Citations
The original work of TwilBERT is presented in the following paper:

- Gonzalez, Jose Angel, Lluís-F. Hurtado, and Ferran Pla. "TWilBert: Pre-trained deep bidirectional transformers for Spanish Twitter." Neurocomputing 426 (2021): 58-69.

The license of the original work is BSD 4-Clause License.
