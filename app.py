import sys
import json
import torch
import numpy as np
import torchvision.models as models

from utils import (
    add_handler,
    download_image,
    init_logger,
    preprocess_image,
    model_prediction,
    number_output,
)

# Open labels
with open("model/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# # Load pretrained model
PATH = "model/mobilenetv2.pth"

mobilenet_v2 = models.mobilenet_v2()
mobilenet_v2.load_state_dict(torch.load(PATH))
mobilenet_v2.eval()

catdogList = ["tabby", "Egyptian cat", "tiger cat", "Siamese cat", "Persian cat", "cougar", "lynx", "leopard", 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah', 
              'Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 
              'Afghan hound', 'basset', 'beagle','bloodhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound', 'English foxhound', 'redbone', 'borzoi', 'Irish wolfhound', 
              'Italian greyhound', 'whippet', 'Ibizan hound', 'Norwegian elkhound', 'otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 
              'Staffordshire bullterrier', 'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 
              'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 
              'Sealyham terrier', 'Airedale', 'cairn', 'Australian terrier', 'Dandie Dinmont', 'Boston bull', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'silky terrier', 'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'clumber', 'English springer', 'Welsh springer spaniel', 'cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog', 'Shetland sheepdog', 'collie', 'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'malamute', 'Siberian husky', 'dalmatian', 'affenpinscher', 
              'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed, Samoyede', 'Pomeranian', 'chow', 'keeshond', 'Brabancon griffon', 
              'Pembroke', 'Cardigan', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf', 'white wolf', 
              'red wolf', 'coyote', 'dingo', 'dhole', 'African hunting dog']

def lambda_handler(event, context):
    # Retrieve inputs
    input_url, n_predictions = event["input_url"], event["n_predictions"]

    # # Download image
    input_image = download_image(input_url)

    # # Process input image
    batch = preprocess_image(input_image)

    # # Generate prediction
    pred = model_prediction(input_batch=batch, mdl=mobilenet_v2)

    # # Top n results
    n_results = number_output(mdl_output=pred, mdl_labels=labels, top_n=n_predictions)

    # prediction = model.predict(url)
    response = {"statusCode": 200, "body": json.dumps(n_results)}

        # Just need top result:
    isCatOrDog = False
    if n_results[0][0] in catdogList:
      isCatOrDog = True
      
    print("Found Cat or Dog? " + str(isCatOrDog))

    return response
