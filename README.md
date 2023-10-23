tokenization.py:
    clean the text
    substitute hashtags
    substitute number
    substitute urls
    substitute mentions

    word tokenizations handled

language_model.py:
1. All the perplexities of LM1,LM2,LM3,LM4 are already stored
command to run:
    python3 language_model.py k input_path
    python3 language_model.py w input_path

input:
    sentence
output:
    probability of sentence

Neural_language model:
1. First the train dataset is trained on google colab 
using torch.save(model.state_dict)
model_1.pth, model_2.pth are the models obtained
then produced model is reused by state dictionary model
2. model is trained on 10 epochs
3. For inference get the perplexities for each train, test dataset of pre-judice and uslyssus

Link for Report and datasets:
https://drive.google.com/drive/folders/1-yJJZUzo-aKBsHXAjiJOfdzNQJ8qzTAZ?usp=sharing