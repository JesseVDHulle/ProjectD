# print("""
# ███    ██  █████  ████████ ██    ██ ██████   █████  ██          ██       █████  ███    ██  ██████  ██    ██  █████   ██████  ███████     ██████  ██████   ██████   ██████ ███████ ███████ ███████ ██ ███    ██  ██████  
# ████   ██ ██   ██    ██    ██    ██ ██   ██ ██   ██ ██          ██      ██   ██ ████   ██ ██       ██    ██ ██   ██ ██       ██          ██   ██ ██   ██ ██    ██ ██      ██      ██      ██      ██ ████   ██ ██       
# ██ ██  ██ ███████    ██    ██    ██ ██████  ███████ ██          ██      ███████ ██ ██  ██ ██   ███ ██    ██ ███████ ██   ███ █████       ██████  ██████  ██    ██ ██      █████   ███████ ███████ ██ ██ ██  ██ ██   ███ 
# ██  ██ ██ ██   ██    ██    ██    ██ ██   ██ ██   ██ ██          ██      ██   ██ ██  ██ ██ ██    ██ ██    ██ ██   ██ ██    ██ ██          ██      ██   ██ ██    ██ ██      ██           ██      ██ ██ ██  ██ ██ ██    ██ 
# ██   ████ ██   ██    ██     ██████  ██   ██ ██   ██ ███████     ███████ ██   ██ ██   ████  ██████   ██████  ██   ██  ██████  ███████     ██      ██   ██  ██████   ██████ ███████ ███████ ███████ ██ ██   ████  ██████  
                                                                                                                                                                                                                        
                                                                                                                                                                                                                        
# """)

from keytotext import pipeline
import os

nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")  #loading the pre-trained model
#params = {"do_sample":True, "num_beams":1, "no_repeat_ngram_size":3, "early_stopping":True, "max_new_tokens":100}    #decoding params
os.system('cls')
while True:
    keywords = []

    print("\nEnter keyword and then enter, to continue enter ' ' (Ctrl + C to Quit)")
    while True:
        t = input()
        if t.lower() != ' ':
            keywords.append(t)
        else:
            break

    print (nlp(keywords))#, **params))  #keywords