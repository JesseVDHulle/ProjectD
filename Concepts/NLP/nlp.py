print("""
███    ██  █████  ████████ ██    ██ ██████   █████  ██          ██       █████  ███    ██  ██████  ██    ██  █████   ██████  ███████     ██████  ██████   ██████   ██████ ███████ ███████ ███████ ██ ███    ██  ██████  
████   ██ ██   ██    ██    ██    ██ ██   ██ ██   ██ ██          ██      ██   ██ ████   ██ ██       ██    ██ ██   ██ ██       ██          ██   ██ ██   ██ ██    ██ ██      ██      ██      ██      ██ ████   ██ ██       
██ ██  ██ ███████    ██    ██    ██ ██████  ███████ ██          ██      ███████ ██ ██  ██ ██   ███ ██    ██ ███████ ██   ███ █████       ██████  ██████  ██    ██ ██      █████   ███████ ███████ ██ ██ ██  ██ ██   ███ 
██  ██ ██ ██   ██    ██    ██    ██ ██   ██ ██   ██ ██          ██      ██   ██ ██  ██ ██ ██    ██ ██    ██ ██   ██ ██    ██ ██          ██      ██   ██ ██    ██ ██      ██           ██      ██ ██ ██  ██ ██ ██    ██ 
██   ████ ██   ██    ██     ██████  ██   ██ ██   ██ ███████     ███████ ██   ██ ██   ████  ██████   ██████  ██   ██  ██████  ███████     ██      ██   ██  ██████   ██████ ███████ ███████ ███████ ██ ██   ████  ██████  
                                                                                                                                                                                                                        
                                                                                                                                                                                                                        
""")

from keytotext import pipeline

nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")  #loading the pre-trained model
#params = {"do_sample":True, "num_beams":1, "no_repeat_ngram_size":3, "early_stopping":True, "max_new_tokens":100}    #decoding params

while True:
    keywords = []

    print("\nEnter keyword and then enter, to continue enter 'C' (Ctrl + C to Quit)")
    while True:
        t = input()
        if t.lower() != 'c':
            keywords.append(t)
        else:
            break

    print (nlp(keywords))#, **params))  #keywords