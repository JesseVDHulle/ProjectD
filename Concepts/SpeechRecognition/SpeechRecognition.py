#imports
import speech_recognition as sr
import keyboard

#listener
_listener = sr.Recognizer()

#listen function
def listen():
    try:
        with sr.Microphone() as source:
            print("Listening")   
            voice = _listener.listen(source, phrase_time_limit = 5)
            command = str(_listener.recognize_google(voice))           
            print(command)
            if command != None:
                return command.lower()
            else:
                return ""
    except sr.UnknownValueError:
        pass
    except Exception as e:
        raise e

#MainLoop
while True:
    if keyboard.is_pressed('end'):
        answer = listen()
        if "exit" in answer:
            break