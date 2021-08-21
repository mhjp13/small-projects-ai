import speech_recognition as sr

def main():
    r = sr.Recognizer()
    mic = sr.Microphone()

    print("Say something...")
    with mic as source:
        audio = r.listen(source)
        print("Processing...")
    
    print("You said:", r.recognize_google(audio))

if __name__ == '__main__':
    main()