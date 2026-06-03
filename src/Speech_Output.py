import subprocess
class Speaker:
    def speak(self,text):
        subprocess.run(["say", text])