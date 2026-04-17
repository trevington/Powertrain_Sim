import numpy as np

print("Hello world!")

class msg:
    def __init__(self, message):
        self.message = message

message = msg("hello world!")

print(message.message)