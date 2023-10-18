__all__ = [
    'Eti'
]

class Eti(object):

    def __init__(self, message:str='hi'):
        self.self = self
        self.message = message

    def get_message(self):
        print(self.message)